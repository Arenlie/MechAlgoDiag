# -*- coding: utf-8 -*-
"""
实时主流程（写死配置）：
- Kafka 拉流（equipNo/pointNo/kpild/byteValues/dataTime/sampleRate/...）
- 监测规则表(Excel/CSV) 设备、测点、数据项三键 AND 命中后，解码 byteValues -> 机理诊断
- 将“机理算法返回的 list[dict]” + “状态机判断结果” 追加保存到 CSV
"""

import os
import csv
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Iterable
from pathlib import Path
import pandas as pd
from kafka_stream import KafkaSimpleConsumer, payload_preview
from bytewave_decode import decode_wave_from_kafka
from model import Speed_Estimate_algorithm
from model_diagnosis import model_diagnosis

# ========= 写死配置 =========
KAFKA_BOOTSTRAP = "192.168.1.252:9092"
KAFKA_TOPIC = "rwadata"
FROM_BEGINNING = False

# 规则表（xlsx/xls/csv），表头为中文默认格式，不允许修改
RULE_TABLE_PATH = r"excel/white_list.xlsx"

# 输出与日志
OUTPUT_DIR = "kafka_realtime_diag"
CSV_PATH = os.path.join(OUTPUT_DIR, "realtime_diag.csv")
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "realtime_diag.log")
PRINT_PREVIEW_MAX = 200
Key = Tuple[str, str, str]


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ========= 日志 =========
ensure_dir(LOG_DIR)
logger = logging.getLogger("realtime")
logger.setLevel(logging.INFO)
fh = RotatingFileHandler(LOG_FILE, maxBytes=10_000_000, backupCount=3, encoding="utf-8")
fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s"))
logger.addHandler(fh)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
logger.addHandler(sh)


# ========= 监测规则表（严格三键 AND 匹配 + 取阈值/转速）=========
def _norm_key(x: Any) -> str:
    return "" if x is None else str(x).strip().lower()


def _to_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


class RuleTable:
    COL_EQUIP_AREA = "设备区域"
    COL_EQUIP_NO = "设备编码"
    COL_EQUIP_NAME = "设备名称"
    COL_POINT_NAME = "测点（点位）名称"
    COL_KPID = "波形数据编码"

    COL_WORK_SPEED = "工作转速"
    COL_NOTICE_TH = "注意阈值"
    COL_WARN_TH = "警告阈值"

    COL_SAMPLE_RATE = "抽样频率（Hz）"  # 可作为采样率兜底

    def __init__(self, path: str, logger: logging.Logger):
        self.logger = logger
        self.path = path
        self.index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if not self.path:
            self.logger.warning("未配置规则表路径，默认不过滤。")
            return

        self.logger.info(f"加载规则表：{self.path}")
        if self.path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(self.path)
        elif self.path.lower().endswith(".csv"):
            df = pd.read_csv(self.path)
        else:
            raise ValueError("规则表仅支持 xlsx/xls/csv")

        for col in [self.COL_EQUIP_NO, self.COL_POINT_NAME, self.COL_KPID]:
            if col not in df.columns:
                raise ValueError(f"规则表缺少关键列：{col}")

        cnt = 0
        for _, row in df.iterrows():
            equip_no = _norm_key(row.get(self.COL_EQUIP_NO))
            point_name = _norm_key(row.get(self.COL_POINT_NAME))
            kpild = _norm_key(row.get(self.COL_KPID))
            if not equip_no or not point_name or not kpild:
                continue

            info = {
                "equip_area": str(row.get(self.COL_EQUIP_AREA, "")),
                "equip_no": str(row.get(self.COL_EQUIP_NO, "")),
                "equip_name": str(row.get(self.COL_EQUIP_NAME, "")),
                "point_name": str(row.get(self.COL_POINT_NAME, "")),
                "kpild": str(row.get(self.COL_KPID, "")),
                "work_speed": _to_float(row.get(self.COL_WORK_SPEED)),
                "notice_threshold": _to_float(row.get(self.COL_NOTICE_TH)),
                "warn_threshold": _to_float(row.get(self.COL_WARN_TH)),
                "sample_rate_rule": _to_float(row.get(self.COL_SAMPLE_RATE)),
            }
            self.index[(equip_no, point_name, kpild)] = info
            cnt += 1

        self.logger.info(f"规则表索引完成：{cnt} 条三键组合。")

    def match(self, equip_no: str, point_name: str, kpild: str) -> Optional[Dict[str, Any]]:
        key = (_norm_key(equip_no), _norm_key(point_name), _norm_key(kpild))
        return self.index.get(key)


# === 新增：报警状态管理器 =========================================
class AlarmStateManager:
    """按 (equip_no, point_no, kpild) 维护报警状态：
       - 未报警：first/latest 均为空
       - 报警中：记录首次(first)与最新(latest)报警时间
    """

    def __init__(self, logger, persist_path: Optional[str] = None, bootstrap_keys: Optional[Iterable[Key]] = None):
        self.logger = logger
        self.persist_path = persist_path
        self.state: Dict[str, Dict[str, Optional[str]]] = {}

        # 先从磁盘恢复（如有）
        if persist_path and Path(persist_path).exists():
            try:
                self.state = json.loads(Path(persist_path).read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"读取报警状态文件失败：{e}")

        # 再按规则表预置为未报警
        if bootstrap_keys:
            for k in bootstrap_keys:
                sk = self._k2s(k)
                self.state.setdefault(sk, {"status": "未报警", "first": None, "latest": None})

        self._persist_safe()

    def _k2s(self, k: Key) -> str:
        return f"{k[0]}|{k[1]}|{k[2]}"

    def _persist_safe(self):
        if not self.persist_path:
            return
        try:
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.persist_path).write_text(json.dumps(self.state, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            self.logger.warning(f"保存报警状态失败：{e}")

    def update(self, k: Key, has_alarm: bool, ts_str: str) -> Dict[str, Optional[str]]:
        """根据本次是否报警更新状态；返回该 key 的最新记录"""
        rec = self.state.setdefault(self._k2s(k), {"status": "未报警", "first": None, "latest": None})
        if has_alarm:
            if rec["status"] != "报警中" or not rec["first"]:
                rec["first"] = ts_str
            rec["latest"] = ts_str
            rec["status"] = "报警中"
        else:
            # 清除首次报警时间与最新报警时间，回到未报警
            rec["status"] = "未报警"
            rec["first"] = None
            rec["latest"] = None
        self._persist_safe()
        return rec

    def get(self, k: Key) -> Dict[str, Optional[str]]:
        return self.state.get(self._k2s(k), {"status": "未报警", "first": None, "latest": None})


def try_bootstrap_keys_from_rule(rule) -> Iterable[Key]:
    """
    尝试从 RuleTable 提取所有 (equip_no, point_no, kpild) 组合用于初始化。
    兼容几种常见实现；若无法提取，则返回空，转为懒初始化。
    """
    # 方案1：类里自带迭代方法
    for attr in ("iter_keys", "all_keys"):
        if hasattr(rule, attr) and callable(getattr(rule, attr)):
            return list(getattr(rule, attr)())

    # 方案2：常见属性 rows / data / df（尽量不引入 pandas 依赖）
    keys = set()
    for attr in ("rows", "data"):
        if hasattr(rule, attr):
            for r in getattr(rule, attr) or []:
                try:
                    keys.add((str(r.get("equip_no", "")),
                              str(r.get("point_no", "")),
                              str(r.get("kpild", ""))))
                except Exception:
                    pass
    return list(keys)


def has_alarm_from_diag(diag) -> bool:
    """
    统一判断本次是否报警：
    - diag 为 None / 空：无报警
    - 有任意项 alarm_level 为 "注意" 或 "警告"，则视为报警
    """
    if not diag:
        return False

    for item in diag:
        if isinstance(item, dict):
            lvl = item.get("alarm_level", item.get("level"))
            if lvl in ("注意", "警告"):
                return True
    return False


# ========= 工具 =========
def safe_get(d: Dict[str, Any], key: str, default=None):
    try:
        return d.get(key, default)
    except Exception:
        return default


def ts_ms_to_str(ts_ms: Optional[int]) -> str:
    try:
        return datetime.fromtimestamp(int(ts_ms) / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    轻量扁平化：只拍平一层嵌套；列表/复杂对象转 JSON 字符串。
    """
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            for k2, v2 in v.items():
                kk = f"{key}.{k2}"
                out[kk] = json.dumps(v2, ensure_ascii=False) if isinstance(v2, (list, dict)) else v2
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(v, ensure_ascii=False)
        else:
            out[key] = v
    return out


def compute_conclusion(diag: Dict[str, Any]) -> str:
    """
    生成诊断结论文本：
    - 若已有 'conclusion'/'diagnosis'/'result_text' 列，直接使用
    - 否则如含 warnLevel，映射成中文等级文本
    - 否则回退到 'result'/'label'/'fault' 等字段
    """
    for k in ("conclusion", "diagnosis", "result_text"):
        if k in diag and diag[k]:
            return str(diag[k])

    lvl = diag.get("warnLevel")
    if lvl is not None:
        try:
            lvl = int(lvl)
            mapping = {0: "正常", 1: "注意", 2: "警告", 3: "严重"}
            tag = mapping.get(lvl, f"等级{lvl}")
            return f"状态：{tag}"
        except Exception:
            pass

    for k in ("result", "label", "fault", "status"):
        if k in diag and diag[k]:
            return str(diag[k])

    return "未定义"


def append_row_to_csv(row: Dict[str, Any], csv_path: str, logger: logging.Logger):
    """
    兼容“列动态增长”的 CSV 追加：
    - 若文件不存在：创建并写入表头
    - 若存在且新列为旧列子集：直接追加
    - 若存在但出现新列：读全量CSV + 合并列 + 回写
    """
    ensure_dir(os.path.dirname(csv_path))
    if not os.path.exists(csv_path):
        # 初次创建
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    # 文件存在：检查列
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            header = []

    old_cols = list(header)
    new_cols = list(row.keys())

    if set(new_cols).issubset(set(old_cols)):
        # 直接追加
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=old_cols, extrasaction="ignore")
            # 填充缺失列为空
            safe_row = {k: row.get(k, "") for k in old_cols}
            writer.writerow(safe_row)
    else:
        # 出现新列：用 pandas 合并写回
        try:
            df_old = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
        except Exception:
            df_old = pd.DataFrame(columns=old_cols)

        df_new = pd.DataFrame([{k: row.get(k, "") for k in new_cols}])
        # 新列并集
        cols_union = list(dict.fromkeys(old_cols + [c for c in new_cols if c not in old_cols]))
        df_old = df_old.reindex(columns=cols_union)
        df_new = df_new.reindex(columns=cols_union)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")


# ========= 主流程 =========
def main():
    try:
        ensure_dir(OUTPUT_DIR)
        # 规则表
        rule = RuleTable(RULE_TABLE_PATH, logger)

        logger.info(f"CSV 输出路径：{CSV_PATH}")

        # === NEW: 初始化报警状态表（预置全部为“未报警”），并启用持久化 ===
        ALARM_STATE_JSON = os.path.join(OUTPUT_DIR, "alarm_state.json")
        bootstrap_keys = try_bootstrap_keys_from_rule(rule)
        alarm_mgr = AlarmStateManager(
            logger=logger,
            persist_path=ALARM_STATE_JSON,  # 不需要持久化可改为 None
            bootstrap_keys=bootstrap_keys
        )
        logger.info(f"报警状态表就绪：{len(alarm_mgr.state)} 个监测项，默认=未报警")

        with KafkaSimpleConsumer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                topic=KAFKA_TOPIC,
                from_beginning=FROM_BEGINNING,
                poll_timeout_ms=500,
                logger=logger,
        ) as stream:

            for record in stream.iter_records():
                preview = payload_preview(record.get("value"), max_len=PRINT_PREVIEW_MAX)
                logger.info(f"p{record['partition']}@{record['offset']} {record.get('ts_readable')} value={preview}")

                val = record.get("value") or {}
                if val.get("format") != "json":
                    logger.warning("消息不是 JSON，跳过。")
                    continue
                data = val["json"]
                if not isinstance(data, dict):
                    logger.warning("JSON 格式异常，跳过。")
                    continue

                # === 业务字段 ===
                equip_no = data.get("equipNo", "")
                point_no = data.get("pointNo", "")
                kpild = data.get("kpild", "")
                byte_values = data.get("byteValues", None)
                data_time = data.get("dataTime", None)  # ms
                sample_rate = data.get("sampleRate", None)  # Hz
                data_len = data.get("dataLen", None)

                # === 三键 AND 匹配 ===
                logger.info(f"收到kafka数据，正在匹配规则表：设备编码={equip_no}，测点={point_no}，数据项={kpild}")
                hit = rule.match(equip_no, point_no, kpild)
                if not hit:
                    logger.info(f"该数据未在监测范围内：设备编码={equip_no}，测点={point_no}，数据项={kpild}")
                    continue
                logger.info(f"匹配数据成功，开始解码数据：设备编码={equip_no}，测点={point_no}，数据项={kpild}")
                # === 解码 byteValues ===
                if byte_values is None or not byte_values:
                    logger.warning("byteValues 为空，无数据，跳过。")
                    continue

                # 采样率优先用 Kafka，否则回落规则表
                sr = None
                if sample_rate is not None:
                    try:
                        sr = int(sample_rate)
                    except Exception:
                        sr = None
                if sr is None and hit.get("sample_rate_rule") is not None:
                    try:
                        sr = int(hit["sample_rate_rule"])
                    except Exception:
                        sr = None
                if sr is None:
                    logger.error("缺少 sampleRate（Kafka 与规则表均未提供），无法解码。")
                    continue

                try:
                    decoded = decode_wave_from_kafka(
                        byte_values=byte_values,
                        sample_rate=sr,
                        data_time_ms=int(data_time) if data_time is not None else None,
                    )
                    # 机理参数也放到顶层，便于 diagnosis_result 使用
                    if hit.get("work_speed") is not None:
                        decoded["work_speed"] = hit.get("work_speed")
                    else:
                        decoded["work_speed"] = Speed_Estimate_algorithm(decoded["values"], sr)
                    decoded["notice_threshold"] = hit.get("notice_threshold")
                    decoded["warn_threshold"] = hit.get("warn_threshold")
                    logger.info(f"数据解码成功，采样率={sr}，数据长度={data_len}")
                except Exception as e:
                    logger.error(f"byteValues 解`码失败：{e}")
                    continue

                # === 机理模型诊断 ===
                try:
                    logger.info(
                        f"开始进行机理模型诊断，数据长度={len(decoded['values'])}，采样率={sr}，工作速度={decoded['work_speed']}")
                    diag = model_diagnosis(decoded["values"], fs=sr, fr=decoded["work_speed"],
                                           notice_th=decoded["notice_threshold"], warn_th=decoded["warn_threshold"])
                    if diag is None or not diag:
                        logger.info("诊断结果显示无故障，跳过。")
                        continue
                except Exception as e:
                    logger.error(f"模型诊断失败：{e}")
                    continue

                # === 先更新状态，再决定是否写 CSV ===
                start_time_str = decoded.get("data_time_str") or ts_ms_to_str(data_time)
                key_triplet = (
                    hit.get("equip_no") or equip_no,
                    hit.get("point_no") or point_no,
                    hit.get("kpild") or kpild
                )
                is_alarm = has_alarm_from_diag(diag)
                state_rec = alarm_mgr.update(key_triplet, is_alarm, start_time_str)

                if not is_alarm:
                    # 无报警：状态机已自动清空 first/latest；无需写 CSV
                    logger.info(
                        f"监测到未报警：设备={key_triplet[0]} 测点={key_triplet[1]} 数据项={key_triplet[2]}；状态已重置为未报警")
                    continue

                # 有报警：需要写 CSV，时间来自状态机（首次/最新） ===
                if not diag:
                    # 理论上 is_alarm 为 True 时，diag 应有内容；稳妥起见做一次兜底
                    diag = []

                for L in diag:
                    if isinstance(L, dict):
                        # i += 1
                        row = {
                            "设备区域": hit.get("equip_area"),
                            "设备编码": hit.get("equip_no"),
                            "设备名称": hit.get("equip_name"),
                            "报警描述": L.get("alarm_message"),
                            "预警等级": L.get("alarm_level"),
                            "首次报警时间": state_rec.get("first"),  # 首次报警时间
                            "最新报警时间": state_rec.get("latest"),  # 持续报警的最新时间
                            "报警状态": "未处理",
                        }

                        # === 追加写入 CSV ===
                        try:
                            append_row_to_csv(row, CSV_PATH, logger)
                            logger.info(f"CSV 已写入 1 行 -> {CSV_PATH}")
                        except Exception as e:
                            logger.error(f"写入 CSV 失败：{e}")

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        logger.error(f"主循环异常：{e}")


if __name__ == "__main__":
    main()

