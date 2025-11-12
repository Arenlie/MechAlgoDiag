# -*- coding: utf-8 -*-

"""
kafka信息中byteValues的解析
"""
from typing import Optional, Dict, Any, Tuple, Union
import base64
import numpy as np
from datetime import datetime


def _to_bytes(byte_values: Union[str, bytes, bytearray, memoryview]) -> bytes:
    """
    将输入统一转成 bytes：
      - 若是 str：视为 base64 字符串，先去空白再 b64decode
      - 若是 bytes/bytearray/memoryview：直接转 bytes
    """
    if byte_values is None:
        raise ValueError("byteValues 为空")
    if isinstance(byte_values, (bytes, bytearray, memoryview)):
        return bytes(byte_values)
    # 其余情况按 base64 字符串处理，允许混入空白/换行
    s = "".join(str(byte_values).split())
    return base64.b64decode(s)


def decode_wave_from_kafka(
    byte_values: Union[str, bytes, bytearray, memoryview],
    sample_rate: Optional[int],
    data_time_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """
    解码 Kafka 的 byteValues，返回诊断可用的结构。
    解析规则：固定按 float32 小端读取（np.frombuffer(..., dtype=np.float32)）。
    """
    if sample_rate is None:
        raise ValueError("sample_rate 不能为空")

    # 1) 统一为 bytes
    raw_bytes = _to_bytes(byte_values)
    raw_len = len(raw_bytes)

    # 2) 按需求指定的方式解析：float32 frombuffer → Python float 列表
    wave_data = [float(value) for value in list(np.frombuffer(raw_bytes, dtype=np.float32))]

    # 3) 转为 float64（便于后续数值计算），并应用缩放
    vals = np.asarray(wave_data, dtype=np.float64)

    # 4) 构造时间轴（相对起点秒）
    sr = float(sample_rate)
    t = np.arange(vals.size, dtype=np.float64) / sr

    # 5) data_time（ms） -> 可读字符串
    ts_str = None
    if data_time_ms is not None:
        ts_str = datetime.fromtimestamp(int(data_time_ms) / 1000.0).strftime("%Y-%m-%d %H:%M:%S")

    # 6) 输出
    out = {
        "values": vals,                   # np.ndarray float64
        "times": t,                       # np.ndarray float64（秒）
        "data_time_str": ts_str,          # 可读时间
    }
    return out
