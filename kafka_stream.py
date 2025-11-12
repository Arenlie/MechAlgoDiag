# -*- coding: utf-8 -*-
"""
Kafka 简单消费者封装。
"""
import time
import base64
import json
from typing import Dict, Any, Iterable, List, Optional
from datetime import datetime

from kafka import KafkaConsumer, TopicPartition


def parse_payload(b: Optional[bytes]) -> Optional[Dict[str, Any]]:
    if b is None:
        return None
    try:
        s = b.decode("utf-8")
        s_stripped = s.strip()
        try:
            obj = json.loads(s_stripped)
            return {"format": "json", "json": obj}
        except Exception:
            return {"format": "utf8", "text": s}
    except UnicodeDecodeError:
        return {"format": "base64", "base64": base64.b64encode(b).decode("ascii")}


def payload_preview(payload: Optional[Dict[str, Any]], max_len: int = 200) -> str:
    if payload is None:
        return "None"
    fmt = payload.get("format")
    if fmt == "json":
        txt = json.dumps(payload["json"], ensure_ascii=False, separators=(",", ":"))
    elif fmt == "utf8":
        txt = payload["text"]
    else:
        b64 = payload.get("base64", "")
        return f"<base64 len={len(b64)}>"
    txt = txt.replace("\n", "\\n")
    return txt[:max_len] + ("…" if len(txt) > max_len else "")


def normalize_record(msg) -> Dict[str, Any]:
    key_payload = parse_payload(msg.key)
    val_payload = parse_payload(msg.value)
    return {
        "topic": msg.topic,
        "partition": msg.partition,
        "offset": msg.offset,
        "timestamp_ms": msg.timestamp,
        "ts_readable": datetime.fromtimestamp(msg.timestamp / 1000.0).isoformat() if msg.timestamp else None,
        "key": key_payload,
        "value": val_payload,
        "headers": [{"key": k, "value": parse_payload(v)} for (k, v) in (msg.headers or [])],
    }


class KafkaSimpleConsumer:
    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        from_beginning: bool = False,
        poll_timeout_ms: int = 500,
        security_protocol: str = "PLAINTEXT",
        consumer_timeout_ms: int = 0,
        logger=None,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.from_beginning = from_beginning
        self.poll_timeout_ms = poll_timeout_ms
        self.security_protocol = security_protocol
        self.consumer_timeout_ms = consumer_timeout_ms
        self.logger = logger
        self.consumer: Optional[KafkaConsumer] = None
        self.tps: List[TopicPartition] = []

    def _log(self, level: str, msg: str):
        if self.logger:
            getattr(self.logger, level)(msg)
        else:
            print(f"[{level.upper()}] {msg}")

    def __enter__(self):
        self.consumer = KafkaConsumer(
            bootstrap_servers=self.bootstrap_servers,
            group_id=None,
            enable_auto_commit=False,
            auto_offset_reset="earliest" if self.from_beginning else "latest",
            security_protocol=self.security_protocol,
            consumer_timeout_ms=self.consumer_timeout_ms,
        )
        parts = None
        for _ in range(30):
            parts = self.consumer.partitions_for_topic(self.topic)
            if parts:
                break
            time.sleep(1)
        if not parts:
            raise RuntimeError(f"获取主题分区失败，主题可能不存在或不可达：{self.topic}")
        self.tps = [TopicPartition(self.topic, p) for p in sorted(parts)]
        self.consumer.assign(self.tps)
        if self.from_beginning:
            self.consumer.seek_to_beginning(*self.tps)
        else:
            self.consumer.seek_to_end(*self.tps)
        self._log("info", f"Kafka connected. topic={self.topic} partitions={sorted(parts)} "
                          f"mode={'earliest' if self.from_beginning else 'latest'}")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.consumer:
                self.consumer.close()
        finally:
            self._log("info", "Kafka consumer closed.")

    def iter_records(self) -> Iterable[Dict[str, Any]]:
        assert self.consumer is not None, "Kafka consumer not initialized. Use 'with' context."
        while True:
            records_map = self.consumer.poll(timeout_ms=self.poll_timeout_ms)
            if not records_map:
                continue
            for _, records in records_map.items():
                for msg in records:
                    yield normalize_record(msg)
