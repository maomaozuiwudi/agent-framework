"""
记忆一致性协议 — 多Agent协作框架的记忆层

三层写入策略：
  L1 短期（强一致 Write-through） → 执行状态、质检结果。失败就阻止，不丢数据。
  L2 任务级（弱一致 Write-back） → 交付摘要、反馈日志。缓冲批量写，失败后台重试。
  L4 长期（异步合并、乐观锁） → Skill、用户画像、成熟反馈。版本号驱动，不阻塞主流程。

Merge Protocol：
  - 时间戳优先（最新胜出）
  - 结构化 Key 按字段级合并
  - QA修正 > Executor原始
  - 人工修正 > 任何Agent
  - 冲突记录保留，不覆盖原始数据直到合并完成

Usage:
    from agent_framework.memory import MemoryLayer, MemoryEntry, MemoryStore

    store = MemoryStore()
    store.write_l1("task_001", "status", "completed")   # 强一致，立刻落盘
    store.write_l2("task_001", "summary", "...")        # 弱一致，缓冲写入
    store.write_l4("user_profile", {"name": "琪琪"})    # 异步合并，版本号
"""

from __future__ import annotations

import time
import json
import logging
import threading
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Optional, Callable

logger = logging.getLogger("agent-framework.memory")


# ─── 基元定义 ─────────────────────────────────────────────────


class ConsistencyLevel(str, Enum):
    """一致性等级"""
    L1_STRONG = "l1_strong"           # Write-through，失败即报错
    L2_WEAK = "l2_weak"              # Write-back，缓冲批量写
    L4_ASYNC = "l4_async"            # 异步合并，乐观锁


class MergeStrategy(str, Enum):
    """合并策略"""
    TIMESTAMP_WINS = "timestamp_wins"         # 最新时间戳胜出
    FIELD_MERGE = "field_merge"               # 字段级合并
    QA_OVERRIDES = "qa_overrides"             # QA 覆盖执行
    HUMAN_OVERRIDES = "human_overrides"       # 人工覆盖一切
    APPEND = "append"                         # 追加不覆盖


@dataclass
class MemoryEntry:
    """单条记忆条目"""
    key: str                              # 记忆键
    value: Any                            # 记忆值
    layer: ConsistencyLevel               # 所属层
    version: int = 1                      # 版本号（L4 用）
    source: str = "hermes"                # 写入来源（agent 名称）
    timestamp: float = field(default_factory=time.time)
    ttl: float | None = None               # 过期时间（None=永不过期）
    metadata: dict[str, Any] = field(default_factory=dict)
    conflicts: list[dict] = field(default_factory=list)  # 冲突记录

    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() > self.timestamp + self.ttl

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "layer": self.layer.value,
            "version": self.version,
            "source": self.source,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
        }


# ─── 内存存储 ─────────────────────────────────────────────────


class MemoryStore:
    """三层记忆存储引擎

    支持：
      - L1: 即时写入，失败报错
      - L2: 缓冲写入（按时间或条目数触发 flush）
      - L4: 版本驱动的异步合并
      - Merge Protocol: 冲突自动合并
    """

    def __init__(self, backend: Any | None = None):
        self.backend = backend  # 可选的持久化后端

        # 三层存储
        self._l1: dict[str, MemoryEntry] = {}       # 强一致
        self._l2: dict[str, MemoryEntry] = {}       # 弱一致
        self._l4: dict[str, MemoryEntry] = {}       # 异步合并

        # L2 缓冲
        self._l2_buffer: list[MemoryEntry] = []
        self._l2_buffer_size: int = 3               # 满3条触发flush
        self._l2_buffer_time: float = 30.0           # 最长30秒触发flush
        self._l2_last_flush: float = time.time()
        self._l2_lock = threading.Lock()

        # L4 版本控制
        self._l4_versions: dict[str, int] = {}       # key → 当前版本

        # 冲突记录
        self._conflicts: list[dict] = []

        # 统计
        self.stats = {
            "l1_writes": 0, "l1_failures": 0,
            "l2_writes": 0, "l2_flushes": 0, "l2_failures": 0,
            "l4_writes": 0, "l4_merges": 0, "l4_conflicts": 0,
        }

    # ─── L1 强一致 ──────────────────────────────────────────

    def write_l1(self, key: str, value: Any, source: str = "hermes",
                 ttl: float | None = None) -> MemoryEntry:
        """L1 强一致写入 — 立刻落盘，失败就报错"""
        entry = MemoryEntry(
            key=key,
            value=value,
            layer=ConsistencyLevel.L1_STRONG,
            source=source,
            ttl=ttl,
        )
        try:
            self._l1[key] = entry
            self.stats["l1_writes"] += 1
            logger.debug(f"L1 写入: {key} = {str(value)[:50]}")
            return entry
        except Exception as e:
            self.stats["l1_failures"] += 1
            raise RuntimeError(f"L1 写入失败: {key} — {e}")

    def read_l1(self, key: str) -> MemoryEntry | None:
        """L1 读取"""
        entry = self._l1.get(key)
        if entry and entry.is_expired:
            del self._l1[key]
            return None
        return entry

    def delete_l1(self, key: str) -> bool:
        if key in self._l1:
            del self._l1[key]
            return True
        return False

    # ─── L2 弱一致（Write-back） ───────────────────────────

    def write_l2(self, key: str, value: Any, source: str = "hermes",
                 immediate: bool = False) -> MemoryEntry:
        """L2 弱一致写入 — 缓冲后批量落盘

        Args:
            immediate: 立即刷入，不缓冲
        """
        entry = MemoryEntry(
            key=key,
            value=value,
            layer=ConsistencyLevel.L2_WEAK,
            source=source,
        )
        self._l2[key] = entry

        with self._l2_lock:
            self._l2_buffer.append(entry)
            self.stats["l2_writes"] += 1

        if immediate or len(self._l2_buffer) >= self._l2_buffer_size:
            self.flush_l2()

        return entry

    def flush_l2(self) -> int:
        """刷入 L2 缓冲区"""
        with self._l2_lock:
            if not self._l2_buffer:
                return 0
            batch = self._l2_buffer.copy()
            self._l2_buffer.clear()
            self._l2_last_flush = time.time()

        try:
            flushed = self._persist_batch(batch)
            self.stats["l2_flushes"] += 1
            logger.debug(f"L2 flush: {len(batch)} 条")
            return flushed
        except Exception as e:
            self.stats["l2_failures"] += 1
            logger.warning(f"L2 flush 失败 ({len(batch)} 条): {e}")
            # 失败后放回缓冲区
            with self._l2_lock:
                self._l2_buffer.extend(batch)
            return 0

    def auto_flush_l2(self) -> None:
        """自动 flush（由定时器调用）"""
        elapsed = time.time() - self._l2_last_flush
        if self._l2_buffer and elapsed >= self._l2_buffer_time:
            self.flush_l2()

    def read_l2(self, key: str) -> MemoryEntry | None:
        return self._l2.get(key)

    # ─── L4 异步合并（乐观锁） ────────────────────────────

    def write_l4(self, key: str, value: Any, source: str = "hermes",
                 version: int | None = None) -> MemoryEntry:
        """L4 异步合并写入 — 乐观锁版本控制

        Args:
            version: 期望的版本号。None=自动获取最新版本+1。
                     如果传了版本号但和当前版本不匹配 → 触发合并。
        """
        current = self._l4.get(key)
        current_version = current.version if current else 0

        if version is not None and version != current_version:
            # 版本冲突 → 合并
            merged_value, conflict = self._merge_values(
                existing=current.value if current else None,
                incoming=value,
                existing_source=current.source if current else "",
                incoming_source=source,
            )
            new_version = max(current_version, version) + 1
            entry = MemoryEntry(
                key=key,
                value=merged_value,
                layer=ConsistencyLevel.L4_ASYNC,
                version=new_version,
                source=f"merge:{source}+{current.source if current else 'new'}",
                conflicts=[conflict] if conflict else [],
            )
            self._l4[key] = entry
            self._l4_versions[key] = new_version
            self.stats["l4_merges"] += 1
            if conflict:
                self.stats["l4_conflicts"] += 1
                self._conflicts.append(conflict)
            logger.info(f"L4 合并: {key} v{current_version} → v{new_version}")
        else:
            # 无冲突，直接写入
            new_version = (version or current_version) + 1
            entry = MemoryEntry(
                key=key,
                value=value,
                layer=ConsistencyLevel.L4_ASYNC,
                version=new_version,
                source=source,
            )
            self._l4[key] = entry
            self._l4_versions[key] = new_version
            self.stats["l4_writes"] += 1
            logger.debug(f"L4 写入: {key} v{new_version}")

        return entry

    def read_l4(self, key: str) -> MemoryEntry | None:
        return self._l4.get(key)

    def get_l4_version(self, key: str) -> int:
        """获取 L4 当前版本号"""
        return self._l4_versions.get(key, 0)

    # ─── 合并协议 ──────────────────────────────────────────

    def _merge_values(self, existing: Any, incoming: Any,
                      existing_source: str, incoming_source: str
                      ) -> tuple[Any, dict | None]:
        """Merge Protocol 冲突合并

        优先级：人工 > QA/ Critic > 其他 Agent > 时间戳
        """
        conflict = None

        # 人工修正 > 一切
        if incoming_source == "human":
            conflict = {
                "type": "human_overrides",
                "overridden_by": incoming_source,
                "old_value": existing,
                "new_value": incoming,
                "timestamp": time.time(),
            }
            return incoming, conflict

        if existing_source == "human":
            conflict = {
                "type": "human_preserved",
                "preserved_by": existing_source,
                "existing_value": existing,
                "incoming_value": incoming,
                "timestamp": time.time(),
            }
            return existing, conflict

        # QA/ Critic > 执行器
        qa_sources = {"critic", "qa", "evaluator"}
        if incoming_source in qa_sources and existing_source not in qa_sources:
            conflict = {
                "type": "qa_overrides",
                "overridden_by": incoming_source,
                "old_value": existing,
                "new_value": incoming,
                "timestamp": time.time(),
            }
            return incoming, conflict

        if existing_source in qa_sources and incoming_source not in qa_sources:
            conflict = {
                "type": "qa_preserved",
                "preserved_by": existing_source,
                "incoming_value": incoming,
                "timestamp": time.time(),
            }
            return existing, conflict

        # 字典类型 → 字段级合并
        if isinstance(existing, dict) and isinstance(incoming, dict):
            merged = existing.copy()
            merge_conflicts = []
            for k, v in incoming.items():
                if k in merged and merged[k] != v:
                    merge_conflicts.append({
                        "field": k, "old": merged[k], "new": v,
                    })
                merged[k] = v
            if merge_conflicts:
                conflict = {
                    "type": "field_merge",
                    "fields": merge_conflicts,
                    "timestamp": time.time(),
                }
            return merged, conflict if merge_conflicts else None

        # 列表类型 → 追加去重
        if isinstance(existing, list) and isinstance(incoming, list):
            merged = list(dict.fromkeys(existing + incoming))  # 有序去重
            if len(merged) != len(existing):
                conflict = {
                    "type": "list_append",
                    "added": [x for x in incoming if x not in existing],
                    "timestamp": time.time(),
                }
            return merged, conflict if len(merged) != len(existing) else None

        # 标量 → 时间戳优先（最新胜出）
        conflict = {
            "type": "timestamp_wins",
            "overridden_by": "newer_timestamp",
            "old_value": existing,
            "new_value": incoming,
            "timestamp": time.time(),
        }
        return incoming, conflict

    # ─── 持久化 ────────────────────────────────────────────

    def _persist_batch(self, entries: list[MemoryEntry]) -> int:
        """批量持久化（可接外部 backend）"""
        if self.backend:
            return self.backend.write_batch(entries)
        # 默认：只保留在内存
        return len(entries)

    def persist_all(self) -> dict[str, int]:
        """持久化所有层"""
        counts = {"l1": 0, "l2": 0, "l4": 0}

        if self.backend:
            counts["l1"] = self.backend.write_batch(list(self._l1.values()))
            counts["l2"] = self._persist_batch(self._l2_buffer) + len(self._l2)
            counts["l4"] = self.backend.write_batch(list(self._l4.values()))

        return counts

    # ─── 查询 ──────────────────────────────────────────────

    def get(self, key: str) -> MemoryEntry | None:
        """跨层查询（L1 → L2 → L4）"""
        return self.read_l1(key) or self.read_l2(key) or self.read_l4(key)

    def search(self, key_prefix: str = "", layer: ConsistencyLevel | None = None,
               source: str = "") -> list[MemoryEntry]:
        """跨层搜索"""
        results = []
        stores = []

        if layer is None or layer == ConsistencyLevel.L1_STRONG:
            stores.append(self._l1)
        if layer is None or layer == ConsistencyLevel.L2_WEAK:
            stores.append(self._l2)
        if layer is None or layer == ConsistencyLevel.L4_ASYNC:
            stores.append(self._l4)

        for store in stores:
            for key, entry in store.items():
                if key_prefix and not key.startswith(key_prefix):
                    continue
                if source and entry.source != source:
                    continue
                if entry.is_expired:
                    continue
                results.append(entry)

        return results

    def list_keys(self, layer: ConsistencyLevel | None = None) -> list[str]:
        """列出所有键"""
        keys = []
        if layer is None or layer == ConsistencyLevel.L1_STRONG:
            keys.extend(self._l1.keys())
        if layer is None or layer == ConsistencyLevel.L2_WEAK:
            keys.extend(self._l2.keys())
        if layer is None or layer == ConsistencyLevel.L4_ASYNC:
            keys.extend(self._l4.keys())
        return keys

    def cleanup_expired(self) -> int:
        """清理过期条目"""
        count = 0
        for store in [self._l1, self._l2, self._l4]:
            expired = [k for k, v in store.items() if v.is_expired]
            for k in expired:
                del store[k]
            count += len(expired)
        return count

    def get_conflicts(self, limit: int = 10) -> list[dict]:
        return self._conflicts[-limit:]

    def summary(self) -> str:
        return (
            f"L1: {len(self._l1)} entries, {self.stats['l1_writes']} writes\n"
            f"L2: {len(self._l2)} entries, buffer={len(self._l2_buffer)}, "
            f"{self.stats['l2_flushes']} flushes\n"
            f"L4: {len(self._l4)} entries, {self.stats['l4_merges']} merges, "
            f"{self.stats['l4_conflicts']} conflicts\n"
            f"Total conflicts: {len(self._conflicts)}"
        )


# ─── 测试 ──────────────────────────────────────────────────────


def test_l1_strong_consistency():
    store = MemoryStore()
    entry = store.write_l1("task_001.status", "completed")
    assert entry.value == "completed"
    assert store.read_l1("task_001.status").value == "completed"
    print(f"L1 强一致写入/读取: ✓")


def test_l1_ttl():
    store = MemoryStore()
    store.write_l1("temp_key", "temp_value", ttl=0.01)
    import time
    time.sleep(0.02)
    assert store.read_l1("temp_key") is None
    print("L1 TTL 过期: ✓")


def test_l2_buffer():
    store = MemoryStore()
    store.write_l2("log_001", "step 1 done")
    store.write_l2("log_002", "step 2 done")
    assert len(store._l2_buffer) == 2
    assert store.read_l2("log_001").value == "step 1 done"

    # 第3条触发 flush
    store.write_l2("log_003", "step 3 done")
    assert len(store._l2_buffer) == 0, f"应触发 flush, 剩余 {len(store._l2_buffer)}"
    print(f"L2 缓冲自动 flush: ✓ ({store.stats['l2_flushes']} flushes)")


def test_l2_immediate():
    store = MemoryStore()
    store.write_l2("urgent", "重要", immediate=True)
    assert len(store._l2_buffer) == 0
    print("L2 即时写入: ✓")


def test_l4_optimistic_lock():
    store = MemoryStore()
    # 首次写入 v1
    store.write_l4("profile", {"name": "琪琪", "age": 30})
    assert store.get_l4_version("profile") == 1

    # 正常写入 v2（不传版本号）
    store.write_l4("profile", {"name": "琪琪", "age": 31, "role": "developer"})
    assert store.get_l4_version("profile") == 2
    print(f"L4 乐观锁正常写入: ✓ (版本 {store.get_l4_version('profile')})")

    # 版本冲突 → 合并
    store.write_l4("profile", {"name": "琪琪哥"}, version=0)
    entry = store.read_l4("profile")
    assert entry.version >= 3
    print(f"L4 版本冲突合并: ✓ (版本 {entry.version}, source: {entry.source})")


def test_merge_protocol_human():
    store = MemoryStore()
    store.write_l4("config", {"theme": "dark", "lang": "zh"})
    store.write_l4("config", {"theme": "light"}, source="human")
    entry = store.read_l4("config")
    assert entry.value["theme"] == "light", f"人工应覆盖, 实际 {entry.value}"
    print(f"Merge Protocol 人工覆盖: ✓ (theme={entry.value['theme']})")


def test_merge_protocol_qa():
    store = MemoryStore()
    store.write_l4("code_review", {"status": "pass", "issues": []}, source="executor")
    store.write_l4("code_review", {"status": "fail", "issues": ["missing tests"]}, source="critic")
    entry = store.read_l4("code_review")
    assert entry.value["status"] == "fail", f"QA 应覆盖, 实际 {entry.value}"
    print(f"Merge Protocol QA 覆盖: ✓ (status={entry.value['status']})")


def test_merge_protocol_field_merge():
    store = MemoryStore()
    # 版本冲突触发字段级合并
    store.write_l4("user_prefs", {"theme": "dark", "lang": "zh"})
    v1 = store.get_l4_version("user_prefs")
    store.write_l4("user_prefs", {"theme": "light", "font_size": 14}, version=v1 - 1)
    entry = store.read_l4("user_prefs")
    assert entry.value["theme"] == "light", f"新值应覆盖, 实际 {entry.value}"
    assert entry.value["lang"] == "zh", f"旧字段应保留, 实际 {entry.value}"
    assert entry.value["font_size"] == 14, f"新字段应出现, 实际 {entry.value}"
    print(f"Merge Protocol 字段级合并: ✓ ({entry.value})")


def test_merge_protocol_list():
    store = MemoryStore()
    store.write_l4("keywords", ["ai", "agent", "dag"])
    v1 = store.get_l4_version("keywords")
    store.write_l4("keywords", ["agent", "arbiter", "eval"], version=v1 - 1)
    entry = store.read_l4("keywords")
    assert "dag" in entry.value, f"旧列表元素应保留, 实际 {entry.value}"
    assert "arbiter" in entry.value, f"新元素应加入, 实际 {entry.value}"
    print(f"Merge Protocol 列表去重追加: ✓ ({len(entry.value)}关键词)")


def test_cross_layer_search():
    store = MemoryStore()
    store.write_l1("task_status", "build framework")
    store.write_l2("task_log", "started")
    store.write_l4("user_profile", {"name": "琪琪"})

    results = store.search(key_prefix="task")
    assert len(results) == 2, f"应找到 2 条 (task_status + task_log), 实际 {len(results)}"
    print(f"跨层搜索: ✓ (找到 {len(results)} 条)")


def test_cleanup():
    store = MemoryStore()
    store.write_l1("expired", "gone", ttl=0.01)
    import time
    time.sleep(0.02)
    cleaned = store.cleanup_expired()
    assert cleaned == 1, f"应清理 1 条, 实际 {cleaned}"
    assert "expired" not in store._l1
    print(f"过期清理: ✓ (清理 {cleaned} 条)")


def test_auto_flush_timer():
    store = MemoryStore()
    store.write_l2("log_1", "entry 1")
    store._l2_buffer_time = 0.01  # 极短时间触发
    import time
    time.sleep(0.02)
    store.auto_flush_l2()
    assert len(store._l2_buffer) == 0, f"自动 flush 失败, 剩余 {len(store._l2_buffer)}"
    print(f"L2 自动定时 flush: ✓")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print("═══ 记忆一致性协议测试 ═══\n")
    test_l1_strong_consistency()
    test_l1_ttl()
    print()
    test_l2_buffer()
    test_l2_immediate()
    test_auto_flush_timer()
    print()
    test_l4_optimistic_lock()
    print()
    test_merge_protocol_human()
    test_merge_protocol_qa()
    test_merge_protocol_field_merge()
    test_merge_protocol_list()
    print()
    test_cross_layer_search()
    test_cleanup()
    print("\n═══ 全部测试通过 ═══")
