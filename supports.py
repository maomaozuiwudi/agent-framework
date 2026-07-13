"""
外围支撑系统 — 多Agent协作框架的段8十二项机制实现

包含：
  8.10 负载均衡+优先级队列 — TaskQueue / LoadBalancer
  8.11 故障熔断 — CircuitBreaker
  8.8  缓存冷热分层 — TieredCache
  8.4  成本预算监控 — BudgetMonitor
  8.6  反馈/指标 — FeedbackFilter + KPITracker
  8.3  故障恢复 — RecoveryAgent

所有组件独立于核心框架，可按需加载。
"""

from __future__ import annotations

import time
import json
import logging
import threading
from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta

logger = logging.getLogger("agent-framework.supports")


# ═══════════════════════════════════════════════════════════
# 8.10 负载均衡 + 优先级队列
# ═══════════════════════════════════════════════════════════


class Priority(IntEnum):
    CRITICAL = 0   # 安全/对外发布 — 立即执行
    HIGH = 1       # 内部重要任务
    NORMAL = 2     # 常规任务
    LOW = 3        # 实验性/可延迟


@dataclass
class QueueItem:
    """队列中的任务项"""
    task_id: str
    goal: str
    priority: Priority = Priority.NORMAL
    agent_type: str = "hermes"
    cost_estimate: int = 0
    created_at: float = field(default_factory=time.time)
    timeout: int = 300
    callback: Callable | None = None
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "goal": self.goal[:50],
            "priority": self.priority.name,
            "agent": self.agent_type,
            "cost": self.cost_estimate,
            "created_at": self.created_at,
            "timeout": self.timeout,
        }


class TaskQueue:
    """优先级任务队列 — 支持优先级排序、并发上限、超时

    策略：
      - 按优先级出队（CRITICAL > HIGH > NORMAL > LOW）
      - 同优先级 FIFO
      - 并发上限（默认每个 agent 类型 3）
      - 超时自动标记失败
      - 排队上限 20，超限拒绝
    """

    MAX_QUEUE_SIZE = 20
    MAX_CONCURRENCY = 3  # 每 Agent 类型

    def __init__(self):
        self._queues: dict[Priority, list[QueueItem]] = {
            p: [] for p in Priority
        }
        self._running: dict[str, list[QueueItem]] = defaultdict(list)
        self._completed: list[QueueItem] = []
        self._lock = threading.Lock()
        self.stats = {
            "enqueued": 0,
            "started": 0,
            "completed": 0,
            "failed": 0,
            "rejected": 0,
            "timeout": 0,
        }

    def enqueue(self, item: QueueItem) -> bool:
        """入队 — 返回 True=成功, False=队列已满被拒绝"""
        with self._lock:
            total = sum(len(q) for q in self._queues.values())
            if total >= self.MAX_QUEUE_SIZE:
                self.stats["rejected"] += 1
                logger.warning(f"队列已满({self.MAX_QUEUE_SIZE}), 拒绝: {item.task_id}")
                return False

            self._queues[item.priority].append(item)
            self.stats["enqueued"] += 1
            logger.debug(f"入队: {item.task_id} (优先级={item.priority.name})")
            return True

    def dequeue(self, agent_type: str | None = None) -> QueueItem | None:
        """出队 — 按优先级取，检查并发上限"""
        with self._lock:
            # 检查并发上限
            if agent_type:
                running_count = len(self._running.get(agent_type, []))
                if running_count >= self.MAX_CONCURRENCY:
                    return None

            for priority in sorted(Priority):
                q = self._queues[priority]
                if not q:
                    continue

                # 优先匹配 agent_type
                if agent_type:
                    for i, item in enumerate(q):
                        if item.agent_type == agent_type:
                            item = q.pop(i)
                            self._running[agent_type].append(item)
                            self.stats["started"] += 1
                            return item

                # 无指定类型或未匹配到，取队首
                if q:
                    item = q.pop(0)
                    at = item.agent_type
                    if at not in self._running:
                        self._running[at] = []
                    self._running[at].append(item)
                    self.stats["started"] += 1
                    return item

            return None

    def complete(self, item: QueueItem, success: bool = True) -> None:
        """标记任务完成"""
        agent_type = item.agent_type
        with self._lock:
            if agent_type in self._running and item in self._running[agent_type]:
                self._running[agent_type].remove(item)
            if success:
                self.stats["completed"] += 1
                self._completed.append(item)
            else:
                self.stats["failed"] += 1

    def timeout_check(self) -> list[QueueItem]:
        """检查并超时所有超时任务"""
        now = time.time()
        timed_out = []

        with self._lock:
            for agent_type, running_list in list(self._running.items()):
                for item in running_list[:]:
                    if now - item.created_at > item.timeout:
                        running_list.remove(item)
                        timed_out.append(item)
                        self.stats["timeout"] += 1
                        self.stats["failed"] += 1

        return timed_out

    def queue_sizes(self) -> dict[str, int]:
        """各优先级队列大小"""
        with self._lock:
            return {p.name: len(q) for p, q in self._queues.items()}

    def running_count(self) -> dict[str, int]:
        """各类型正在运行的任务数"""
        with self._lock:
            return {k: len(v) for k, v in self._running.items()}

    @property
    def total_waiting(self) -> int:
        return sum(len(q) for q in self._queues.values())

    def summary(self) -> str:
        sizes = self.queue_sizes()
        running = self.running_count()
        s = self.stats
        return (
            f"队列: {self.total_waiting}等待 | "
            f"{sum(running.values())}运行 | "
            f"{s['completed']}完成 {s['failed']}失败 {s['timeout']}超时 {s['rejected']}拒绝"
        )


class LoadBalancer:
    """负载均衡器 — 多 Agent 类型的任务分发

    基于 TaskQueue 实现优先级调度和并发控制。
    """

    def __init__(self):
        self.queue = TaskQueue()
        self.agent_weights: dict[str, float] = {
            "cc": 1.0,
            "codex": 1.0,
            "hermes": 1.0,
        }
        self.stats = {
            "dispatched": 0,
            "load_shed": 0,
        }

    def submit(self, item: QueueItem) -> bool:
        """提交一个任务到负载均衡器"""
        return self.queue.enqueue(item)

    def dispatch(self) -> QueueItem | None:
        """派发下一个可执行的任务"""
        # 按权重轮询 agent 类型
        for agent_type in sorted(self.agent_weights.keys()):
            item = self.queue.dequeue(agent_type)
            if item:
                self.stats["dispatched"] += 1
                return item

        # 无指定类型匹配，取任意
        item = self.queue.dequeue()
        if item:
            self.stats["dispatched"] += 1
        return item

    def get_load(self, agent_type: str = "") -> float:
        """获取负载率 (0-1)"""
        running = self.queue.running_count()
        if agent_type:
            count = running.get(agent_type, 0)
            return count / TaskQueue.MAX_CONCURRENCY
        total = sum(running.values())
        max_total = TaskQueue.MAX_CONCURRENCY * len(self.agent_weights)
        return total / max(max_total, 1)

    def can_accept(self) -> bool:
        return self.queue.total_waiting < TaskQueue.MAX_QUEUE_SIZE

    def summary(self) -> str:
        return (
            f"负载均衡: {self.queue.summary()} | "
            f"负载率={self.get_load()*100:.0f}%"
        )


# ═══════════════════════════════════════════════════════════
# 8.11 故障熔断
# ═══════════════════════════════════════════════════════════


class CircuitState:
    CLOSED = "closed"       # 正常
    OPEN = "open"           # 熔断开启（拒绝请求）
    HALF_OPEN = "half_open" # 半开（允许探测）


class CircuitBreaker:
    """故障熔断器 — 连续失败自动暂停，恢复后自动关闭

    策略：
      - 连续 N 次失败 → OPEN（熔断开启）
      - 熔断开启持续 T 秒 → HALF_OPEN（允许一次探测）
      - 探测成功 → CLOSED（恢复正常）
      - 探测失败 → OPEN（重新计时）
    """

    def __init__(self, name: str = "default",
                 failure_threshold: int = 3,
                 recovery_timeout: float = 600.0,  # 10分钟
                 half_open_max_retries: int = 1):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_retries = half_open_max_retries

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.last_success_time = 0.0
        self.half_open_attempts = 0
        self.stats = {"opened": 0, "closed": 0, "rejected": 0}

    def call(self, fn: Callable, *args, **kwargs) -> Any:
        """带熔断保护地调用函数

        Raises:
            CircuitBreakerOpenError: 熔断开启，拒绝调用
        """
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_attempts = 0
                logger.info(f"熔断器 {self.name}: OPEN → HALF_OPEN (超时恢复)")
            else:
                self.stats["rejected"] += 1
                raise CircuitBreakerOpenError(
                    f"熔断器 {self.name} 开启中 ({self.recovery_timeout - (time.time() - self.last_failure_time):.0f}s 后重试)"
                )

        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        self.last_success_time = time.time()
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.stats["closed"] += 1
            logger.info(f"熔断器 {self.name}: HALF_OPEN → CLOSED (探测成功)")
        else:
            self.failure_count = 0

    def _on_failure(self) -> None:
        self.last_failure_time = time.time()
        self.failure_count += 1

        if self.state == CircuitState.HALF_OPEN:
            self.half_open_attempts += 1
            if self.half_open_attempts >= self.half_open_max_retries:
                self.state = CircuitState.OPEN
                self.stats["opened"] += 1
                logger.warning(f"熔断器 {self.name}: HALF_OPEN → OPEN (探测失败)")
            return

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.stats["opened"] += 1
            logger.warning(
                f"熔断器 {self.name}: CLOSED → OPEN "
                f"(连续 {self.failure_count} 次失败)"
            )

    def reset(self) -> None:
        """手动重置熔断器"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.stats["closed"] += 1
        logger.info(f"熔断器 {self.name}: 手动重置 → CLOSED")

    @property
    def is_available(self) -> bool:
        return self.state != CircuitState.OPEN

    def summary(self) -> str:
        remaining = max(0, self.recovery_timeout - (time.time() - self.last_failure_time))
        return (
            f"熔断器 '{self.name}': {self.state} "
            f"(失败 {self.failure_count}/{self.failure_threshold}, "
            f"恢复倒计时 {remaining:.0f}s)"
        )


class CircuitBreakerOpenError(Exception):
    pass


class CircuitBreakerRegistry:
    """熔断器注册表 — 管理多个命名熔断器"""

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name)
        return self._breakers[name]

    def all(self) -> list[CircuitBreaker]:
        return list(self._breakers.values())

    def check_all(self) -> list[str]:
        """检查所有熔断器状态，返回开启中的列表"""
        return [b.name for b in self._breakers.values() if not b.is_available]

    def reset_all(self) -> None:
        for b in self._breakers.values():
            b.reset()


# ═══════════════════════════════════════════════════════════
# 8.8 缓存冷热分层
# ═══════════════════════════════════════════════════════════


@dataclass
class CacheEntry:
    key: str
    value: Any
    tier: str = "hot"        # hot / warm / cold
    hit_count: int = 0
    created_at: float = field(default_factory=time.time)
    ttl: float = 3600.0      # 默认1小时
    last_access: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        return time.time() > self.created_at + self.ttl

    @property
    def age(self) -> float:
        return time.time() - self.created_at


class TieredCache:
    """冷热分层缓存 — LRU 淘汰 + 热区保护

    三层：
      HOT:   高频命中，永不淘汰（最多 20 条）
      WARM:  普通缓存，LRU 淘汰（最多 200 条）
      COLD:  低频数据，到期即删

    策略：
      - 命中次数 >= 3 自动升级为 HOT
      - 连续 30 分钟未命中降级为 WARM
      - HOT 区满 → 最旧移入 WARM
      - WARM 满 → LRU 淘汰
    """

    HOT_MAX = 20
    WARM_MAX = 200
    HOT_PROMOTION_THRESHOLD = 3    # 命中 N 次升热区
    COLD_DOWNGRADE_TIME = 1800.0   # 30 分钟未命中降级

    def __init__(self):
        self._hot: OrderedDict[str, CacheEntry] = OrderedDict()
        self._warm: OrderedDict[str, CacheEntry] = OrderedDict()
        self._cold: dict[str, CacheEntry] = {}
        self.stats = {"hit": 0, "miss": 0, "evict": 0, "promote": 0, "downgrade": 0}

    def get(self, key: str) -> Any | None:
        """读取缓存项 — 命中后自动升级"""
        entry = self._hot.get(key) or self._warm.get(key) or self._cold.get(key)

        if entry is None:
            self.stats["miss"] += 1
            return None

        entry.hit_count += 1
        entry.last_access = time.time()
        self.stats["hit"] += 1

        # 命中次数达标 → 升级热区
        if entry.hit_count >= self.HOT_PROMOTION_THRESHOLD and key not in self._hot:
            self._promote(key, entry)

        return entry.value

    def set(self, key: str, value: Any, ttl: float = 3600.0,
            tier: str = "warm") -> None:
        """写入缓存"""
        entry = CacheEntry(key=key, value=value, tier=tier, ttl=ttl)

        if tier == "hot":
            if len(self._hot) >= self.HOT_MAX:
                _, oldest = self._hot.popitem(last=False)
                self._to_warm(oldest.key, oldest)
                self.stats["evict"] += 1
            self._hot[key] = entry
        elif tier == "cold":
            self._cold[key] = entry
        else:
            self._to_warm(key, entry)

    def _promote(self, key: str, entry: CacheEntry) -> None:
        """升级到热区"""
        self._warm.pop(key, None)
        self._cold.pop(key, None)

        if len(self._hot) >= self.HOT_MAX:
            _, oldest = self._hot.popitem(last=False)
            self._to_warm(oldest.key, oldest)
            self.stats["evict"] += 1

        entry.tier = "hot"
        self._hot[key] = entry
        self._hot.move_to_end(key)
        self.stats["promote"] += 1

    def _to_warm(self, key: str, entry: CacheEntry) -> None:
        """放入温区"""
        entry.tier = "warm"
        if len(self._warm) >= self.WARM_MAX:
            self._warm.popitem(last=False)
            self.stats["evict"] += 1
        self._warm[key] = entry

    def cleanup(self) -> int:
        """清理过期条目 + 降级长期未命中"""
        now = time.time()
        removed = 0

        for store_name, store in [("hot", self._hot), ("warm", self._warm), ("cold", self._cold)]:
            expired = [k for k, v in store.items() if v.is_expired]
            for k in expired:
                del store[k]
                removed += 1

        # 温区长期未命中降级冷区
        for key in list(self._warm.keys()):
            entry = self._warm[key]
            if now - entry.last_access > self.COLD_DOWNGRADE_TIME:
                entry.tier = "cold"
                del self._warm[key]
                self._cold[key] = entry
                self.stats["downgrade"] += 1

        return removed

    def invalidate(self, key: str) -> bool:
        """失效单条缓存"""
        for store in [self._hot, self._warm, self._cold]:
            if key in store:
                del store[key]
                return True
        return False

    def invalidate_prefix(self, prefix: str) -> int:
        """按前缀失效"""
        count = 0
        for store in [self._hot, self._warm, self._cold]:
            keys = [k for k in store if k.startswith(prefix)]
            for k in keys:
                del store[k]
                count += 1
        return count

    def sizes(self) -> dict[str, int]:
        return {"hot": len(self._hot), "warm": len(self._warm), "cold": len(self._cold)}

    def summary(self) -> str:
        s = self.sizes()
        return (
            f"缓存: HOT={s['hot']}/{self.HOT_MAX} "
            f"WARM={s['warm']}/{self.WARM_MAX} "
            f"COLD={s['cold']} "
            f"| 命中={self.stats['hit']} 未命中={self.stats['miss']} "
            f"淘汰={self.stats['evict']}"
        )


# ═══════════════════════════════════════════════════════════
# 8.4 成本预算监控
# ═══════════════════════════════════════════════════════════


@dataclass
class BudgetQuota:
    """预算配额"""
    max_per_task: int = 50000       # 单任务最大 token
    max_per_hour: int = 200000      # 每小时最大 token
    max_per_day: int = 1000000      # 每天最大 token
    warning_threshold: float = 0.6  # 60% 预警
    critical_threshold: float = 0.8 # 80% 限制
    hard_limit: float = 1.0         # 100% 硬上限


class BudgetMonitor:
    """成本预算监控 — 动态预算控制 + 梯度降级

    三阈值：
      < 60%: 全流程
      60-80%: 预警，压缩质检
      80-100%: 限制，跳质检仅验收
      >= 100%: 拒绝新任务
    """

    def __init__(self, quota: BudgetQuota | None = None):
        self.quota = quota or BudgetQuota()
        self._hourly_usage: list[tuple[float, int]] = []
        self._daily_usage: list[tuple[float, int]] = []
        self._task_costs: dict[str, int] = {}
        self.stats = {"tasks": 0, "total_tokens": 0, "rejected": 0, "degraded": 0}

    def record_usage(self, task_id: str, tokens: int) -> None:
        """记录 token 消耗"""
        now = time.time()
        self._hourly_usage.append((now, tokens))
        self._daily_usage.append((now, tokens))
        self._task_costs[task_id] = tokens
        self.stats["tasks"] += 1
        self.stats["total_tokens"] += tokens

    def hourly_usage(self) -> int:
        """过去一小时的 token 消耗"""
        cutoff = time.time() - 3600
        self._hourly_usage = [(t, c) for t, c in self._hourly_usage if t > cutoff]
        return sum(c for _, c in self._hourly_usage)

    def daily_usage(self) -> int:
        """过去一天的 token 消耗"""
        cutoff = time.time() - 86400
        self._daily_usage = [(t, c) for t, c in self._daily_usage if t > cutoff]
        return sum(c for _, c in self._daily_usage)

    def check(self, estimated_cost: int = 0) -> dict:
        """检查当前预算状态

        Returns:
            {level: "normal"/"warning"/"critical"/"hard_limit",
             recommendation: str,
             ratios: {...}}
        """
        hour_ratio = self.hourly_usage() / max(self.quota.max_per_hour, 1)
        day_ratio = self.daily_usage() / max(self.quota.max_per_day, 1)
        task_ratio = estimated_cost / max(self.quota.max_per_task, 1)

        max_ratio = max(hour_ratio, day_ratio, task_ratio)

        if max_ratio >= self.quota.hard_limit:
            return {
                "level": "hard_limit",
                "recommendation": "拒绝新任务",
                "ratios": {"hour": hour_ratio, "day": day_ratio, "task": task_ratio},
            }
        elif max_ratio >= self.quota.critical_threshold:
            self.stats["degraded"] += 1
            return {
                "level": "critical",
                "recommendation": "跳过质检，仅验收",
                "ratios": {"hour": hour_ratio, "day": day_ratio, "task": task_ratio},
            }
        elif max_ratio >= self.quota.warning_threshold:
            return {
                "level": "warning",
                "recommendation": "压缩质检+检索",
                "ratios": {"hour": hour_ratio, "day": day_ratio, "task": task_ratio},
            }
        else:
            return {
                "level": "normal",
                "recommendation": "维持全流程",
                "ratios": {"hour": hour_ratio, "day": day_ratio, "task": task_ratio},
            }

    def can_accept(self, estimated_cost: int = 0) -> tuple[bool, str]:
        """是否可以接受新任务"""
        status = self.check(estimated_cost)
        if status["level"] == "hard_limit":
            self.stats["rejected"] += 1
            return (False, f"预算超限: {status['recommendation']}")
        return (True, status["recommendation"])

    def degradation_level(self) -> str:
        """当前降级等级"""
        return self.check()["level"]

    def summary(self) -> str:
        status = self.check()
        h = self.hourly_usage()
        d = self.daily_usage()
        return (
            f"预算: {h/1000:.0f}K/h({self.quota.max_per_hour/1000:.0f}K) "
            f"{d/1000:.0f}K/d({self.quota.max_per_day/1000:.0f}K) "
            f"| 等级={status['level']} "
            f"| {self.stats['tasks']}任务 {self.stats['degraded']}降级 {self.stats['rejected']}拒绝"
        )


# ═══════════════════════════════════════════════════════════
# 8.6 反馈过滤 + KPI 指标
# ═══════════════════════════════════════════════════════════


class FeedbackFilter:
    """负反馈抑制 — 临时/矛盾/一次性反馈降权

    策略：
      - 临时性: 与 L1 冲突的标记临时，权重 50%
      - 矛盾性: 与上一条反馈冲突的，重置计数器
      - 一次性: 同 topic 只出现 1 次的不固化
      - 连续 3 次确认的稳定反馈 → 标记"已确认"
    """

    def __init__(self):
        self._topic_counters: dict[str, int] = {}
        self._topic_values: dict[str, list] = defaultdict(list)
        self._confirmed: set[str] = set()

    def add(self, topic: str, value: str, source: str = "user") -> dict:
        """添加一条反馈

        Returns:
            {status: "accepted"/"rejected"/"pending",
             confidence: float,
             reason: str}
        """
        self._topic_counters[topic] = self._topic_counters.get(topic, 0) + 1
        self._topic_values[topic].append({"value": value, "source": source, "time": time.time()})

        counter = self._topic_counters[topic]
        history = self._topic_values[topic]

        # 矛盾检测
        if len(history) >= 2 and source != "user":
            last = history[-2]["value"]
            if last != value and last != "":
                return {
                    "status": "rejected",
                    "confidence": 0.0,
                    "reason": f"矛盾反馈: 上次='{last[:30]}', 本次='{value[:30]}'",
                }

        # 一次性（仅出现 1 次）→ 低置信度
        if counter == 1:
            return {
                "status": "pending",
                "confidence": 0.3,
                "reason": "首次反馈，待确认",
            }

        # 连续 3 次 → 确认
        if counter >= 3:
            self._confirmed.add(topic)
            return {
                "status": "accepted",
                "confidence": 0.95,
                "reason": f"连续 {counter} 次确认",
            }

        # 2 次 → 中等置信度
        return {
            "status": "accepted",
            "confidence": 0.6,
            "reason": f"第 {counter} 次反馈，基本可信",
        }

    def is_confirmed(self, topic: str) -> bool:
        return topic in self._confirmed

    def confidence(self, topic: str) -> float:
        if topic in self._confirmed:
            return 0.95
        c = self._topic_counters.get(topic, 0)
        return min(0.3 + c * 0.2, 0.9)

    def summary(self) -> str:
        return (
            f"反馈过滤: {len(self._topic_counters)} topic, "
            f"{len(self._confirmed)} 已确认"
        )


class KPITracker:
    """12 项 KPI 指标追踪

    跟踪核心指标，用于闭环调控。
    """

    def __init__(self):
        self._metrics: dict[str, list[float]] = defaultdict(list)
        self._timestamps: dict[str, list[float]] = defaultdict(list)

    KPI_NAMES = [
        "throughput",          # 任务吞吐量（任务/小时）
        "success_rate",        # 成功率
        "avg_execution_time",  # 平均执行时间
        "avg_cost",            # 平均 token 消耗
        "qa_pass_rate",        # 质检通过率
        "eval_score",          # 评估平均分
        "cache_hit_rate",      # 缓存命中率
        "queue_wait_time",     # 队列等待时间
        "circuit_breaker_rate",# 熔断率
        "budget_usage",        # 预算使用率
        "user_satisfaction",   # 用户满意度（0-1）
        "memory_hit_rate",     # 记忆命中率
    ]

    def record(self, name: str, value: float) -> None:
        if name in self.KPI_NAMES:
            self._metrics[name].append(value)
            self._timestamps[name].append(time.time())

    def average(self, name: str, window: int = 20) -> float:
        values = self._metrics.get(name, [])[-window:]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def latest(self, name: str) -> float:
        values = self._metrics.get(name, [])
        return values[-1] if values else 0.0

    def trend(self, name: str, window: int = 10) -> str:
        """趋势判断：上升/下降/稳定"""
        values = self._metrics.get(name, [])[-window:]
        if len(values) < 3:
            return "stable"
        first_half = sum(values[:len(values)//2]) / max(len(values)//2, 1)
        second_half = sum(values[len(values)//2:]) / max(len(values) - len(values)//2, 1)
        diff = second_half - first_half
        threshold = first_half * 0.05  # 5% 变化
        if diff > threshold:
            return "up"
        elif diff < -threshold:
            return "down"
        return "stable"

    def health_score(self) -> float:
        """综合健康评分 0-1"""
        scores = []
        metrics = {
            "success_rate": 1.0,
            "qa_pass_rate": 0.7,
            "eval_score": 0.7,
            "cache_hit_rate": 0.5,
        }
        for name, target in metrics.items():
            val = self.average(name)
            if val > 0:
                scores.append(min(1.0, val / target))
        return sum(scores) / max(len(scores), 1) if scores else 0.5

    def summary(self) -> str:
        active = [n for n in self.KPI_NAMES if self._metrics.get(n)]
        return (
            f"KPI: {len(active)}/{len(self.KPI_NAMES)} 活跃 "
            f"| 健康={self.health_score():.2f}"
        )


# ═══════════════════════════════════════════════════════════
# 8.3 Recovery Agent
# ═══════════════════════════════════════════════════════════


class RecoveryAgent:
    """故障恢复 Agent — 三级回滚 + 重试策略

    Level 1: 回重执行（重新跑失败的节点，换 Agent 或不换）
    Level 2: 回重规划（重新拆解任务）
    Level 3: 熔断转人工（也支持自动恢复路径）
    """

    MAX_RETRIES_PER_TASK = 3  # 同任务最多恢复 3 次

    def __init__(self):
        self._attempts: dict[str, int] = {}
        self._recovery_log: list[dict] = []
        self.stats = {"recovered": 0, "escalated": 0, "failed": 0}

    def recover(self, task_id: str, error: str,
                rollback_level: int = 1,
                context: dict | None = None) -> dict:
        """执行恢复

        Args:
            task_id: 失败任务 ID
            error: 错误描述
            rollback_level: 1=重执行 / 2=重规划 / 3=熔断
            context: 附加上下文

        Returns:
            {action: str, target_state: str, reason: str, switch_agent: bool}
        """
        self._attempts[task_id] = self._attempts.get(task_id, 0) + 1
        attempt = self._attempts[task_id]

        if attempt >= self.MAX_RETRIES_PER_TASK:
            self.stats["escalated"] += 1
            self._log(task_id, error, 3, "超过最大重试次数，强制熔断")
            return {
                "action": "escalate",
                "target_state": "failed",
                "reason": f"重试 {attempt} 次后仍失败，转人工",
                "switch_agent": False,
            }

        if rollback_level == 1:
            self.stats["recovered"] += 1
            switch = attempt > 1  # 第 2 次重试换 Agent
            self._log(task_id, error, 1, f"Level 1 恢复 (尝试 {attempt})")
            return {
                "action": "retry_execute",
                "target_state": "executing",
                "reason": f"Level 1 回滚执行 ({error[:50]})",
                "switch_agent": switch,
            }

        elif rollback_level == 2:
            self.stats["recovered"] += 1
            self._log(task_id, error, 2, f"Level 2 恢复 (尝试 {attempt})")
            return {
                "action": "replan",
                "target_state": "planning",
                "reason": f"Level 2 回滚规划 ({error[:50]})",
                "switch_agent": True,
            }

        else:  # Level 3
            self.stats["escalated"] += 1
            self._log(task_id, error, 3, "熔断转人工")
            return {
                "action": "escalate",
                "target_state": "failed",
                "reason": f"Level 3 熔断: {error[:100]}",
                "switch_agent": False,
            }

    def _log(self, task_id: str, error: str, level: int, action: str) -> None:
        entry = {
            "task_id": task_id,
            "error": error[:100],
            "level": level,
            "action": action,
            "attempt": self._attempts.get(task_id, 0),
            "timestamp": time.time(),
        }
        self._recovery_log.append(entry)
        logger.info(f"[Recovery] {task_id}: L{level} — {action}")

    def get_history(self, task_id: str = "") -> list[dict]:
        if task_id:
            return [e for e in self._recovery_log if e["task_id"] == task_id]
        return self._recovery_log

    def reset_attempts(self, task_id: str) -> None:
        self._attempts.pop(task_id, None)

    def summary(self) -> str:
        recent = self._recovery_log[-5:] if self._recovery_log else []
        return (
            f"Recovery: {self.stats['recovered']}恢复 "
            f"{self.stats['escalated']}熔断 {self.stats['failed']}失败"
        )


# ═══════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════

def test_task_queue():
    q = TaskQueue()
    q.enqueue(QueueItem("t1", "普通任务"))
    q.enqueue(QueueItem("t2", "紧急任务", priority=Priority.CRITICAL))
    q.enqueue(QueueItem("t3", "低优先级", priority=Priority.LOW))

    i1 = q.dequeue()
    assert i1.task_id == "t2", f"应出紧急任务, 实际={i1.task_id}"
    i2 = q.dequeue()
    assert i2.task_id == "t1"
    i3 = q.dequeue()
    assert i3.task_id == "t3"
    print(f"TaskQueue 优先级排序: ✓")


def test_queue_concurrency():
    q = TaskQueue()
    for i in range(5):
        q.enqueue(QueueItem(f"t{i}", f"task {i}", agent_type="cc"))

    dequed = []
    for _ in range(5):
        item = q.dequeue("cc")
        if item:
            dequed.append(item.task_id)

    assert len(dequed) == TaskQueue.MAX_CONCURRENCY, f"并发应=3, 实际={len(dequed)}"
    print(f"TaskQueue 并发上限 {TaskQueue.MAX_CONCURRENCY}: ✓")


def test_queue_full():
    q = TaskQueue()
    q.MAX_QUEUE_SIZE = 3
    for i in range(4):
        q.enqueue(QueueItem(f"t{i}", f"task {i}"))
    assert q.stats["rejected"] == 1
    print(f"TaskQueue 队列满拒绝: ✓")


def test_circuit_breaker():
    cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)

    def fail():
        raise ValueError("模拟失败")

    def succeed():
        return "ok"

    # 失败 2 次 → 熔断
    for _ in range(2):
        try: cb.call(fail)
        except: pass
    assert cb.state == CircuitState.OPEN
    assert cb.stats["opened"] == 1

    # 熔断中拒绝
    try:
        cb.call(succeed)
        assert False, "应拒绝"
    except CircuitBreakerOpenError:
        assert cb.stats["rejected"] == 1

    # 等恢复
    import time
    time.sleep(0.12)
    result = cb.call(succeed)
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED
    print(f"CircuitBreaker 熔断/恢复: ✓")


def test_tiered_cache():
    tc = TieredCache()
    tc.set("k1", "v1", tier="warm")
    tc.set("k2", "v2", tier="warm")
    tc.set("k3", "v3", tier="hot")

    assert tc.get("k1") == "v1"
    assert tc.get("k2") == "v2"
    assert tc.get("k3") == "v3"
    assert tc.get("nonexist") is None

    # 多次命中升热区
    tc.get("k1"); tc.get("k1"); tc.get("k1")
    assert "k1" in tc._hot

    print(f"TieredCache 冷热分层: ✓ (sizes={tc.sizes()})")


def test_budget_monitor():
    bm = BudgetMonitor()
    assert bm.can_accept()[0] == True

    # 模拟消耗
    bm.record_usage("t1", 100)
    bm.record_usage("t2", 200)

    status = bm.check()
    assert status["level"] == "normal"

    print(f"BudgetMonitor 预算监控: ✓ ({bm.summary()})")


def test_feedback_filter():
    ff = FeedbackFilter()

    r1 = ff.add("color", "蓝色")
    assert r1["status"] == "pending"
    assert r1["confidence"] == 0.3

    r2 = ff.add("color", "蓝色")
    assert r2["status"] == "accepted"
    assert r2["confidence"] == 0.6

    r3 = ff.add("color", "蓝色")
    assert r3["status"] == "accepted"
    assert r3["confidence"] == 0.95
    assert ff.is_confirmed("color")

    # 矛盾检测
    r4 = ff.add("theme", "dark", source="critic")
    r5 = ff.add("theme", "light", source="critic")
    assert r5["status"] == "rejected"

    print(f"FeedbackFilter 负反馈抑制: ✓ ({ff.summary()})")


def test_kpi():
    kpi = KPITracker()
    for i in range(5):
        kpi.record("success_rate", 0.8 + i * 0.03)
        kpi.record("qa_pass_rate", 0.75)
        kpi.record("eval_score", 0.7)
        kpi.record("cache_hit_rate", 0.6)

    assert kpi.average("success_rate") > 0
    assert kpi.health_score() > 0
    print(f"KPI 指标追踪: ✓ ({kpi.summary()})")


def test_recovery():
    ra = RecoveryAgent()

    # Level 1
    r1 = ra.recover("t1", "连接超时", 1)
    assert r1["action"] == "retry_execute"
    assert r1["switch_agent"] == False  # 首次不换 Agent

    # 再试一次 → 换 Agent
    r2 = ra.recover("t1", "连接超时", 1)
    assert r2["switch_agent"] == True

    # Level 2
    r3 = ra.recover("t2", "规划错误", 2)
    assert r3["action"] == "replan"

    # Level 3
    r4 = ra.recover("t3", "安全违规", 3)
    assert r4["action"] == "escalate"

    # 超限熔断
    ra.recover("t4", "bug", 1)
    ra.recover("t4", "bug", 1)
    ra.recover("t4", "bug", 1)
    r5 = ra.recover("t4", "bug", 1)
    assert r5["action"] == "escalate"

    print(f"RecoveryAgent 故障恢复: ✓ ({ra.summary()})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print("═══ 外围支撑系统测试 ═══\n")

    print("─ 8.10 负载均衡+优先级队列 ─")
    test_task_queue()
    test_queue_concurrency()
    test_queue_full()
    print()

    print("─ 8.11 故障熔断 ─")
    test_circuit_breaker()
    print()

    print("─ 8.8 缓存冷热分层 ─")
    test_tiered_cache()
    print()

    print("─ 8.4 成本预算监控 ─")
    test_budget_monitor()
    print()

    print("─ 8.6 反馈过滤+KPI ─")
    test_feedback_filter()
    test_kpi()
    print()

    print("─ 8.3 Recovery Agent ─")
    test_recovery()
    print()

    print("\n═══ 全部测试通过 ═══")
