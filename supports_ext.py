"""
外围支撑系统扩展 — 段8剩余机制

包含：
  8.1  通信协议Schema          — MessageSchema
  8.2  记忆访问控制             — MemoryAccessControl
  8.5  人工干预点               — InterventionManager
  8.7  记忆过期清理             — MemoryCleanupManager
  8.9  特殊任务适配             — SpecialTaskAdapter
  8.12 手动纪律告警             — DisciplineWatcher
"""

from __future__ import annotations

import time
import json
import re
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger("agent-framework.supports-ext")


# ═══════════════════════════════════════════════════════════
# 8.1 通信协议 Schema
# ═══════════════════════════════════════════════════════════


class MessageType(str, Enum):
    """Agent间消息类型"""
    TASK = "task"                     # 任务
    RESULT = "result"                 # 执行结果
    QUERY = "query"                   # 查询
    RESPONSE = "response"             # 响应
    ERROR = "error"                   # 错误
    STATUS = "status"                 # 状态更新
    ARBITRATION = "arbitration"       # 仲裁请求
    ARBITRATION_RESULT = "arb_result" # 仲裁结果
    HEARTBEAT = "heartbeat"           # 心跳
    LOG = "log"                       # 日志


@dataclass
class Message:
    """标准化Agent间消息"""
    msg_id: str
    msg_type: MessageType | str
    source: str                       # 发送方 Agent 名称
    target: str                       # 接收方 Agent 名称
    payload: Any                      # 消息体
    trace_id: str = ""                # 追踪链 ID
    correlation_id: str = ""          # 关联消息 ID（回复时用）
    priority: int = 2                 # 0=critical 1=high 2=normal 3=low
    timestamp: float = field(default_factory=time.time)
    ttl: float = 300.0                # 消息过期时间（秒）
    version: str = "1.0"             # Schema版本

    @property
    def is_expired(self) -> bool:
        return time.time() > self.timestamp + self.ttl

    def to_dict(self) -> dict:
        return {
            "msg_id": self.msg_id,
            "type": self.msg_type.value if isinstance(self.msg_type, MessageType) else self.msg_type,
            "source": self.source,
            "target": self.target,
            "payload": self.payload,
            "trace_id": self.trace_id,
            "correlation_id": self.correlation_id,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "version": self.version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @staticmethod
    def from_dict(data: dict) -> Message:
        return Message(
            msg_id=data.get("msg_id", ""),
            msg_type=data.get("type", ""),
            source=data.get("source", ""),
            target=data.get("target", ""),
            payload=data.get("payload", ""),
            trace_id=data.get("trace_id", ""),
            correlation_id=data.get("correlation_id", ""),
            priority=data.get("priority", 2),
            timestamp=data.get("timestamp", time.time()),
        )

    @staticmethod
    def reply_to(original: Message, payload: Any,
                 msg_type: MessageType = MessageType.RESPONSE) -> Message:
        """创建回复消息"""
        return Message(
            msg_id=f"rsp-{original.msg_id}",
            msg_type=msg_type,
            source=original.target,
            target=original.source,
            payload=payload,
            trace_id=original.trace_id,
            correlation_id=original.msg_id,
            priority=original.priority,
        )


class MessageValidator:
    """消息格式验证器"""

    REQUIRED_FIELDS = ["msg_id", "type", "source", "target", "payload"]

    @staticmethod
    def validate(msg: Message) -> list[str]:
        errors = []
        d = msg.to_dict() if not isinstance(msg, dict) else msg
        for field in MessageValidator.REQUIRED_FIELDS:
            if not d.get(field):
                errors.append(f"缺少必需字段: {field}")
        if d.get("type") and d["type"] not in [t.value for t in MessageType]:
            errors.append(f"未知消息类型: {d['type']}")
        return errors


class MessageBus:
    """简易消息总线 — 记录和转发 Agent 间消息"""

    def __init__(self):
        self._messages: list[Message] = []
        self._by_trace: dict[str, list[Message]] = defaultdict(list)
        self.stats = {"sent": 0, "expired": 0, "invalid": 0}

    def send(self, msg: Message) -> bool:
        errors = MessageValidator.validate(msg)
        if errors:
            self.stats["invalid"] += 1
            logger.warning(f"消息验证失败: {errors}")
            return False

        self._messages.append(msg)
        if msg.trace_id:
            self._by_trace[msg.trace_id].append(msg)
        self.stats["sent"] += 1
        return True

    def get_by_trace(self, trace_id: str) -> list[Message]:
        return self._by_trace.get(trace_id, [])

    def get_by_source(self, source: str) -> list[Message]:
        return [m for m in self._messages if m.source == source]

    def get_by_target(self, target: str) -> list[Message]:
        return [m for m in self._messages if m.target == target]

    def cleanup_expired(self) -> int:
        before = len(self._messages)
        self._messages = [m for m in self._messages if not m.is_expired]
        # 清理 trace 索引
        for trace_id in list(self._by_trace.keys()):
            self._by_trace[trace_id] = [
                m for m in self._by_trace[trace_id] if not m.is_expired
            ]
        removed = before - len(self._messages)
        self.stats["expired"] += removed
        return removed

    def summary(self) -> str:
        return (
            f"MessageBus: {len(self._messages)} 活跃, "
            f"总发送={self.stats['sent']}, "
            f"过期清理={self.stats['expired']}"
        )


# ═══════════════════════════════════════════════════════════
# 8.2 记忆访问控制
# ═══════════════════════════════════════════════════════════


class MemoryAccessControl:
    """记忆访问控制 — 脱敏白名单 + 权限过滤

    策略：
      - 白名单模式：仅允许写入匹配白名单前缀的 key
      - 脱敏规则：敏感字段自动替换
      - 访问等级：agent / system / human 三级
    """

    SENSITIVE_PATTERNS = [
        (r"api[_-]?key[=:]\s*\S+", "api_key: ***"),
        (r"token[=:]\s*\S{8,}", "token: ***"),
        (r"password[=:]\s*\S+", "password: ***"),
        (r"secret[=:]\s*\S+", "secret: ***"),
        (r"sk-[a-zA-Z0-9]{20,}", "sk-***"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "***@***"),
    ]

    def __init__(self):
        # 默认白名单前缀
        self._whitelist_prefixes: set[str] = {
            "pref:", "task:", "skill:", "output:", "log:", "qa:",
            "l4:", "l4v:", "l2:", "l1:",
        }
        self._agent_whitelist: set[str] = {
            "pref:", "task:", "l2:", "l4:",
        }
        self._human_only_prefixes: set[str] = {
            "secret:", "credential:",
        }
        self._denied_agents: set[str] = set()
        self.stats = {"allowed": 0, "denied": 0, "sanitized": 0}

    def is_allowed(self, key: str, agent: str = "hermes",
                   level: str = "agent") -> bool:
        """检查写入是否允许

        Args:
            key: 记忆键名
            agent: 请求的 Agent 名称
            level: 访问等级 (agent/system/human)
        """
        # 敏感区域仅 human 可写
        for prefix in self._human_only_prefixes:
            if key.startswith(prefix):
                if level != "human":
                    self.stats["denied"] += 1
                    return False
                break

        # 黑名单 Agent
        if agent in self._denied_agents:
            self.stats["denied"] += 1
            return False

        # human 级 — 可写任何白名单+human_only前缀
        if level == "human":
            for prefix in self._whitelist_prefixes.union(self._human_only_prefixes):
                if key.startswith(prefix):
                    self.stats["allowed"] += 1
                    return True
            self.stats["denied"] += 1
            return False
        # agent 级 — 仅 agent_whitelist
        for prefix in self._agent_whitelist:
            if key.startswith(prefix):
                self.stats["allowed"] += 1
                return True

        self.stats["denied"] += 1
        return False

    def sanitize(self, content: str) -> str:
        """脱敏处理"""
        result = content
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            before = result
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            if result != before:
                self.stats["sanitized"] += 1
        return result

    def add_whitelist(self, prefix: str) -> None:
        self._whitelist_prefixes.add(prefix)
        self._agent_whitelist.add(prefix)

    def add_denied_agent(self, agent: str) -> None:
        self._denied_agents.add(agent)

    def summary(self) -> str:
        return (
            f"访问控制: 允许={self.stats['allowed']} "
            f"拒绝={self.stats['denied']} "
            f"脱敏={self.stats['sanitized']}"
        )


# ═══════════════════════════════════════════════════════════
# 8.5 人工干预点
# ═══════════════════════════════════════════════════════════


class InterventionPoint(str, Enum):
    """三处人工干预点"""
    AFTER_ROUTING = "after_routing"           # 段3：可否决路由，不能升复杂度、不能改分析
    AFTER_QA = "after_qa"                     # 段5：可修改质检结果，重新触发质检
    BEFORE_CLOSURE = "before_closure"         # 段7：可修反馈，不可改执行结果


class InterventionRecord:
    """人工干预记录"""
    def __init__(self):
        self._records: list[dict] = []

    def add(self, point: InterventionPoint, task_id: str,
            action: str, reason: str, handler: str = "human") -> None:
        self._records.append({
            "point": point.value,
            "task_id": task_id,
            "action": action,
            "reason": reason,
            "handler": handler,
            "timestamp": time.time(),
        })

    def get_by_task(self, task_id: str) -> list[dict]:
        return [r for r in self._records if r["task_id"] == task_id]

    def count(self, point: InterventionPoint | None = None) -> int:
        if point:
            return len([r for r in self._records if r["point"] == point.value])
        return len(self._records)


class InterventionManager:
    """人工干预管理器 — 三处干预点 + 约束规则

    干预点：
      - 段3后：否决路由（只能升复杂度不能降，不能改分析和预算）
      - 段5后：修改质检结果，重新触发质检
      - 段7前：修反馈，改记忆内容

    Args:
        node: DAG 节点上下文
    """

    POINT_CONSTRAINTS = {
        InterventionPoint.AFTER_ROUTING: {
            "allowed_actions": ["override_route", "upgrade_complexity"],
            "forbidden_actions": ["downgrade_complexity", "change_analysis", "change_budget"],
            "description": "路由后：只能升复杂度，不能改分析和预算",
        },
        InterventionPoint.AFTER_QA: {
            "allowed_actions": ["modify_qa_result", "retrigger_qa"],
            "forbidden_actions": ["bypass_qa", "modify_execution"],
            "description": "质检后：可修改质检结果、重新质检，不能改执行",
        },
        InterventionPoint.BEFORE_CLOSURE: {
            "allowed_actions": ["modify_feedback", "override_memory"],
            "forbidden_actions": ["modify_execution_result"],
            "description": "闭环前：可修反馈和记忆，不能改执行结果",
        },
    }

    def __init__(self):
        self.records = InterventionRecord()
        self.stats = {"interventions": 0, "overrides": 0}

    def get_constraints(self, point: InterventionPoint) -> dict:
        return self.POINT_CONSTRAINTS.get(point, {})

    def intervene(self, point: InterventionPoint, task_id: str,
                  action: str, reason: str,
                  handler: str = "human") -> tuple[bool, str]:
        """执行干预

        Returns:
            (allowed: bool, message: str)
        """
        constraints = self.get_constraints(point)
        if not constraints:
            return (False, f"未知干预点: {point}")

        # 检查操作是否被允许
        allowed = constraints.get("allowed_actions", [])
        forbidden = constraints.get("forbidden_actions", [])

        if action in forbidden:
            return (False, f"禁止操作 '{action}' — {constraints.get('description', '')}")

        if action not in allowed:
            return (False, f"不支持的操作 '{action}'，允许: {allowed}")

        # 记录干预
        self.records.add(point, task_id, action, reason, handler)
        self.stats["interventions"] += 1
        if action.startswith("override"):
            self.stats["overrides"] += 1
        return (True, f"干预已记录: {point.value} → {action}")

    def get_history(self, task_id: str = "") -> list[dict]:
        return self.records.get_by_task(task_id) if task_id else self.records._records

    def summary(self) -> str:
        return (
            f"干预: {self.stats['interventions']} 次干预 "
            f"({self.stats['overrides']} 次覆盖) "
            f"段3={self.records.count(InterventionPoint.AFTER_ROUTING)} "
            f"段5={self.records.count(InterventionPoint.AFTER_QA)} "
            f"段7={self.records.count(InterventionPoint.BEFORE_CLOSURE)}"
        )


# ═══════════════════════════════════════════════════════════
# 8.7 记忆过期清理
# ═══════════════════════════════════════════════════════════


@dataclass
class MemorySnapshot:
    """记忆快照"""
    id: str
    data: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    label: str = ""


class MemoryCleanupManager:
    """记忆过期清理 — TTL + 快照 + 检索权重

    策略：
      - L3 会话: 90天归档, 180天删除
      - L5 反馈: 60天过期, 180天清空
      - 快照: 保留最新 3 份
      - 检索权重：热区1.0 / 普通0.7 / 过期0.1
    """

    DEFAULT_TTL = {
        "l1": 3600,           # L1 短期上下文: 1小时
        "l2": 86400 * 7,      # L2 任务日志: 7天
        "l3_conversation": 86400 * 90,    # L3 会话归档: 90天
        "l3_delete": 86400 * 180,         # L3 删除: 180天
        "l4_skill": 86400 * 365 * 2,      # L4 技能: 2年
        "l5_feedback": 86400 * 60,        # L5 反馈: 60天
        "l5_delete": 86400 * 180,         # L5 删除: 180天
    }

    RETRIEVAL_WEIGHTS = {
        "hot": 1.0,
        "normal": 0.7,
        "expired": 0.1,
    }

    MAX_SNAPSHOTS = 3

    def __init__(self):
        self._snapshots: list[MemorySnapshot] = []
        self._cleanup_log: list[dict] = []
        self.stats = {"archived": 0, "deleted": 0, "snapshots": 0}

    def get_ttl(self, layer: str) -> int:
        """获取指定层的 TTL（秒）"""
        return self.DEFAULT_TTL.get(layer, 86400)

    def check_expired(self, key: str, layer: str,
                      created_at: float) -> tuple[bool, str]:
        """检查是否过期

        Returns:
            (expired: bool, action: "keep"/"archive"/"delete")
        """
        age = time.time() - created_at

        if layer == "l3":
            if age > self.DEFAULT_TTL["l3_delete"]:
                return (True, "delete")
            if age > self.DEFAULT_TTL["l3_conversation"]:
                return (True, "archive")
            return (False, "keep")

        if layer == "l5":
            if age > self.DEFAULT_TTL["l5_delete"]:
                return (True, "delete")
            if age > self.DEFAULT_TTL["l5_feedback"]:
                return (True, "archive")
            return (False, "keep")

        ttl = self.DEFAULT_TTL.get(layer, 86400)
        if age > ttl:
            return (True, "delete")
        return (False, "keep")

    def take_snapshot(self, data: dict, label: str = "") -> MemorySnapshot:
        """创建快照"""
        snap = MemorySnapshot(
            id=f"snap-{int(time.time())}",
            data=data,
            label=label,
        )
        self._snapshots.append(snap)
        # 保留最新 MAX_SNAPSHOTS 份
        if len(self._snapshots) > self.MAX_SNAPSHOTS:
            removed = self._snapshots.pop(0)
            self.stats["snapshots"] += 1
        return snap

    def get_snapshots(self) -> list[MemorySnapshot]:
        return self._snapshots

    def get_retrieval_weight(self, age: float, layer: str) -> float:
        """根据过期状态返回检索权重"""
        expired, action = self.check_expired("dummy", layer, time.time() - age)
        if action == "delete":
            return self.RETRIEVAL_WEIGHTS["expired"]
        elif action == "archive" or expired:
            return 0.5
        return self.RETRIEVAL_WEIGHTS["normal"]

    def log_cleanup(self, action: str, key: str, layer: str) -> None:
        self._cleanup_log.append({
            "action": action,
            "key": key,
            "layer": layer,
            "timestamp": time.time(),
        })
        if action == "delete":
            self.stats["deleted"] += 1
        elif action == "archive":
            self.stats["archived"] += 1

    def summary(self) -> str:
        return (
            f"清理: {self.stats['archived']}归档 "
            f"{self.stats['deleted']}删除 "
            f"{len(self._snapshots)}快照"
        )


# ═══════════════════════════════════════════════════════════
# 8.9 特殊任务适配
# ═══════════════════════════════════════════════════════════


class SpecialTaskAdapter:
    """特殊任务适配 — 超长/流式/API限流

    处理：
      1. 超长上下文（>50K tokens）
      2. 流式任务（不缓存、不排队）
      3. API 限流 + 熔断重试
    """

    LONG_CONTEXT_THRESHOLD = 50000

    def __init__(self):
        from supports import CircuitBreaker, LoadBalancer
        self.circuit_breakers: dict[str, Any] = {}
        self.stats = {
            "long_context": 0,
            "streaming": 0,
            "api_limited": 0,
            "circuit_triggered": 0,
        }

    def detect_task_type(self, estimated_tokens: int,
                         is_streaming: bool = False) -> list[str]:
        """检测任务类型，返回适配标识列表"""
        flags = []

        if estimated_tokens > self.LONG_CONTEXT_THRESHOLD:
            flags.append("long_context")
            self.stats["long_context"] += 1

        if is_streaming:
            flags.append("streaming")
            self.stats["streaming"] += 1

        return flags

    def adapt_long_context(self, graph) -> dict:
        """超长上下文适配策略：
        - 分块执行
        - 每块独立检索记忆
        - 结果聚合后汇总
        """
        from dag import TaskNode
        # 标记所有节点为长上下文模式
        for nid, node in graph.nodes.items():
            if not node.context:
                node.context = {}
            node.context["long_context_mode"] = True
            node.context["chunk_size"] = 40000
        return {"adaptation": "chunked", "chunk_size": 40000}

    def adapt_streaming(self, task_id: str) -> dict:
        """流式任务适配：
        - 不入缓存
        - 不排队等待
        - 结果直接推送
        """
        return {
            "adaptation": "streaming",
            "no_cache": True,
            "no_queue": True,
            "direct_delivery": True,
        }

    def get_api_circuit_breaker(self, api_name: str):
        """获取 API 专用的熔断器"""
        if api_name not in self.circuit_breakers:
            from supports import CircuitBreaker
            self.circuit_breakers[api_name] = CircuitBreaker(
                name=f"api:{api_name}",
                failure_threshold=3,
                recovery_timeout=60,
            )
        return self.circuit_breakers[api_name]

    def call_with_retry(self, api_name: str, fn: Callable,
                        *args, **kwargs) -> Any:
        """带 API 限流和熔断的调用"""
        cb = self.get_api_circuit_breaker(api_name)
        try:
            return cb.call(fn, *args, **kwargs)
        except Exception as e:
            self.stats["circuit_triggered"] += 1
            raise

    def summary(self) -> str:
        s = self.stats
        return (
            f"特殊任务: 超长={s['long_context']} "
            f"流式={s['streaming']} API熔断={s['circuit_triggered']}"
        )


# ═══════════════════════════════════════════════════════════
# 8.12 手动纪律告警
# ═══════════════════════════════════════════════════════════


class DisciplineRule(str, Enum):
    """纪律规则类型"""
    BYPASS_QA = "bypass_qa"                       # 跳过质检
    OVERRIDE_ARBITER = "override_arbiter"         # 绕过仲裁器
    FORCE_EXECUTION = "force_execution"           # 强制执行被拒任务
    DIRECT_MEMORY_WRITE = "direct_memory_write"   # 直接写记忆绕过访问控制
    MANUAL_ROUTE_CHANGE = "manual_route_change"   # 手动改路由分配
    EXECUTE_INTERVENTION = "execute_intervention" # 在执行阶段干预


@dataclass
class DisciplineAlert:
    """纪律告警"""
    rule: DisciplineRule
    agent: str
    task_id: str
    description: str
    severity: int = 1  # 1=警告 2=违规 3=严重违规
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "rule": self.rule.value,
            "agent": self.agent,
            "task_id": self.task_id,
            "description": self.description,
            "severity": self.severity,
            "timestamp": self.timestamp,
        }


class DisciplineWatcher:
    """手动纪律告警 — 检测和记录违反纪律的操作

    检测场景：
      - 跳过质检直接验收
      - 绕过仲裁器强制转换
      - 强制执行 Utility 判定不值得的任务
      - 直接写记忆绕过访问控制
    """

    def __init__(self):
        self._alerts: list[DisciplineAlert] = []
        self._violation_counts: dict[str, int] = defaultdict(int)
        self.stats = {"alerts": 0, "violations": 0}

    def watch(self, action: str, agent: str, task_id: str,
              context: dict | None = None) -> DisciplineAlert | None:
        """监控操作，返回告警（如有）"""
        ctx = context or {}

        # 跳过质检
        if action == "bypass_qa":
            return self._alert(DisciplineRule.BYPASS_QA, agent, task_id,
                               f"Agent {agent} 在任务 {task_id} 中跳过质检",
                               severity=2)

        # 绕过仲裁器
        if action == "override_arbiter":
            return self._alert(DisciplineRule.OVERRIDE_ARBITER, agent, task_id,
                               f"Agent {agent} 绕过仲裁器强制转换",
                               severity=3)

        # 强制执行被拒任务
        if action == "force_execution" and ctx.get("utility", 0) < 0:
            return self._alert(DisciplineRule.FORCE_EXECUTION, agent, task_id,
                               f"强制执行 Utility={ctx.get('utility'):.2f} 的被拒任务",
                               severity=2)

        # 直接写记忆
        if action == "direct_memory_write" and not ctx.get("access_allowed"):
            return self._alert(DisciplineRule.DIRECT_MEMORY_WRITE, agent, task_id,
                               f"Agent {agent} 直接写入记忆绕过访问控制",
                               severity=2)

        return None

    def _alert(self, rule: DisciplineRule, agent: str, task_id: str,
               description: str, severity: int = 1) -> DisciplineAlert:
        alert = DisciplineAlert(
            rule=rule, agent=agent, task_id=task_id,
            description=description, severity=severity,
        )
        self._alerts.append(alert)
        self._violation_counts[rule.value] += 1
        self.stats["alerts"] += 1
        if severity >= 2:
            self.stats["violations"] += 1
            logger.warning(f"[纪律] {description}")
        return alert

    def get_alerts(self, severity: int | None = None,
                   agent: str = "") -> list[DisciplineAlert]:
        results = self._alerts
        if severity:
            results = [a for a in results if a.severity >= severity]
        if agent:
            results = [a for a in results if a.agent == agent]
        return results

    def violation_rate(self) -> float:
        total = self.stats["alerts"] + self.stats["violations"]
        return self.stats["violations"] / max(total, 1)

    def summary(self) -> str:
        return (
            f"纪律: {self.stats['alerts']} 告警 "
            f"({self.stats['violations']} 违规) "
            f"违规率={self.violation_rate()*100:.0f}%"
        )


# ═══════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════

def test_message_schema():
    msg = Message(
        msg_id="msg_001",
        msg_type=MessageType.TASK,
        source="planner",
        target="executor",
        payload={"goal": "写代码"},
        trace_id="trace_abc",
    )
    d = msg.to_dict()
    assert d["type"] == "task"
    assert d["source"] == "planner"

    # 验证
    errors = MessageValidator.validate(msg)
    assert len(errors) == 0

    # 回复
    reply = Message.reply_to(msg, {"result": "ok"})
    assert reply.source == "executor"
    assert reply.target == "planner"
    assert reply.correlation_id == "msg_001"

    print(f"Message Schema: ✓")


def test_message_bus():
    bus = MessageBus()
    msg = Message(msg_id="m1", msg_type=MessageType.TASK,
                  source="a", target="b", payload="hello", trace_id="t1")
    assert bus.send(msg)
    assert bus.send(Message(msg_id="", msg_type="", source="", target="", payload="")) == False
    assert len(bus.get_by_trace("t1")) == 1
    print(f"MessageBus: ✓ ({bus.summary()})")


def test_access_control():
    ac = MemoryAccessControl()

    # agent 可以写 pref: 前缀
    assert ac.is_allowed("pref:theme", agent="executor", level="agent")

    # agent 不能写 secret: 前缀（仅 human）
    assert not ac.is_allowed("secret:api_key", agent="executor", level="agent")
    assert ac.is_allowed("secret:api_key", agent="human", level="human")

    # 脱敏
    sanitized = ac.sanitize("my api_key=abc123 and email=test@example.com")
    assert "***" in sanitized
    assert "abc123" not in sanitized

    print(f"MemoryAccessControl: ✓ ({ac.summary()})")


def test_intervention():
    im = InterventionManager()

    # 合法干预
    ok, msg = im.intervene(InterventionPoint.AFTER_ROUTING, "t1",
                           "override_route", "路由分配错误")
    assert ok

    # 非法干预
    ok, msg = im.intervene(InterventionPoint.AFTER_ROUTING, "t1",
                           "bypass_qa", "想跳过质检")
    assert not ok

    # 段5干预
    ok, msg = im.intervene(InterventionPoint.AFTER_QA, "t1",
                           "modify_qa_result", "质检误判")
    assert ok

    print(f"InterventionManager: ✓ ({im.summary()})")


def test_cleanup():
    cm = MemoryCleanupManager()

    # TTL 检查
    expired, action = cm.check_expired("test", "l3", time.time() - 86400 * 200)
    assert action == "delete"

    expired, action = cm.check_expired("test", "l3", time.time() - 86400 * 100)
    assert action == "archive"

    expired, action = cm.check_expired("test", "l3", time.time() - 86400)
    assert action == "keep"

    # 快照
    snap = cm.take_snapshot({"key": "val"}, "pre-cleanup")
    assert snap.id.startswith("snap-")

    # 检索权重
    w = cm.get_retrieval_weight(age=86400 * 200, layer="l3")
    assert w == 0.1

    print(f"MemoryCleanupManager: ✓ ({cm.summary()})")


def test_special_task():
    sta = SpecialTaskAdapter()

    flags = sta.detect_task_type(60000, is_streaming=True)
    assert "long_context" in flags
    assert "streaming" in flags

    flags2 = sta.detect_task_type(10000)
    assert len(flags2) == 0

    print(f"SpecialTaskAdapter: ✓ ({sta.summary()})")


def test_discipline():
    dw = DisciplineWatcher()

    # 跳过质检
    alert = dw.watch("bypass_qa", "executor", "t1")
    assert alert is not None
    assert alert.severity >= 2

    # 强制执行被拒任务
    alert2 = dw.watch("force_execution", "executor", "t2",
                      {"utility": -0.5})
    assert alert2 is not None

    # 正常操作不触发
    alert3 = dw.watch("normal_execute", "executor", "t3")
    assert alert3 is None

    print(f"DisciplineWatcher: ✓ ({dw.summary()})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print("═══ 外围支撑系统扩展测试 ═══\n")

    print("─ 8.1 通信协议Schema ─")
    test_message_schema()
    test_message_bus()
    print()

    print("─ 8.2 记忆访问控制 ─")
    test_access_control()
    print()

    print("─ 8.5 人工干预点 ─")
    test_intervention()
    print()

    print("─ 8.7 记忆过期清理 ─")
    test_cleanup()
    print()

    print("─ 8.9 特殊任务适配 ─")
    test_special_task()
    print()

    print("─ 8.12 手动纪律告警 ─")
    test_discipline()
    print()

    print("\n═══ 全部测试通过 ═══")
