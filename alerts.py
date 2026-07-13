"""
告警推送链路 — 异常/熔断/违规 多渠道通知

三通道：
  WeChat: 通过 cronjob deliver=origin 推微信
  Memory: 写 L1 alert:* 前缀，轮询读取
  Console: 日志输出（兜底）

集成点：
  - CircuitBreaker 熔断 → 自动发告警
  - BudgetMonitor 超限 → 自动发告警
  - DisciplineWatcher 违规 → 自动发告警
  - RecoveryAgent 多次失败 → 自动发告警
"""

from __future__ import annotations

import time
import json
import logging
from enum import IntEnum, Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from datetime import datetime

logger = logging.getLogger("agent-framework.alerts")


# ─── 告警等级 ───────────────────────────────────────────────


class AlertSeverity(IntEnum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


SEVERITY_ICON = {
    AlertSeverity.DEBUG: "🔹",
    AlertSeverity.INFO: "ℹ️",
    AlertSeverity.WARNING: "⚠️",
    AlertSeverity.ERROR: "❌",
    AlertSeverity.CRITICAL: "🚨",
}


class AlertSource(str, Enum):
    CIRCUIT_BREAKER = "circuit_breaker"
    BUDGET = "budget"
    DISCIPLINE = "discipline"
    RECOVERY = "recovery"
    SYSTEM = "system"
    TASK_FAILURE = "task_failure"
    QA_FAILURE = "qa_failure"
    DRIFT = "drift"


# ─── 告警事件 ───────────────────────────────────────────────


@dataclass
class AlertEvent:
    """一条告警事件"""
    title: str
    message: str
    severity: AlertSeverity = AlertSeverity.WARNING
    source: AlertSource = AlertSource.SYSTEM
    source_name: str = ""           # 具体来源名（如熔断器名 "api:deepseek"）
    task_id: str = ""               # 关联任务
    details: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    dedup_key: str = ""             # 去重键（同 key 的告警合并）

    def __post_init__(self):
        if not self.dedup_key:
            self.dedup_key = f"{self.source.value}:{self.source_name}"

    def format_wechat(self) -> str:
        """格式化微信消息"""
        icon = SEVERITY_ICON.get(self.severity, "🔹")
        ts = datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")
        lines = [
            f"{icon} **{self.title}**",
            f"　{self.message}",
            f"　来源: {self.source.value}",
        ]
        if self.task_id:
            lines.append(f"　任务: `{self.task_id[:12]}...`")
        if self.details:
            for k, v in list(self.details.items())[:3]:
                lines.append(f"　{k}: {str(v)[:50]}")
        lines.append(f"　🕐 {ts}")
        return "\n".join(lines)

    def format_log(self) -> str:
        return f"[{self.severity.name}] {self.source.value}: {self.title} — {self.message}"

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "message": self.message,
            "severity": self.severity.name,
            "source": self.source.value,
            "source_name": self.source_name,
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "dedup_key": self.dedup_key,
        }


# ─── 告警管理器 ──────────────────────────────────────────────


class AlertManager:
    """告警管理 — 聚合、去重、分发

    策略：
      - 同 dedup_key 的告警 5 分钟内合并
      - 不同通道的分发器可注册多个
      - 推送频率限制：同源最多 1条/5分钟
    """

    DEDUP_WINDOW = 300  # 5分钟去重窗口

    def __init__(self):
        self._dispatchers: list[Callable[[AlertEvent], None]] = []
        self._history: list[AlertEvent] = []
        self._last_sent: dict[str, float] = {}  # dedup_key → last_send_time
        self._suppressed: set[str] = set()       # 静音的 dedup_key
        self.stats = {"total": 0, "sent": 0, "suppressed": 0, "deduped": 0}

    def register_dispatcher(self, fn: Callable[[AlertEvent], None]) -> None:
        """注册分发器

        Args:
            fn: 接收 AlertEvent，将告警推送到对应渠道
        """
        self._dispatchers.append(fn)

    def emit(self, event: AlertEvent) -> bool:
        """发射一条告警

        Returns:
            True=已发送, False=被去重/静音
        """
        self._history.append(event)
        self.stats["total"] += 1

        # 静音检查
        if event.dedup_key in self._suppressed:
            self.stats["suppressed"] += 1
            return False

        # 去重检查：同 key 5分钟内已发过
        last_time = self._last_sent.get(event.dedup_key, 0)
        if time.time() - last_time < self.DEDUP_WINDOW:
            self.stats["deduped"] += 1
            return False

        # 分发
        self._last_sent[event.dedup_key] = time.time()
        self.stats["sent"] += 1

        for dispatcher in self._dispatchers:
            try:
                dispatcher(event)
            except Exception as e:
                logger.error(f"告警分发失败: {e}")

        return True

    def suppress(self, dedup_key: str) -> None:
        """静音某个告警源"""
        self._suppressed.add(dedup_key)

    def unsuppress(self, dedup_key: str) -> None:
        self._suppressed.discard(dedup_key)

    def get_history(self, severity: AlertSeverity | None = None,
                    source: AlertSource | None = None,
                    limit: int = 20) -> list[AlertEvent]:
        results = self._history
        if severity:
            results = [e for e in results if e.severity >= severity]
        if source:
            results = [e for e in results if e.source == source]
        return results[-limit:]

    def summary(self) -> str:
        s = self.stats
        return f"告警: 总计{s['total']} 已发送{s['sent']} 去重{s['deduped']} 静音{s['suppressed']}"


# ─── 分发器实现 ──────────────────────────────────────────────


class WeChatDispatcher:
    """微信分发器 — 通过 memory 写 + cron 轮询推微信"""

    MEMORY_PREFIX = "alert:wechat:"

    def __call__(self, event: AlertEvent) -> None:
        """写入 memory，cron job 轮询推送"""
        content = json.dumps(event.to_dict(), ensure_ascii=False)
        try:
            from hermes_tools import memory as hermes_memory
            hermes_memory(
                action="add",
                target="memory",
                content=f"{self.MEMORY_PREFIX}{event.dedup_key}: {content}",
            )
            logger.info(f"告警写入 memory: {event.dedup_key}")
        except ImportError:
            logger.warning(f"告警无法推送 (无 hermes_tools): {event.title}")


class ConsoleDispatcher:
    """控制台分发器 — 日志输出"""

    def __call__(self, event: AlertEvent) -> None:
        level_map = {
            AlertSeverity.DEBUG: logging.DEBUG,
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }
        logger.log(level_map.get(event.severity, logging.INFO), event.format_log())


# ─── cron 拉取脚本 ──────────────────────────────────────────


CRON_PULL_SCRIPT = r'''
"""告警轮询推送脚本 — 由 cron job 定时执行
读取 memory 中 alert:wechat: 前缀的新条目，汇总推微信。

用法:
  cronjob action=create name="告警推送" schedule="every 5m" \
    script="D:\agent_framework\alert_pull.py" no_agent=True
"""

import sys, json
sys.path.insert(0, "D:")

try:
    from agent_framework.hermes_memory import l3_search

    results = l3_search("alert:wechat:", limit=10)
    if not results:
        exit(0)  # 无告警时静默

    alerts = []
    for r in results:
        snippet = r.get("snippet", "")
        # 解析 JSON 内容
        if "alert:wechat:" in snippet:
            # 提取 JSON 部分
            try:
                json_str = snippet.split(": ", 1)[1] if ": " in snippet else snippet
                data = json.loads(json_str)
                alerts.append(data)
            except (json.JSONDecodeError, IndexError):
                alerts.append({"title": "未知告警", "message": snippet[:100]})

    if alerts:
        print(f"📢 **告警汇总 ({len(alerts)} 条)**")
        print()
        for a in alerts[-5:]:  # 最多5条
            icon = {"WARNING": "⚠️", "ERROR": "❌", "CRITICAL": "🚨"}.get(
                a.get("severity", "WARNING"), "🔹")
            print(f"{icon} {a.get('title', '未知')}")
            print(f"  {a.get('message', '')}")
            print()

except Exception as e:
    print(f"❌ 告警轮询异常: {e}")
    exit(1)
'''


# ═══════════════════════════════════════════════════════════
# 集成：自动挂载到框架组件
# ═══════════════════════════════════════════════════════════


class AlertIntegration:
    """告警集成 — 自动挂载到 CircuitBreaker/BudgetMonitor 等组件"""

    def __init__(self):
        self.manager = AlertManager()
        self.manager.register_dispatcher(ConsoleDispatcher())
        self.manager.register_dispatcher(WeChatDispatcher())

    def watch_circuit_breaker(self, cb) -> None:
        """挂载到熔断器"""
        original_call = cb.call

        def wrapped_call(fn, *args, **kwargs):
            try:
                return original_call(fn, *args, **kwargs)
            except Exception as e:
                if cb.state == "open":
                    self.manager.emit(AlertEvent(
                        title=f"熔断开启: {cb.name}",
                        message=f"连续 {cb.failure_count} 次失败，熔断 {cb.recovery_timeout:.0f}s",
                        severity=AlertSeverity.ERROR,
                        source=AlertSource.CIRCUIT_BREAKER,
                        source_name=cb.name,
                        details={"failures": cb.failure_count, "timeout": cb.recovery_timeout},
                        dedup_key=f"cb:{cb.name}",
                    ))
                raise

        cb.call = wrapped_call

    def watch_budget(self, bm) -> None:
        """挂载到预算监控"""
        original_check = bm.check

        def wrapped_check(*args, **kwargs):
            status = original_check(*args, **kwargs)
            if status["level"] in ("critical", "hard_limit"):
                self.manager.emit(AlertEvent(
                    title=f"预算 {status['level']}: {status['recommendation']}",
                    message=f"小时使用率={status['ratios']['hour']*100:.0f}%",
                    severity=AlertSeverity.WARNING if status["level"] == "critical" else AlertSeverity.ERROR,
                    source=AlertSource.BUDGET,
                    details=status["ratios"],
                    dedup_key="budget:monitor",
                ))
            return status

        bm.check = wrapped_check

    def watch_discipline(self, dw) -> None:
        """挂载到纪律告警"""
        original_watch = dw.watch

        def wrapped_watch(action, agent, task_id, context=None):
            alert = original_watch(action, agent, task_id, context)
            if alert and alert.severity >= 2:
                self.manager.emit(AlertEvent(
                    title=f"纪律违规: {alert.rule.value}",
                    message=f"Agent {agent} 在 {task_id}: {alert.description[:80]}",
                    severity=AlertSeverity.WARNING if alert.severity == 2 else AlertSeverity.ERROR,
                    source=AlertSource.DISCIPLINE,
                    source_name=agent,
                    task_id=task_id,
                    dedup_key=f"discipline:{agent}:{alert.rule.value}",
                ))
            return alert

        dw.watch = wrapped_watch

    def watch_recovery(self, ra) -> None:
        """挂载到 Recovery Agent"""
        original_recover = ra.recover

        def wrapped_recover(task_id, error, rollback_level=1, context=None):
            result = original_recover(task_id, error, rollback_level, context)
            attempts = ra._attempts.get(task_id, 0)
            if result["action"] == "escalate" or attempts >= 2:
                self.manager.emit(AlertEvent(
                    title=f"任务恢复失败: {task_id[:12]}",
                    message=f"尝试 {attempts} 次后 {result['action']}: {error[:60]}",
                    severity=AlertSeverity.ERROR,
                    source=AlertSource.RECOVERY,
                    source_name=task_id[:12],
                    task_id=task_id,
                    details={"attempts": attempts, "action": result["action"]},
                    dedup_key=f"recovery:{task_id}",
                ))
            return result

        ra.recover = wrapped_recover

    def emit(self, title: str, message: str,
             severity: AlertSeverity = AlertSeverity.WARNING,
             source: AlertSource = AlertSource.SYSTEM,
             **kwargs) -> bool:
        """快捷发射告警"""
        return self.manager.emit(AlertEvent(
            title=title, message=message,
            severity=severity, source=source, **kwargs,
        ))


# ═══════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════

def test_alert_event():
    event = AlertEvent(
        title="API 熔断",
        message="DeepSeek 连续 5 次超时",
        severity=AlertSeverity.ERROR,
        source=AlertSource.CIRCUIT_BREAKER,
        source_name="api:deepseek",
        details={"failures": 5, "timeout": 300},
    )
    assert event.dedup_key == "circuit_breaker:api:deepseek"
    assert "❌" in event.format_wechat()
    print(f"AlertEvent 格式化: ✓")


def test_alert_manager():
    am = AlertManager()
    am.register_dispatcher(ConsoleDispatcher())

    # 发送
    sent = am.emit(AlertEvent("测试", "测试告警", severity=AlertSeverity.INFO,
                               source=AlertSource.SYSTEM, dedup_key="test:1"))
    assert sent
    assert am.stats["sent"] == 1

    # 去重（同 key 5分钟内不发）
    sent2 = am.emit(AlertEvent("测试2", "重复", dedup_key="test:1"))
    assert not sent2
    assert am.stats["deduped"] == 1

    # 不同 key 可发
    sent3 = am.emit(AlertEvent("测试3", "新的", dedup_key="test:2"))
    assert sent3

    print(f"AlertManager: ✓ ({am.summary()})")


def test_integration():
    """测试集成挂载"""
    from supports import CircuitBreaker, BudgetMonitor, RecoveryAgent
    from supports_ext import DisciplineWatcher

    ai = AlertIntegration()

    # 挂载熔断器
    cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1)
    ai.watch_circuit_breaker(cb)
    try:
        cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
    except ValueError:
        pass
    assert ai.manager.stats["sent"] >= 1
    print(f"熔断器集成: ✓")

    # 挂载预算
    bm = BudgetMonitor()
    ai.watch_budget(bm)
    bm.record_usage("big", 999999)
    status = bm.check(estimated_cost=50000)
    assert status["level"] in ("critical", "hard_limit")
    print(f"预算集成: ✓ (level={status['level']})")

    # 挂载纪律
    dw = DisciplineWatcher()
    ai.watch_discipline(dw)
    dw.watch("bypass_qa", "executor", "t1")
    assert ai.manager.stats["sent"] >= 2
    print(f"纪律集成: ✓")

    # 快捷发射
    ai.emit("快捷告警", "直接发一条", severity=AlertSeverity.WARNING)
    print(f"快捷发射: ✓")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print("═══ 告警推送链路测试 ═══\n")
    test_alert_event()
    test_alert_manager()
    print()
    test_integration()
    print("\n═══ 全部通过 ═══")
