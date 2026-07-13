"""
统一状态机 — 多Agent协作框架的任务生命周期管理

任务生命周期：
  INIT → PLANNED → EXECUTING → QA_CHECKING → MEMORY_WRITING → FINALIZED
                                                                  FAILED → RECOVERED

设计原则：
  1. 所有转换由事件驱动，经仲裁器确认
  2. 三种回滚等级（Level 1重执行 / Level 2重规划 / Level 3熔断）
  3. 状态转换可追踪、可审计、可回放

Usage:
    from agent_framework.state_machine import TaskStateMachine, TaskState, Event

    sm = TaskStateMachine(task_id="task_001")
    sm.transition(Event.PLAN_COMPLETE)
    sm.transition(Event.EXECUTE_COMPLETE)
    sm.state  # TaskState.QA_CHECKING
"""

from __future__ import annotations

import time
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("agent-framework.state-machine")


# ─── 基元定义 ─────────────────────────────────────────────────


class TaskState(str, Enum):
    """任务状态 — 生命周期全阶段"""
    INIT = "init"                     # 初始态：任务刚创建
    PLANNING = "planning"             # 规划中：Planner 正在拆解
    PLANNED = "planned"               # 规划完成：DAG 已生成
    EXECUTING = "executing"           # 执行中
    QA_CHECKING = "qa_checking"       # 质检中
    MEMORY_WRITING = "memory_writing" # 记忆写入中
    FINALIZED = "finalized"           # 完成态：成功结束
    FAILED = "failed"                 # 失败态：无法恢复
    RECOVERING = "recovering"         # 恢复中
    RECOVERED = "recovered"           # 已恢复
    REJECTED = "rejected"             # 已拒绝（Arbiter判定不值得做）
    CANCELLED = "cancelled"           # 已取消（用户手动取消）
    BLOCKED = "blocked"               # 阻塞态：等待外部条件

    def is_terminal(self) -> bool:
        return self in (TaskState.FINALIZED, TaskState.FAILED,
                        TaskState.REJECTED, TaskState.CANCELLED)

    def is_active(self) -> bool:
        return not self.is_terminal()


class Event(str, Enum):
    """触发状态转换的事件"""
    # 正向流程
    TASK_RECEIVED = "task_received"           # 任务已接收
    PLAN_COMPLETE = "plan_complete"           # 规划完成
    EXECUTE_START = "execute_start"           # 开始执行
    EXECUTE_COMPLETE = "execute_complete"     # 执行完成
    QA_PASS = "qa_pass"                       # 质检通过
    QA_FAIL = "qa_fail"                       # 质检不通过
    MEMORY_WRITE_DONE = "memory_write_done"   # 记忆写入完成
    FINALIZE = "finalize"                     # 终态

    # 异常流程
    SAFETY_BREACH = "safety_breach"           # 安全违规
    ERROR_OCCURRED = "error_occurred"         # 错误发生
    TIMEOUT = "timeout"                       # 超时
    BUDGET_OVERFLOW = "budget_overflow"       # 预算超限
    LOW_UTILITY = "low_utility"               # Utility 不足

    # 恢复
    RECOVERY_START = "recovery_start"         # 开始恢复
    RECOVERY_COMPLETE = "recovery_complete"   # 恢复完成
    RECOVERY_FAILED = "recovery_failed"       # 恢复失败

    # 人工干预
    USER_CANCEL = "user_cancel"               # 用户取消
    USER_BYPASS = "user_bypass"               # 用户绕过（强制通过）
    BLOCK = "block"                           # 阻塞
    UNBLOCK = "unblock"                       # 解除阻塞


class RollbackLevel(Enum):
    """回滚等级"""
    LEVEL_1 = 1   # 回重执行（重新跑失败的节点）
    LEVEL_2 = 2   # 回重规划（重新拆解任务）
    LEVEL_3 = 3   # 熔断转人工


# ─── 状态转换表 ──────────────────────────────────────────────


class StateMachineError(Exception):
    """状态机异常"""
    pass


class InvalidTransitionError(StateMachineError):
    """非法状态转换"""
    pass


# 状态转换映射表：{ (当前状态, 事件): 目标状态 }
TRANSITION_TABLE: dict[tuple[TaskState, Event], TaskState] = {
    # 初始态
    (TaskState.INIT, Event.TASK_RECEIVED): TaskState.PLANNING,

    # 规划
    (TaskState.PLANNING, Event.PLAN_COMPLETE): TaskState.PLANNED,
    (TaskState.PLANNING, Event.ERROR_OCCURRED): TaskState.FAILED,
    (TaskState.PLANNING, Event.SAFETY_BREACH): TaskState.FAILED,

    # 规划完成
    (TaskState.PLANNED, Event.EXECUTE_START): TaskState.EXECUTING,
    (TaskState.PLANNED, Event.LOW_UTILITY): TaskState.REJECTED,
    (TaskState.PLANNED, Event.USER_CANCEL): TaskState.CANCELLED,
    (TaskState.PLANNED, Event.BUDGET_OVERFLOW): TaskState.REJECTED,

    # 执行中
    (TaskState.EXECUTING, Event.EXECUTE_COMPLETE): TaskState.QA_CHECKING,
    (TaskState.EXECUTING, Event.TIMEOUT): TaskState.FAILED,
    (TaskState.EXECUTING, Event.SAFETY_BREACH): TaskState.FAILED,
    (TaskState.EXECUTING, Event.ERROR_OCCURRED): TaskState.FAILED,
    (TaskState.EXECUTING, Event.BUDGET_OVERFLOW): TaskState.FAILED,

    # 质检中
    (TaskState.QA_CHECKING, Event.QA_PASS): TaskState.MEMORY_WRITING,
    (TaskState.QA_CHECKING, Event.QA_FAIL): TaskState.EXECUTING,  # Level 1 回滚
    (TaskState.QA_CHECKING, Event.SAFETY_BREACH): TaskState.FAILED,
    (TaskState.QA_CHECKING, Event.USER_BYPASS): TaskState.MEMORY_WRITING,

    # 记忆写入中
    (TaskState.MEMORY_WRITING, Event.MEMORY_WRITE_DONE): TaskState.FINALIZED,
    (TaskState.MEMORY_WRITING, Event.ERROR_OCCURRED): TaskState.FAILED,
    (TaskState.MEMORY_WRITING, Event.TIMEOUT): TaskState.FAILED,

    # 失败态
    (TaskState.FAILED, Event.RECOVERY_START): TaskState.RECOVERING,
    (TaskState.FAILED, Event.USER_CANCEL): TaskState.CANCELLED,

    # 恢复中
    (TaskState.RECOVERING, Event.RECOVERY_COMPLETE): TaskState.EXECUTING,  # Level 1
    (TaskState.RECOVERING, Event.RECOVERY_COMPLETE): TaskState.PLANNING,   # Level 2 (也有到PLANNING的)
    (TaskState.RECOVERING, Event.RECOVERY_FAILED): TaskState.FAILED,

    # 阻塞态
    (TaskState.BLOCKED, Event.UNBLOCK): TaskState.EXECUTING,
    (TaskState.BLOCKED, Event.USER_CANCEL): TaskState.CANCELLED,
    (TaskState.BLOCKED, Event.TIMEOUT): TaskState.FAILED,

    # 已恢复
    (TaskState.RECOVERED, Event.RECOVERY_COMPLETE): TaskState.EXECUTING,

    # RECOVERING + RECOVERY_COMPLETE -> 根据回滚等级决定目标
    # 这个由 go_to 直接处理
}

# 特殊处理：RECOVERING 根据回滚等级走不同分支
_RECOVERY_TARGETS = {
    RollbackLevel.LEVEL_1: TaskState.EXECUTING,  # 回重执行
    RollbackLevel.LEVEL_2: TaskState.PLANNING,   # 回重规划
}


# ─── 状态机核心 ─────────────────────────────────────────────


class TaskStateMachine:
    """统一任务状态机 — 事件驱动，支持回滚和审计"""

    def __init__(self, task_id: str, initial_state: TaskState = TaskState.INIT):
        self.task_id = task_id
        self.state: TaskState = initial_state
        self.history: list[StateTransition] = []
        self.rollback_level: RollbackLevel | None = None
        self.created_at: float = time.time()
        self.updated_at: float = time.time()
        self.completed_at: float | None = None
        self.context: dict[str, Any] = {}  # 附加上下文

    @property
    def is_active(self) -> bool:
        return self.state.is_active()

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    @property
    def duration(self) -> float | None:
        if self.completed_at:
            return self.completed_at - self.created_at
        return time.time() - self.created_at

    def can_transition(self, event: Event) -> bool:
        """检查是否可以进行该转换"""
        # RECOVERING 特殊处理
        if self.state == TaskState.RECOVERING and event == Event.RECOVERY_COMPLETE:
            return True
        return (self.state, event) in TRANSITION_TABLE

    def valid_events(self) -> list[Event]:
        """返回当前状态下所有合法事件"""
        return [e for s, e in TRANSITION_TABLE if s == self.state]

    def transition(self, event: Event, reason: str = "",
                   arbiter_approved: bool = True,
                   metadata: dict | None = None) -> StateTransition:
        """执行状态转换

        Args:
            event: 触发事件
            reason: 转换原因
            arbiter_approved: 是否已获仲裁器批准
            metadata: 附加元数据

        Returns:
            转换记录

        Raises:
            InvalidTransitionError: 非法转换
        """
        target = self._get_target_state(event)

        if target is None:
            raise InvalidTransitionError(
                f"非法转换: {self.state.value} → {event.value} "
                f"(任务 {self.task_id})")

        if self.state.is_terminal() and self.state != TaskState.FAILED:
            raise InvalidTransitionError(
                f"任务 {self.task_id} 已到达终态 {self.state.value}，无法转换")

        old_state = self.state
        self.state = target
        self.updated_at = time.time()

        if self.state.is_terminal():
            self.completed_at = time.time()

        transition = StateTransition(
            task_id=self.task_id,
            from_state=old_state,
            to_state=target,
            event=event,
            reason=reason,
            timestamp=self.updated_at,
            arbiter_approved=arbiter_approved,
            metadata=metadata or {},
        )
        self.history.append(transition)

        logger.info(
            f"状态转换: {old_state.value} → {target.value} "
            f"({event.value}) [{reason}]"
        )
        return transition

    def _get_target_state(self, event: Event) -> TaskState | None:
        """获取转换目标状态"""
        # RECOVERING 特殊分支
        if self.state == TaskState.RECOVERING and event == Event.RECOVERY_COMPLETE:
            if self.rollback_level in _RECOVERY_TARGETS:
                return _RECOVERY_TARGETS[self.rollback_level]
            return TaskState.EXECUTING  # 默认回执行

        return TRANSITION_TABLE.get((self.state, event))

    def transition_or_raise(self, event: Event, reason: str = "",
                            arbiter_approved: bool = True) -> StateTransition:
        """强制转换（用户 bypass 场景）"""
        return self.transition(event, reason, arbiter_approved=True)

    def rollback(self, level: RollbackLevel, reason: str = "") -> StateTransition:
        """触发回滚

        Level 1: 回重执行（重新跑失败的节点）
        Level 2: 回重规划（重新拆解任务）
        Level 3: 熔断转人工
        """
        self.rollback_level = level

        if level == RollbackLevel.LEVEL_1:
            # 从当前状态回 EXECUTING 重新执行
            return self.transition(
                Event.ERROR_OCCURRED,
                reason=f"Level 1 回滚: {reason}",
            )
        elif level == RollbackLevel.LEVEL_2:
            # 回到 PLANNING 重新规划
            return self.transition(
                Event.QA_FAIL,
                reason=f"Level 2 回滚（重新规划）: {reason}",
            )
        elif level == RollbackLevel.LEVEL_3:
            # 熔断转人工
            return self.transition(
                Event.SAFETY_BREACH,
                reason=f"Level 3 熔断: {reason}",
            )

    def reset(self) -> None:
        """重置到初始状态"""
        self.state = TaskState.INIT
        self.rollback_level = None
        self.completed_at = None
        self.updated_at = time.time()
        logger.info(f"状态机重置: {self.task_id} → INIT")

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "state": self.state.value,
            "is_active": self.is_active,
            "is_terminal": self.is_terminal,
            "duration": self.duration,
            "rollback_level": self.rollback_level.value if self.rollback_level else None,
            "history_count": len(self.history),
            "history": [h.to_dict() for h in self.history[-10:]],  # 最近10条
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ─── 转换记录 ─────────────────────────────────────────────────


@dataclass
class StateTransition:
    """单次状态转换记录"""
    task_id: str
    from_state: TaskState
    to_state: TaskState
    event: Event
    reason: str
    timestamp: float
    arbiter_approved: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "from": self.from_state.value,
            "to": self.to_state.value,
            "event": self.event.value,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "arbiter_approved": self.arbiter_approved,
        }

    def __repr__(self) -> str:
        return (f"<Transition {self.task_id}: "
                f"{self.from_state.value} → {self.to_state.value} "
                f"({self.event.value})>")


# ─── 状态机包装器：关联 DAG ──────────────────────────────────


class DagStateMachine:
    """将 DAG 执行和状态机绑定

    一个 TaskGraph + TaskStateMachine 的组合体。
    DAG 执行中的关键节点自动触发状态机转换。
    """

    def __init__(self, task_id: str, graph, arbiter: Any | None = None):
        self.task_id = task_id
        self.graph = graph
        self.state_machine = TaskStateMachine(task_id)
        self.arbiter = arbiter
        self.log: list[dict] = []

    def on_event(self, event: Event, reason: str = "",
                 arbiter_check: bool = True) -> StateTransition | None:
        """事件处理（带可选的仲裁器检查）"""
        if arbiter_check and self.arbiter:
            # 请求仲裁器批准转换
            approved, arbiter_reason = self.arbiter.approve_transition(
                self.state_machine.state, event, self.graph)
            if not approved:
                logger.warning(f"仲裁器拒绝转换: {event.value} — {arbiter_reason}")
                return None

        return self.state_machine.transition(event, reason)

    def start(self) -> None:
        """启动任务"""
        self.on_event(Event.TASK_RECEIVED, "任务接收")
        self.on_event(Event.PLAN_COMPLETE, "规划完成")

    def run(self, executor) -> dict[str, Any]:
        """完整运行：规划 → 执行 → 质检 → 记忆写入 → 完成

        Args:
            executor: 执行器对象，需有 run(graph) 和 qa_check(graph) 方法

        Returns:
            graph 执行结果
        """
        self.start()
        self.log_event("start", "任务启动")

        # 执行
        self.on_event(Event.EXECUTE_START, "开始执行")
        try:
            results = executor.run(self.graph)
        except Exception as e:
            self.state_machine.rollback(RollbackLevel.LEVEL_1, str(e))
            self.log_event("execution_failed", str(e))
            return results if 'results' in dir() else {}

        self.graph.execute_completed = True  # 标记执行完成
        self.on_event(Event.EXECUTE_COMPLETE, "执行完成")

        # 质检
        try:
            qa_result = executor.qa_check(self.graph)
        except Exception as e:
            qa_result = {"passed": False, "issues": [str(e)]}

        if qa_result.get("passed", False):
            self.on_event(Event.QA_PASS, "质检通过")
        else:
            self.on_event(Event.QA_FAIL, f"质检未通过: {qa_result.get('issues', [])}")
            self.log_event("qa_failed", str(qa_result.get("issues", [])))
            return self.graph._result or {}

        # 记忆写入
        self.on_event(Event.MEMORY_WRITE_DONE, "记忆写入完成")
        self.on_event(Event.FINALIZE, "任务完成")
        self.log_event("completed", "任务成功完成")

        return self.graph._result or {}

    def log_event(self, event_type: str, message: str) -> None:
        self.log.append({
            "type": event_type,
            "message": message,
            "timestamp": time.time(),
            "state": self.state_machine.state.value,
        })


# ─── 测试 ──────────────────────────────────────────────────────


def test_basic_transitions():
    """基本状态转换测试"""
    sm = TaskStateMachine("test_001")

    assert sm.state == TaskState.INIT
    assert sm.is_active
    assert not sm.is_terminal

    # 正向流程
    assert sm.can_transition(Event.TASK_RECEIVED)
    sm.transition(Event.TASK_RECEIVED, "收到任务")
    assert sm.state == TaskState.PLANNING

    sm.transition(Event.PLAN_COMPLETE, "规划完成")
    assert sm.state == TaskState.PLANNED

    sm.transition(Event.EXECUTE_START, "开始执行")
    assert sm.state == TaskState.EXECUTING

    sm.transition(Event.EXECUTE_COMPLETE, "执行完成")
    assert sm.state == TaskState.QA_CHECKING

    sm.transition(Event.QA_PASS, "质检通过")
    assert sm.state == TaskState.MEMORY_WRITING

    sm.transition(Event.MEMORY_WRITE_DONE, "记忆写入完成")
    assert sm.state == TaskState.FINALIZED
    assert sm.is_terminal
    assert sm.completed_at is not None

    print(f"正向流程: ✓ ({len(sm.history)} 次转换)")
    return sm


def test_invalid_transition():
    """非法转换测试"""
    sm = TaskStateMachine("test_002")
    sm.transition(Event.TASK_RECEIVED, "接收")  # INIT → PLANNING

    try:
        sm.transition(Event.MEMORY_WRITE_DONE, "不能直接写记忆")
        assert False, "应该抛出 InvalidTransitionError"
    except InvalidTransitionError:
        print("非法转换检测: ✓")


def test_fail_recover():
    """失败与恢复测试"""
    sm = TaskStateMachine("test_003")

    # 正向到执行
    sm.transition(Event.TASK_RECEIVED, "接收")
    sm.transition(Event.PLAN_COMPLETE, "规划")
    sm.transition(Event.EXECUTE_START, "开始执行")

    # 执行时出错
    sm.transition(Event.ERROR_OCCURRED, "执行出错")
    assert sm.state == TaskState.FAILED

    # 恢复
    sm.transition(Event.RECOVERY_START, "开始恢复")
    assert sm.state == TaskState.RECOVERING

    sm.rollback_level = RollbackLevel.LEVEL_1
    sm.transition(Event.RECOVERY_COMPLETE, "恢复完成 (Level 1)")
    assert sm.state == TaskState.EXECUTING, f"应回 EXECUTING, 实际 {sm.state}"

    print(f"失败恢复 (Level 1): ✓")


def test_qa_rollback():
    """质检回滚测试"""
    sm = TaskStateMachine("test_004")

    sm.transition(Event.TASK_RECEIVED, "接收")
    sm.transition(Event.PLAN_COMPLETE, "规划")
    sm.transition(Event.EXECUTE_START, "开始执行")
    sm.transition(Event.EXECUTE_COMPLETE, "执行完成")
    assert sm.state == TaskState.QA_CHECKING

    # 质检不通过 → 回 EXECUTING (Level 1)
    sm.transition(Event.QA_FAIL, "质检不通过，重新执行")
    assert sm.state == TaskState.EXECUTING

    print(f"质检回滚: ✓")


def test_level2_rollback():
    """Level 2 回滚测试（重新规划）"""
    # 路径: EXECUTING → FAILED → RECOVERING → PLANNING
    sm = TaskStateMachine("test_005")
    sm.transition(Event.TASK_RECEIVED, "接收")
    sm.transition(Event.PLAN_COMPLETE, "规划完成")
    sm.transition(Event.EXECUTE_START, "开始执行")
    sm.transition(Event.ERROR_OCCURRED, "执行出错")
    assert sm.state == TaskState.FAILED

    sm.transition(Event.RECOVERY_START, "开始 Level 2 恢复")
    assert sm.state == TaskState.RECOVERING

    sm.rollback_level = RollbackLevel.LEVEL_2
    sm.transition(Event.RECOVERY_COMPLETE, "恢复完成 (Level 2)")
    assert sm.state == TaskState.PLANNING, f"Level 2 应回 PLANNING, 实际 {sm.state}"

    print(f"Level 2 回滚: ✓")


def test_terminal_state():
    """终态不可转换测试"""
    sm = TaskStateMachine("test_006")
    sm.transition(Event.TASK_RECEIVED, "接收")
    sm.transition(Event.PLAN_COMPLETE, "规划")
    sm.transition(Event.LOW_UTILITY, "Utility 不足")
    assert sm.state == TaskState.REJECTED
    assert sm.is_terminal

    try:
        sm.transition(Event.TASK_RECEIVED, "不能再转换")
        assert False, "终态应拒绝转换"
    except InvalidTransitionError:
        print("终态锁定: ✓")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("═══ 状态机测试 ═══\n")
    test_basic_transitions()
    print()
    test_invalid_transition()
    print()
    test_fail_recover()
    print()
    test_qa_rollback()
    print()
    test_level2_rollback()
    print()
    test_terminal_state()
    print("\n═══ 全部测试通过 ═══")
