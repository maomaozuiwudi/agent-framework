"""
统一仲裁器 — 多Agent协作框架的决策核心

仲裁器解决所有跨Agent冲突，不进模块内部协商。

设计原则：
  1. 优先级链焊死：Safety > Correctness > QA > Budget > Performance
  2. Utility = TaskValue - Cost - RiskPenalty 动态判定值不值得做
  3. 不枚举规则 — 用优先级链 + Utility 函数推导决策
  4. 无法裁决 → 转人工

Usage:
    from agent_framework.arbiter import Arbiter, ConflictEvent, ArbiterDecision

    arbiter = Arbiter()
    decision = arbiter.resolve(
        ConflictEvent(
            type="qa_vs_executor",
            source_a="critic",
            source_b="executor",
            claim_a="代码有bug",
            claim_b="按设计实现",
        )
    )
    # decision: {resolution: "qa_overrides_executor", reason: "...", confidence: 0.85}
"""

from __future__ import annotations

import time
import logging
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("agent-framework.arbiter")


# ─── 优先级体系 ──────────────────────────────────────────────


class PriorityDomain(IntEnum):
    """优先级域 — 数值越低优先级越高"""
    SAFETY = 0          # 安全 — 最高优先级
    CORRECTNESS = 1     # 正确性
    QA_SIGNAL = 2       # 质检信号
    BUDGET = 3          # 预算
    PERFORMANCE = 4     # 性能 — 最低优先级


PRIORITY_CHAIN = [
    PriorityDomain.SAFETY,
    PriorityDomain.CORRECTNESS,
    PriorityDomain.QA_SIGNAL,
    PriorityDomain.BUDGET,
    PriorityDomain.PERFORMANCE,
]

PRIORITY_NAMES = {
    PriorityDomain.SAFETY: "安全",
    PriorityDomain.CORRECTNESS: "正确性",
    PriorityDomain.QA_SIGNAL: "质检",
    PriorityDomain.BUDGET: "预算",
    PriorityDomain.PERFORMANCE: "性能",
}


# ─── 冲突类型 ─────────────────────────────────────────────────


class ConflictType(str, Enum):
    """仲裁事件类型"""
    QA_VS_EXECUTOR = "qa_vs_executor"           # 质检 vs 执行
    BUDGET_VS_QUALITY = "budget_vs_quality"      # 预算 vs 质量
    SAFETY_BREACH = "safety_breach"              # 安全违规
    AGENT_CONFLICT = "agent_conflict"            # Agent间输出不一致
    TRANSITION_DENIED = "transition_denied"      # 状态转换被拒
    ROLLBACK_DECISION = "rollback_decision"      # 回滚决策
    RESOURCE_CONTENTION = "resource_contention"  # 资源争用
    POLICY_CONFLICT = "policy_conflict"          # 策略冲突
    UTILITY_EVALUATION = "utility_evaluation"    # Utility 评估
    MANUAL_ESCALATION = "manual_escalation"      # 转人工


# 冲突类型的默认优先级映射
CONFLICT_PRIORITY_MAP = {
    ConflictType.SAFETY_BREACH: PriorityDomain.SAFETY,
    ConflictType.QA_VS_EXECUTOR: PriorityDomain.QA_SIGNAL,
    ConflictType.BUDGET_VS_QUALITY: PriorityDomain.BUDGET,
    ConflictType.AGENT_CONFLICT: PriorityDomain.CORRECTNESS,
    ConflictType.TRANSITION_DENIED: PriorityDomain.QA_SIGNAL,
    ConflictType.ROLLBACK_DECISION: PriorityDomain.SAFETY,
    ConflictType.RESOURCE_CONTENTION: PriorityDomain.PERFORMANCE,
    ConflictType.POLICY_CONFLICT: PriorityDomain.CORRECTNESS,
    ConflictType.UTILITY_EVALUATION: PriorityDomain.BUDGET,
    ConflictType.MANUAL_ESCALATION: PriorityDomain.SAFETY,
}


# ─── 数据模型 ─────────────────────────────────────────────────


@dataclass
class ConflictEvent:
    """需仲裁的冲突事件"""
    type: ConflictType | str          # 冲突类型
    source_a: str                     # 冲突方 A（Agent 名称）
    source_b: str                     # 冲突方 B（Agent 名称）
    claim_a: str                      # A 的主张
    claim_b: str                      # B 的主张
    context: dict[str, Any] = field(default_factory=dict)  # 附加上下文
    task_id: str = ""                 # 关联任务 ID
    priority_hint: PriorityDomain | None = None  # 优先级提示
    timestamp: float = field(default_factory=time.time)


@dataclass
class ArbiterDecision:
    """仲裁决策结果"""
    conflict_type: ConflictType | str
    resolution: str                   # 裁决结论
    reason: str                       # 裁决理由
    winning_side: str | None          # 胜方（None 表示平局转人工）
    priority_applied: PriorityDomain  # 应用的优先级
    confidence: float = 1.0           # 置信度 0-1
    action: str = ""                  # 建议动作
    trace_id: str = ""                # 追踪 ID
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "conflict_type": self.conflict_type.value if isinstance(self.conflict_type, ConflictType) else self.conflict_type,
            "resolution": self.resolution,
            "reason": self.reason,
            "winning_side": self.winning_side,
            "priority_applied": self.priority_applied.name,
            "confidence": self.confidence,
            "action": self.action,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
        }


# ─── Utility 成本模型 ────────────────────────────────────────


@dataclass
class UtilityInput:
    """Utility 函数输入"""
    task_value: float = 1.0           # 任务价值 (0-1)
    cost_tokens: int = 0              # 预估 token 数
    cost_dollars: float = 0.0         # 预估费用（可选）
    risk_penalty: float = 0.0         # 风险惩罚 (0-1)
    task_urgency: float = 0.5         # 任务紧急度 (0-1)
    expected_impact: float = 0.5      # 预期影响 (0-1)

    @property
    def normalized_cost(self) -> float:
        """归一化成本 (0-1)，假设 10K token = 1.0"""
        return min(1.0, self.cost_tokens / 10000)


@dataclass
class UtilityResult:
    """Utility 计算结果"""
    utility: float                    # 最终效用值
    is_worthwhile: bool               # 是否值得做
    recommendation: str               # 建议
    value_component: float            # 价值分量
    cost_component: float             # 成本分量
    risk_component: float             # 风险分量
    details: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "utility": self.utility,
            "is_worthwhile": self.is_worthwhile,
            "recommendation": self.recommendation,
            "value_component": self.value_component,
            "cost_component": self.cost_component,
            "risk_component": self.risk_component,
        }


def compute_utility(input: UtilityInput) -> UtilityResult:
    """Utility = TaskValue - Cost - RiskPenalty

    预算阈值：
      Utility > 0.3 → 维持全流程
      Utility 0~0.3 → 压缩质检/检索
      Utility < 0   → 快速通道或拒绝
    """
    # 价值分量：任务价值 + 紧急度加权
    value_component = input.task_value * 0.6 + input.task_urgency * 0.2 + input.expected_impact * 0.2

    # 成本分量：归一化成本
    cost_component = input.normalized_cost

    # 风险分量
    risk_component = input.risk_penalty

    utility = value_component - cost_component - risk_component

    if utility > 0.3:
        recommendation = "维持全流程"
        is_worthwhile = True
    elif utility > 0:
        recommendation = "压缩质检/检索，降低预算"
        is_worthwhile = True
    elif utility > -0.3:
        recommendation = "快速通道（仅核心执行）"
        is_worthwhile = False
    else:
        recommendation = "拒绝执行"
        is_worthwhile = False

    return UtilityResult(
        utility=utility,
        is_worthwhile=is_worthwhile,
        recommendation=recommendation,
        value_component=value_component,
        cost_component=cost_component,
        risk_component=risk_component,
    )


# ─── 仲裁规则引擎 ────────────────────────────────────────────


class Arbiter:
    """统一仲裁器 — 所有跨Agent冲突的裁决中心

    优先级链：Safety > Correctness > QA > Budget > Performance
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.decisions: list[ArbiterDecision] = []
        self.rules: list[ArbitrationRule] = []
        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """加载默认仲裁规则"""
        self.rules = [
            # 安全违规 → 无条件停止，转人工
            ArbitrationRule(
                name="safety_breach_override",
                conflicts=[ConflictType.SAFETY_BREACH],
                priority=PriorityDomain.SAFETY,
                resolver=self._resolve_safety_breach,
                description="安全违规 → 熔断转人工",
            ),
            # 质检 vs 执行 → 质检覆盖执行（除非有特殊上下文）
            ArbitrationRule(
                name="qa_overrides_executor",
                conflicts=[ConflictType.QA_VS_EXECUTOR],
                priority=PriorityDomain.QA_SIGNAL,
                resolver=self._resolve_qa_vs_executor,
                description="质检发现的问题覆盖执行方的主张",
            ),
            # 预算 vs 质量 → Utility 判定
            ArbitrationRule(
                name="budget_vs_quality",
                conflicts=[ConflictType.BUDGET_VS_QUALITY, ConflictType.UTILITY_EVALUATION],
                priority=PriorityDomain.BUDGET,
                resolver=self._resolve_budget_vs_quality,
                description="用 Utility 函数判断预算 vs 质量的取舍",
            ),
            # Agent 输出不一致 → 选择置信度/权威性更高的
            ArbitrationRule(
                name="agent_conflict_resolution",
                conflicts=[ConflictType.AGENT_CONFLICT],
                priority=PriorityDomain.CORRECTNESS,
                resolver=self._resolve_agent_conflict,
                description="Agent输出冲突 → 选择置信度更高的",
            ),
            # 回滚决策 → 按失败等级
            ArbitrationRule(
                name="rollback_decision",
                conflicts=[ConflictType.ROLLBACK_DECISION],
                priority=PriorityDomain.SAFETY,
                resolver=self._resolve_rollback,
                description="回滚决策 → Level 1/2/3",
            ),
        ]

    def resolve(self, event: ConflictEvent) -> ArbiterDecision:
        """仲裁一个冲突事件

        Step 1: 匹配合适的规则
        Step 2: 应用规则决议
        Step 3: 记录决策
        """
        # 1. 找匹配规则
        conflict_type = event.type if isinstance(event.type, ConflictType) else ConflictType(event.type)
        matched_rules = [r for r in self.rules if conflict_type in r.conflicts]

        if not matched_rules:
            # 无匹配规则 → 转人工
            decision = ArbiterDecision(
                conflict_type=conflict_type,
                resolution="no_matching_rule",
                reason=f"无匹配规则处理 {conflict_type.value}，转人工",
                winning_side=None,
                priority_applied=PriorityDomain.SAFETY,
                confidence=0.0,
                action="escalate_to_human",
                trace_id=self._gen_trace_id(),
            )
        else:
            # 按优先级排序（最高优先级规则优先）
            matched_rules.sort(key=lambda r: r.priority.value)
            rule = matched_rules[0]
            decision = rule.resolver(event)

        # 附加 trace_id
        if not decision.trace_id:
            decision.trace_id = self._gen_trace_id()

        self.decisions.append(decision)
        logger.info(
            f"仲裁 [{decision.trace_id[:8]}]: "
            f"{conflict_type.value} → {decision.resolution} "
            f"({decision.reason})"
        )
        return decision

    def approve_transition(self, current_state, event, graph=None) -> tuple[bool, str]:
        """审批状态转换（状态机调用）

        Returns:
            (approved: bool, reason: str)
        """
        # 安全检查：是否可能破坏数据一致性
        if current_state == "executing" and event == "qa_fail":
            return (True, "质检失败回执行，自动批准")

        if current_state == "planned" and event == "low_utility":
            # 用 Utility 判断
            if graph and hasattr(graph, 'estimate_total_cost'):
                util = compute_utility(UtilityInput(
                    task_value=0.5,
                    cost_tokens=graph.estimate_total_cost(),
                    risk_penalty=0.1,
                ))
                if not util.is_worthwhile:
                    return (True, f"Utility={util.utility:.2f} < 0，拒绝执行")
            return (True, "允许转换")

        # 默认批准
        return (True, "默认批准")

    def evaluate_utility(self, task_value: float, cost_tokens: int,
                         risk_penalty: float = 0.0,
                         urgency: float = 0.5, impact: float = 0.5) -> UtilityResult:
        """Utility 评估"""
        return compute_utility(UtilityInput(
            task_value=task_value,
            cost_tokens=cost_tokens,
            risk_penalty=risk_penalty,
            task_urgency=urgency,
            expected_impact=impact,
        ))

    def compare_utility(self, options: list[dict]) -> list[tuple[int, UtilityResult]]:
        """对比多个方案的 Utility，返回排序结果

        Args:
            options: [{value, cost, risk, urgency, impact}, ...]

        Returns:
            [(index, UtilityResult), ...] 按 utility 降序
        """
        results = []
        for i, opt in enumerate(options):
            result = compute_utility(UtilityInput(
                task_value=opt.get("value", 0.5),
                cost_tokens=opt.get("cost_tokens", 0),
                risk_penalty=opt.get("risk", 0),
                task_urgency=opt.get("urgency", 0.5),
                expected_impact=opt.get("impact", 0.5),
            ))
            results.append((i, result))
        results.sort(key=lambda x: x[1].utility, reverse=True)
        return results

    # ─── 内置裁决器 ─────────────────────────────────────────

    def _resolve_safety_breach(self, event: ConflictEvent) -> ArbiterDecision:
        return ArbiterDecision(
            conflict_type=event.type,
            resolution="fuse_and_escalate",
            reason=f"安全违规: {event.claim_a} vs {event.claim_b} → 无条件熔断转人工",
            winning_side=None,
            priority_applied=PriorityDomain.SAFETY,
            confidence=1.0,
            action="escalate_to_human",
        )

    def _resolve_qa_vs_executor(self, event: ConflictEvent) -> ArbiterDecision:
        """质检 vs 执行 — 默认质检覆盖执行

        特殊例外：
          - 执行方提供充分证据（如引用规范文档）
          - 质检方误判（如工具链差异）
        """
        # 检查执行方是否有强证据
        executor_evidence = event.context.get("executor_evidence", "")
        qa_evidence = event.context.get("qa_evidence", "")

        # 如果有可验证的证据（引用文档、代码输出），允许执行方申诉
        if executor_evidence and not qa_evidence:
            return ArbiterDecision(
                conflict_type=event.type,
                resolution="executor_overrides_qa",
                reason=f"执行方提供证据: {executor_evidence[:100]}，覆盖质检判断",
                winning_side=event.source_b,
                priority_applied=PriorityDomain.QA_SIGNAL,
                confidence=0.7,
                action="proceed_with_executor_result",
            )

        # 默认：质检覆盖执行
        return ArbiterDecision(
            conflict_type=event.type,
            resolution="qa_overrides_executor",
            reason=f"质检发现问题: {event.claim_a[:100]}，执行方需修正",
            winning_side=event.source_a,
            priority_applied=PriorityDomain.QA_SIGNAL,
            confidence=0.85,
            action="rollback_level1",
        )

    def _resolve_budget_vs_quality(self, event: ConflictEvent) -> ArbiterDecision:
        """预算 vs 质量 — 用 Utility 判断"""
        util_input = UtilityInput(
            task_value=event.context.get("task_value", 0.5),
            cost_tokens=event.context.get("cost_tokens", 0),
            risk_penalty=event.context.get("risk_penalty", 0),
        )
        result = compute_utility(util_input)

        return ArbiterDecision(
            conflict_type=event.type,
            resolution="utility_based",
            reason=f"Utility={result.utility:.2f} → {result.recommendation}",
            winning_side="budget" if not result.is_worthwhile else "quality",
            priority_applied=PriorityDomain.BUDGET,
            confidence=0.8,
            action=result.recommendation,
        )

    def _resolve_agent_conflict(self, event: ConflictEvent) -> ArbiterDecision:
        """Agent 输出不一致 — 选置信度更高者"""
        conf_a = event.context.get("confidence_a", 0.5)
        conf_b = event.context.get("confidence_b", 0.5)

        if conf_a > conf_b:
            return ArbiterDecision(
                conflict_type=event.type,
                resolution=f"select_source_a",
                reason=f"Agent {event.source_a} 置信度 {conf_a:.2f} > {event.source_b} {conf_b:.2f}",
                winning_side=event.source_a,
                priority_applied=PriorityDomain.CORRECTNESS,
                confidence=abs(conf_a - conf_b),
                action="use_source_a_result",
            )
        elif conf_b > conf_a:
            return ArbiterDecision(
                conflict_type=event.type,
                resolution=f"select_source_b",
                reason=f"Agent {event.source_b} 置信度 {conf_b:.2f} > {event.source_a} {conf_a:.2f}",
                winning_side=event.source_b,
                priority_applied=PriorityDomain.CORRECTNESS,
                confidence=abs(conf_b - conf_a),
                action="use_source_b_result",
            )
        else:
            return ArbiterDecision(
                conflict_type=event.type,
                resolution="ambiguous_escalate",
                reason=f"置信度相同 ({conf_a:.2f})，转人工裁决",
                winning_side=None,
                priority_applied=PriorityDomain.CORRECTNESS,
                confidence=0.0,
                action="escalate_to_human",
            )

    def _resolve_rollback(self, event: ConflictEvent) -> ArbiterDecision:
        """回滚决策"""
        error_level = event.context.get("error_level", 1)

        if error_level == 1:
            return ArbiterDecision(
                conflict_type=event.type,
                resolution="rollback_level1",
                reason="Level 1: 回重执行",
                winning_side="recovery",
                priority_applied=PriorityDomain.SAFETY,
                confidence=0.95,
                action="rollback_to_executing",
            )
        elif error_level == 2:
            return ArbiterDecision(
                conflict_type=event.type,
                resolution="rollback_level2",
                reason="Level 2: 回重规划",
                winning_side="recovery",
                priority_applied=PriorityDomain.SAFETY,
                confidence=0.9,
                action="rollback_to_planning",
            )
        else:
            return ArbiterDecision(
                conflict_type=event.type,
                resolution="rollback_level3",
                reason="Level 3: 熔断转人工",
                winning_side="human",
                priority_applied=PriorityDomain.SAFETY,
                confidence=1.0,
                action="escalate_to_human",
            )

    def _gen_trace_id(self) -> str:
        return f"arb-{int(time.time()*1000000)}"

    def get_history(self, limit: int = 10) -> list[ArbiterDecision]:
        return self.decisions[-limit:]

    def clear_history(self) -> None:
        self.decisions.clear()


# ─── 仲裁规则 ─────────────────────────────────────────────────


@dataclass
class ArbitrationRule:
    """一条仲裁规则"""
    name: str
    conflicts: list[ConflictType]
    priority: PriorityDomain
    resolver: callable
    description: str = ""


# ─── 测试 ──────────────────────────────────────────────────────


def test_safety_breach():
    arbiter = Arbiter()
    event = ConflictEvent(
        type=ConflictType.SAFETY_BREACH,
        source_a="critic",
        source_b="executor",
        claim_a="代码不合规",
        claim_b="按规范实现",
        context={"rule_violated": "data_encryption"},
    )
    decision = arbiter.resolve(event)
    assert decision.resolution == "fuse_and_escalate"
    assert decision.winning_side is None
    assert decision.action == "escalate_to_human"
    print(f"安全违规仲裁: ✓ → {decision.reason}")


def test_qa_vs_executor():
    arbiter = Arbiter()
    # 情况1: 质检发现问题
    event = ConflictEvent(
        type=ConflictType.QA_VS_EXECUTOR,
        source_a="critic",
        source_b="executor",
        claim_a="输出格式不符合规范，缺少表头",
        claim_b="表头已在之前步骤生成",
    )
    decision = arbiter.resolve(event)
    assert decision.resolution == "qa_overrides_executor"
    print(f"质检 vs 执行 (默认质检赢): ✓ → {decision.reason}")

    # 情况2: 执行方有强证据
    event2 = ConflictEvent(
        type=ConflictType.QA_VS_EXECUTOR,
        source_a="critic",
        source_b="executor",
        claim_a="代码有语法错误",
        claim_b="代码已通过编译",
        context={"executor_evidence": "编译器返回 exit_code=0"},
    )
    decision2 = arbiter.resolve(event2)
    assert decision2.resolution == "executor_overrides_qa"
    print(f"质检 vs 执行 (执行方有证据): ✓ → {decision2.reason}")


def test_budget_vs_quality():
    arbiter = Arbiter()
    # 高价值任务
    event = ConflictEvent(
        type=ConflictType.BUDGET_VS_QUALITY,
        source_a="budget_controller",
        source_b="quality_gate",
        claim_a="预算超限，建议跳过质检",
        claim_b="必须质检保证质量",
        context={"task_value": 0.9, "cost_tokens": 3000, "risk_penalty": 0.1},
    )
    decision = arbiter.resolve(event)
    print(f"预算 vs 质量 (高价值): ✓ → {decision.reason}")

    # 低价值任务
    event2 = ConflictEvent(
        type=ConflictType.BUDGET_VS_QUALITY,
        source_a="budget_controller",
        source_b="quality_gate",
        claim_a="预算超限，跳过质检",
        claim_b="必须质检",
        context={"task_value": 0.1, "cost_tokens": 8000, "risk_penalty": 0.3},
    )
    decision2 = arbiter.resolve(event2)
    print(f"预算 vs 质量 (低价值): ✓ → {decision2.reason}")


def test_agent_conflict():
    arbiter = Arbiter()
    event = ConflictEvent(
        type=ConflictType.AGENT_CONFLICT,
        source_a="cc_writer",
        source_b="hermes_reviewer",
        claim_a="使用 FastAPI",
        claim_b="使用 Flask",
        context={"confidence_a": 0.85, "confidence_b": 0.6},
    )
    decision = arbiter.resolve(event)
    assert decision.winning_side == "cc_writer"
    print(f"Agent 冲突 (选高置信度): ✓ → {decision.reason}")


def test_rollback_decision():
    arbiter = Arbiter()
    for level, name in [(1, "Level 1"), (2, "Level 2"), (3, "Level 3")]:
        event = ConflictEvent(
            type=ConflictType.ROLLBACK_DECISION,
            source_a="executor",
            source_b="state_machine",
            claim_a="可以恢复",
            claim_b=f"错误等级 {level}",
            context={"error_level": level},
        )
        decision = arbiter.resolve(event)
        print(f"回滚决策 {name}: ✓ → {decision.reason}")


def test_utility_function():
    # 高价值低成本的值得做
    util = compute_utility(UtilityInput(task_value=0.9, cost_tokens=1000, risk_penalty=0.0))
    assert util.is_worthwhile
    print(f"Utility 高价值: U={util.utility:.2f} → {util.recommendation}")

    # 低价值高成本的不值得
    util2 = compute_utility(UtilityInput(task_value=0.1, cost_tokens=20000, risk_penalty=0.5))
    assert not util2.is_worthwhile
    print(f"Utility 低价值: U={util2.utility:.2f} → {util2.recommendation}")

    # 对比多个方案
    options = [
        {"value": 0.8, "cost_tokens": 5000, "risk": 0.1},
        {"value": 0.5, "cost_tokens": 1000, "risk": 0.0},
        {"value": 0.9, "cost_tokens": 15000, "risk": 0.3},
    ]
    arbiter = Arbiter()
    ranked = arbiter.compare_utility(options)
    print(f"方案排序: {[(i, f'U={r.utility:.2f}') for i, r in ranked]}")

    print("Utility 函数: ✓")


def test_approve_transition():
    arbiter = Arbiter()
    approved, reason = arbiter.approve_transition("executing", "qa_fail")
    assert approved
    print(f"审批转换 (qa_fail): ✓ → {reason}")

    # 模拟低 utility 的拒绝
    class FakeGraph:
        @staticmethod
        def estimate_total_cost():
            return 50000

    approved, reason = arbiter.approve_transition("planned", "low_utility", FakeGraph())
    print(f"审批转换 (low_utility): → {reason}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("═══ 仲裁器测试 ═══\n")
    test_safety_breach()
    print()
    test_qa_vs_executor()
    print()
    test_budget_vs_quality()
    print()
    test_agent_conflict()
    print()
    test_rollback_decision()
    print()
    test_utility_function()
    print()
    test_approve_transition()
    print("\n═══ 全部测试通过 ═══")
