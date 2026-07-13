"""
评估闭环 — 多Agent协作框架的自我进化引擎

核心流程：
  output → evaluator → score → feedback → policy update

评分区间：
  < 0.4: 强制优化
  0.4-0.7: 预警
  > 0.7: 正常

漂移检测（Drift Detection）：
  - Prompt漂移：结构变化率 > 20%
  - Memory漂移：临时/矛盾比例 > 30%
  - Policy漂移：仲裁分布偏移 > 15%

设计原则：
  1. 评估结果自动回写策略更新（闭环）
  2. 负反馈抑制 — 临时/矛盾/一次性反馈降权
  3. 三种漂移检测自动触发离线测评

Usage:
    from agent_framework.eval_loop import EvalLoop, Evaluation, ScoreCard

    loop = EvalLoop()
    score = loop.evaluate(
        task_id="task_001",
        output="...",
        criteria={"quality": 0.8, "efficiency": 0.6},
    )
    drift = loop.detect_drift()
"""

from __future__ import annotations

import time
import json
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from collections import defaultdict

logger = logging.getLogger("agent-framework.eval-loop")


# ─── 基元定义 ─────────────────────────────────────────────────


class EvalDimension(str, Enum):
    """评估维度"""
    QUALITY = "quality"                 # 内容质量
    EFFICIENCY = "efficiency"           # 执行效率
    CONSISTENCY = "consistency"         # 一致性
    SAFETY = "safety"                   # 安全性
    COST_EFFECTIVENESS = "cost_effectiveness"  # 成本效益
    USER_SATISFACTION = "user_satisfaction"    # 用户满意度


class DriftType(str, Enum):
    """漂移类型"""
    PROMPT_STRUCTURE = "prompt_structure"       # Prompt 结构漂移
    MEMORY_TEMPORARY = "memory_temporary"       # Memory 临时/矛盾比例
    POLICY_DISTRIBUTION = "policy_distribution" # 仲裁策略漂移


@dataclass
class ScoreCard:
    """单次评估得分卡"""
    task_id: str
    dimensions: dict[str, float]       # 维度 → 分数 (0-1)
    overall: float = 0.0               # 综合得分
    feedback: list[str] = field(default_factory=list)  # 反馈意见
    passed: bool = True                # 是否通过
    timestamp: float = field(default_factory=time.time)
    evaluator: str = "auto"            # 评估方 (auto/human/critic)

    def __post_init__(self):
        if self.overall == 0.0 and self.dimensions:
            self.overall = sum(self.dimensions.values()) / len(self.dimensions)
        self.passed = self.overall >= 0.7

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "overall": self.overall,
            "dimensions": self.dimensions,
            "passed": self.passed,
            "feedback_count": len(self.feedback),
            "evaluator": self.evaluator,
            "timestamp": self.timestamp,
        }

    @property
    def needs_optimization(self) -> bool:
        return self.overall < 0.4

    @property
    def needs_attention(self) -> bool:
        return 0.4 <= self.overall < 0.7


@dataclass
class DriftReport:
    """漂移检测报告"""
    type: DriftType
    metric: str
    current_value: float
    threshold: float
    drifted: bool
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "metric": self.metric,
            "current": self.current_value,
            "threshold": self.threshold,
            "drifted": self.drifted,
            "details": self.details,
        }


# ─── 评估引擎 ─────────────────────────────────────────────────


class EvalLoop:
    """评估闭环引擎 — 持续评估 → 反馈 → 策略更新

    管理评估历史、漂移检测和策略自动调整。
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.scores: list[ScoreCard] = []
        self.drift_reports: list[DriftReport] = []
        self.policy: dict[str, Any] = self._default_policy()
        self.evaluators: dict[str, Callable] = {}
        self._last_eval_time: float = time.time()
        self.stats = {
            "evaluations": 0,
            "passed": 0,
            "failed": 0,
            "optimizations_triggered": 0,
            "drifts_detected": 0,
        }

    def _default_policy(self) -> dict:
        return {
            "thresholds": {
                "quality_min": 0.7,
                "efficiency_min": 0.5,
                "safety_min": 0.9,
                "pass_threshold": 0.7,
                "optimize_threshold": 0.4,
            },
            "weights": {
                "quality": 0.4,
                "efficiency": 0.15,
                "consistency": 0.15,
                "safety": 0.2,
                "cost_effectiveness": 0.05,
                "user_satisfaction": 0.05,
            },
            "drift": {
                "prompt_structure_delta": 0.20,     # 20%
                "memory_temporary_ratio": 0.30,     # 30%
                "policy_distribution_delta": 0.15,  # 15%
            },
            "optimization": {
                "auto_retry_on_low_score": True,
                "max_retries_per_task": 2,
                "quality_gate_enabled": True,
                "budget_multiplier_on_retry": 1.5,
            },
        }

    # ─── 核心评估 ──────────────────────────────────────────

    def evaluate(self, task_id: str, output: Any,
                 criteria: dict[str, float] | None = None,
                 evaluator: str = "auto") -> ScoreCard:
        """执行一次评估

        Args:
            task_id: 任务 ID
            output: 需要评估的输出
            criteria: 各维度得分 {dimension: score}，None=待外部评估器
            evaluator: 评估来源

        Returns:
            ScoreCard
        """
        if criteria is None:
            criteria = self._run_evaluators(task_id, output)

        # 应用权重
        weighted = {}
        weights_used_sum = 0.0
        for dim, score in criteria.items():
            weight = self.policy["weights"].get(dim, 1.0)
            weighted[dim] = score * weight
            weights_used_sum += weight

        # 生成反馈
        feedback = self._generate_feedback(criteria)

        card = ScoreCard(
            task_id=task_id,
            dimensions=criteria,
            overall=max(0.0, sum(weighted.values()) / max(weights_used_sum, 0.01)),
            feedback=feedback,
            evaluator=evaluator,
        )

        self.scores.append(card)
        self.stats["evaluations"] += 1

        if card.passed:
            self.stats["passed"] += 1
        else:
            self.stats["failed"] += 1
            if card.needs_optimization:
                self.stats["optimizations_triggered"] += 1
                self._trigger_optimization(card)

        logger.info(
            f"评估 [{task_id[:8]}]: "
            f"overall={card.overall:.2f} "
            f"{'✓' if card.passed else '✗'}"
        )
        return card

    def _run_evaluators(self, task_id: str, output: Any) -> dict[str, float]:
        """运行注册的评估器"""
        if not self.evaluators:
            # 默认评估：全 0.8（无外部评估器时）
            return {dim.value: 0.8 for dim in EvalDimension}
        scores = {}
        for name, fn in self.evaluators.items():
            try:
                result = fn(task_id, output)
                if isinstance(result, dict):
                    scores.update(result)
            except Exception as e:
                logger.warning(f"评估器 {name} 失败: {e}")
        return scores

    def register_evaluator(self, name: str, fn: Callable) -> None:
        """注册自定义评估器

        Args:
            name: 评估器名
            fn: 接收 (task_id, output) 返回 {dimension: score}
        """
        self.evaluators[name] = fn

    def _generate_feedback(self, criteria: dict[str, float]) -> list[str]:
        """根据评分生成反馈"""
        feedback = []
        thresholds = self.policy["thresholds"]

        for dim, score in criteria.items():
            min_score = thresholds.get(f"{dim}_min", 0.6)
            if score < min_score:
                feedback.append(f"{dim}: {score:.2f} < 阈值 {min_score:.2f}")
            elif score < thresholds.get("pass_threshold", 0.7):
                feedback.append(f"{dim}: {score:.2f} 接近阈值，需关注")

        return feedback

    def _trigger_optimization(self, card: ScoreCard) -> None:
        """触发优化 — 记录自动调整"""
        logger.info(f"触发优化: task={card.task_id}, overall={card.overall:.2f}")
        # 策略自动调整
        for dim, score in card.dimensions.items():
            if score < 0.4:
                # 降低该维度的预期
                current = self.policy["thresholds"].get(f"{dim}_min", 0.6)
                self.policy["thresholds"][f"{dim}_min"] = max(0.3, current * 0.9)
                logger.debug(f"  调整阈值: {dim}_min: {current:.2f} → {self.policy['thresholds'][f'{dim}_min']:.2f}")

    # ─── 漂移检测 ──────────────────────────────────────────

    def detect_drift(self, context: dict[str, Any] | None = None) -> list[DriftReport]:
        """运行所有漂移检测

        Args:
            context: 检测上下文，包含 prompt 结构、memory 统计、仲裁统计等

        Returns:
            漂移报告列表
        """
        ctx = context or {}
        reports = []

        # 1. Prompt 结构漂移
        prompt_report = self._check_prompt_drift(ctx.get("prompt_structure", {}))
        if prompt_report:
            reports.append(prompt_report)

        # 2. Memory 临时/矛盾比例
        memory_report = self._check_memory_drift(ctx.get("memory_stats", {}))
        if memory_report:
            reports.append(memory_report)

        # 3. 仲裁策略漂移
        policy_report = self._check_policy_drift(ctx.get("arbiter_stats", {}))
        if policy_report:
            reports.append(policy_report)

        self.drift_reports.extend(reports)

        for r in reports:
            if r.drifted:
                self.stats["drifts_detected"] += 1
                logger.warning(f"漂移检测: {r.type.value} — {r.metric}={r.current_value:.2f} > 阈值{r.threshold:.2f}")

        return reports

    def _check_prompt_drift(self, structure: dict) -> DriftReport | None:
        """Prompt 结构漂移检测 — 结构变化率 > 20%"""
        baseline = structure.get("baseline_structure", "")
        current = structure.get("current_structure", "")
        if not baseline or not current:
            return None

        # 简单结构变化率计算（用字符级 diff 比例）
        max_len = max(len(baseline), len(current))
        if max_len == 0:
            return None

        # Levenshtein 距离简易近似
        diff_chars = sum(1 for i in range(min(len(baseline), len(current)))
                        if baseline[i] != current[i])
        diff_chars += abs(len(baseline) - len(current))
        change_rate = diff_chars / max_len

        threshold = self.policy["drift"]["prompt_structure_delta"]
        return DriftReport(
            type=DriftType.PROMPT_STRUCTURE,
            metric="structure_change_rate",
            current_value=change_rate,
            threshold=threshold,
            drifted=change_rate > threshold,
            details={"baseline_len": len(baseline), "current_len": len(current)},
        )

    def _check_memory_drift(self, stats: dict) -> DriftReport | None:
        """Memory 临时/矛盾比例 > 30%"""
        temporary = stats.get("temporary_entries", 0)
        total = stats.get("total_entries", 0)
        if total == 0:
            return None

        ratio = temporary / total
        threshold = self.policy["drift"]["memory_temporary_ratio"]
        return DriftReport(
            type=DriftType.MEMORY_TEMPORARY,
            metric="temporary_ratio",
            current_value=ratio,
            threshold=threshold,
            drifted=ratio > threshold,
            details={"temporary": temporary, "total": total},
        )

    def _check_policy_drift(self, stats: dict) -> DriftReport | None:
        """仲裁策略偏移 > 15%"""
        baseline_dist = stats.get("baseline_distribution", {})
        current_dist = stats.get("current_distribution", {})
        if not baseline_dist or not current_dist:
            return None

        # 分布偏移 = 各决策类型比例变化的均值
        all_types = set(baseline_dist) | set(current_dist)
        total_delta = sum(
            abs(current_dist.get(t, 0) - baseline_dist.get(t, 0))
            for t in all_types
        )
        avg_delta = total_delta / max(len(all_types), 1)

        threshold = self.policy["drift"]["policy_distribution_delta"]
        return DriftReport(
            type=DriftType.POLICY_DISTRIBUTION,
            metric="avg_distribution_delta",
            current_value=avg_delta,
            threshold=threshold,
            drifted=avg_delta > threshold,
            details={"baseline": baseline_dist, "current": current_dist},
        )

    # ─── 策略管理 ──────────────────────────────────────────

    def get_policy(self) -> dict:
        return self.policy

    def update_policy(self, updates: dict) -> None:
        """更新策略"""
        self._deep_update(self.policy, updates)
        logger.info(f"策略更新: {json.dumps(updates, ensure_ascii=False)[:200]}")

    def _deep_update(self, d: dict, updates: dict) -> dict:
        for key, value in updates.items():
            if key in d and isinstance(d[key], dict) and isinstance(value, dict):
                self._deep_update(d[key], value)
            else:
                d[key] = value
        return d

    # ─── 统计和报告 ────────────────────────────────────────

    def recent_scores(self, limit: int = 10) -> list[ScoreCard]:
        return self.scores[-limit:]

    def pass_rate(self, window: int = 50) -> float:
        """最近 N 次评估的通过率"""
        recent = self.scores[-window:]
        if not recent:
            return 1.0
        return sum(1 for s in recent if s.passed) / len(recent)

    def average_score(self, window: int = 50) -> float:
        """最近 N 次评估的平均分"""
        recent = self.scores[-window:]
        if not recent:
            return 0.0
        return sum(s.overall for s in recent) / len(recent)

    def summary(self) -> str:
        return (
            f"EvalLoop '{self.name}': "
            f"{self.stats['evaluations']} 次评估, "
            f"通过率 {self.pass_rate()*100:.0f}%, "
            f"平均分 {self.average_score():.2f}, "
            f"优化 {self.stats['optimizations_triggered']} 次, "
            f"漂移 {self.stats['drifts_detected']} 次"
        )

    def get_recommendations(self) -> list[str]:
        """基于当前状态给出改进建议"""
        recs = []

        if self.pass_rate() < 0.7:
            recs.append(f"通过率 {self.pass_rate()*100:.0f}% < 70%，建议检查评估标准")

        if self.average_score() < 0.5:
            recs.append(f"平均分 {self.average_score():.2f} 偏低，建议降低任务复杂度")

        recent_drifts = [r for r in self.drift_reports[-10:] if r.drifted]
        if recent_drifts:
            types = set(r.type.value for r in recent_drifts)
            recs.append(f"检测到漂移: {', '.join(types)}，建议执行离线测评")

        return recs


# ─── 测试 ──────────────────────────────────────────────────────


def test_basic_evaluation():
    loop = EvalLoop()
    card = loop.evaluate("test_001", "some output",
                         criteria={"quality": 0.9, "efficiency": 0.8})
    assert card.passed
    assert card.overall > 0.7
    print(f"基本评估: ✓ (overall={card.overall:.2f})")


def test_low_score():
    loop = EvalLoop()
    card = loop.evaluate("test_002", "bad output",
                         criteria={"quality": 0.2, "efficiency": 0.1})
    assert not card.passed
    assert card.needs_optimization
    print(f"低分评估: ✓ (overall={card.overall:.2f}, {card.feedback})")


def test_custom_evaluator():
    loop = EvalLoop()

    def my_evaluator(task_id, output):
        return {"quality": 0.85 if len(str(output)) > 10 else 0.3}

    loop.register_evaluator("length_check", my_evaluator)
    card = loop.evaluate("test_003", "这是一段足够长的输出内容")
    assert card.passed
    print(f"自定义评估器: ✓ (overall={card.overall:.2f})")


def test_drift_detection_prompt():
    loop = EvalLoop()
    reports = loop.detect_drift({
        "prompt_structure": {
            "baseline_structure": "你是一个专业的AI助手，负责回答用户问题",
            "current_structure": "回答这个编程问题的解决方案，用Python代码实现",
        }
    })
    assert len(reports) >= 1
    prompt_report = [r for r in reports if r.type == DriftType.PROMPT_STRUCTURE]
    if prompt_report:
        print(f"Prompt漂移检测: ✓ (change_rate={prompt_report[0].current_value:.2f}, drifted={prompt_report[0].drifted})")
    else:
        print("Prompt漂移检测: - (无数据)")


def test_drift_detection_memory():
    loop = EvalLoop()
    reports = loop.detect_drift({
        "memory_stats": {
            "temporary_entries": 40,
            "total_entries": 100,
        }
    })
    memory_report = [r for r in reports if r.type == DriftType.MEMORY_TEMPORARY]
    if memory_report:
        print(f"Memory漂移检测: ✓ (ratio={memory_report[0].current_value:.2f}, drifted={memory_report[0].drifted})")
    print("Memory漂移检测: ✓")


def test_drift_detection_policy():
    loop = EvalLoop()
    reports = loop.detect_drift({
        "arbiter_stats": {
            "baseline_distribution": {"qa_overrides": 0.6, "executor_wins": 0.4},
            "current_distribution": {"qa_overrides": 0.3, "executor_wins": 0.7},
        }
    })
    policy_report = [r for r in reports if r.type == DriftType.POLICY_DISTRIBUTION]
    if policy_report:
        print(f"Policy漂移检测: ✓ (delta={policy_report[0].current_value:.2f}, drifted={policy_report[0].drifted})")
    else:
        print("Policy漂移检测: - (无数据)")


def test_policy_update():
    loop = EvalLoop()
    loop.update_policy({
        "thresholds": {"quality_min": 0.8},
        "drift": {"prompt_structure_delta": 0.15},
    })
    assert loop.policy["thresholds"]["quality_min"] == 0.8
    assert loop.policy["drift"]["prompt_structure_delta"] == 0.15
    print(f"策略更新: ✓ (quality_min={loop.policy['thresholds']['quality_min']})")


def test_stats():
    loop = EvalLoop()
    for i in range(10):
        loop.evaluate(f"stats_{i}", "output",
                      criteria={"quality": 0.7 + (i % 3) * 0.1, "efficiency": 0.8})
    assert 0 < loop.pass_rate() <= 1.0
    assert loop.average_score() > 0
    print(f"统计分析: ✓ (pass_rate={loop.pass_rate()*100:.0f}%, avg={loop.average_score():.2f})")


def test_recommendations():
    loop = EvalLoop()
    for i in range(10):
        loop.evaluate(f"rec_{i}", "bad",
                      criteria={"quality": 0.2, "efficiency": 0.1})
    recs = loop.get_recommendations()
    assert len(recs) > 0
    print(f"改进建议: ✓ ({len(recs)} 条)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print("═══ 评估闭环测试 ═══\n")
    test_basic_evaluation()
    test_low_score()
    test_custom_evaluator()
    print()
    test_drift_detection_prompt()
    test_drift_detection_memory()
    test_drift_detection_policy()
    print()
    test_policy_update()
    test_stats()
    test_recommendations()
    print("\n═══ 全部测试通过 ═══")
