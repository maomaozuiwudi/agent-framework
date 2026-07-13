"""
执行器集成 — 多Agent协作框架的执行层

三通道执行：CC / Codex / Hermes
质检关：Critic 交叉审查
聚合器：Aggregator 对齐并行结果

完整管线：DagStateMachine 驱动的端到端执行流程

Usage:
    from executor import FullPipeline, Executor

    pipeline = FullPipeline()
    result = pipeline.run(task_id="test", graph=dag)
"""

from __future__ import annotations

import time
import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("agent-framework.executor")


# ─── 执行器 ─────────────────────────────────────────────────


@dataclass
class ExecutionResult:
    """单节点执行结果"""
    node_id: str
    success: bool
    output: Any = None
    error: str | None = None
    duration: float = 0.0
    agent_used: str = ""
    token_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "success": self.success,
            "error": self.error,
            "duration": f"{self.duration:.2f}s",
            "agent": self.agent_used,
            "tokens": self.token_used,
        }


class Executor:
    """执行器 — 管理 Agent 三通道的执行

    三个执行通道：
      CC:     通过 delegate_task 派给 CC 子代理（代码/架构/长上下文）
      Codex:  通过 delegate_task 派给 Codex（视频/图片）
      Hermes: 直接在上下文中执行（文本/简单任务）

    也支持自定义 runner 回调。
    """

    def __init__(self):
        self.runners: dict[str, Callable] = {}
        self.results: dict[str, ExecutionResult] = {}
        self.total_tokens: int = 0
        self.total_time: float = 0.0

    def register_runner(self, agent_type: str, runner: Callable) -> None:
        """注册 Agent 执行器

        Args:
            agent_type: "cc", "codex", "hermes", 或其他自定义类型
            runner: 接收 (node) 返回 ExecutionResult
        """
        self.runners[agent_type] = runner

    def execute_node(self, node) -> ExecutionResult:
        """执行单个 DAG 节点"""
        agent = node.agent.value if hasattr(node.agent, 'value') else node.agent
        runner = self.runners.get(agent)

        if runner:
            start = time.time()
            try:
                raw_result = runner(node)
                if isinstance(raw_result, ExecutionResult):
                    result = raw_result
                else:
                    result = ExecutionResult(
                        node_id=node.id,
                        success=True,
                        output=raw_result,
                        agent_used=agent,
                        duration=time.time() - start,
                    )
                result.node_id = node.id
                result.agent_used = agent
                result.duration = time.time() - start
                node.result = result.output
                return result
            except Exception as e:
                node.error = str(e)
                return ExecutionResult(
                    node_id=node.id,
                    success=False,
                    error=str(e),
                    agent_used=agent,
                    duration=time.time() - start,
                )

        # 无注册 runner → 默认模拟执行（用于测试）
        return ExecutionResult(
            node_id=node.id,
            success=True,
            output=f"[模拟执行] {node.goal}",
            agent_used=agent,
            duration=0.001,
        )

    def execute_graph(self, graph) -> dict[str, ExecutionResult]:
        """执行整个 DAG

        按拓扑批次执行，同批次节点可并行（当前串行执行，
        可替换为线程池/异步）
        """
        self.results = {}
        batches = graph.topological_sort()
        start = time.time()

        for batch_idx, batch in enumerate(batches):
            logger.info(f"批次 {batch_idx + 1}/{len(batches)}: {batch}")
            for nid in batch:
                node = graph.nodes[nid]
                result = self.execute_node(node)
                self.results[nid] = result
                self.total_tokens += result.token_used

                if not result.success:
                    logger.error(f"节点失败: {nid} — {result.error}")
                    # 标记所有下游节点为 skipped
                    for dep_id in graph.get_descendants(nid):
                        graph.skip_node(dep_id)

        self.total_time = time.time() - start
        return self.results

    def summary(self) -> str:
        success = sum(1 for r in self.results.values() if r.success)
        failed = sum(1 for r in self.results.values() if not r.success)
        return (
            f"Executor: {len(self.results)} 节点, "
            f"{success}✓ {failed}✗, "
            f"耗时 {self.total_time:.1f}s"
        )


# ─── 质检器 ──────────────────────────────────────────────────


@dataclass
class QAResult:
    """质检结果"""
    passed: bool
    score: float = 1.0
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "score": self.score,
            "issues": self.issues,
            "suggestions": self.suggestions,
        }


class Critic:
    """质检器 — 交叉审查、AI写作检测、质量评估

    架构中段5定义的质检能力：
      - 交叉审查（不同模型/Agent互审）
      - 语义级去AI化检测
      - 质量评分
      - 最终复杂度判定权
    """

    def __init__(self):
        self.checkers: list[Callable] = []
        self._setup_default_checkers()

    def _setup_default_checkers(self):
        """注册默认质检规则"""
        self.checkers = [
            self._check_empty_output,
            self._check_error_in_output,
            self._check_completeness,
        ]

    def register_checker(self, fn: Callable) -> None:
        """注册自定义质检函数

        Args:
            fn: 接收 (node_id, output, context) 返回 (passed, issues)
        """
        self.checkers.append(fn)

    def check(self, graph) -> QAResult:
        """对 DAG 执行结果进行全面质检"""
        all_issues = []
        all_suggestions = []
        scores = []

        for nid, node in graph.nodes.items():
            if node.status.value == "skipped":
                continue
            result = node.result
            context = {"graph": graph, "node": node}

            for checker in self.checkers:
                try:
                    checker_result = checker(nid, result, context)
                    if isinstance(checker_result, tuple):
                        passed, issues_or_msg = checker_result
                        if not passed:
                            all_issues.append(f"[{nid}] {issues_or_msg}")
                            scores.append(0.0)
                    elif isinstance(checker_result, dict):
                        if not checker_result.get("passed", True):
                            all_issues.append(f"[{nid}] {checker_result.get('issue', '')}")
                        scores.append(checker_result.get("score", 1.0))
                except Exception as e:
                    all_issues.append(f"[{nid}] 质检异常: {e}")

        overall_score = sum(scores) / max(len(scores), 1) if scores else 1.0
        return QAResult(
            passed=len(all_issues) == 0,
            score=overall_score,
            issues=all_issues,
            suggestions=[f"修复: {i}" for i in all_issues],
        )

    def _check_empty_output(self, nid: str, output: Any, ctx: dict) -> tuple:
        return (output is not None and output != "", "输出为空")

    def _check_error_in_output(self, nid: str, output: Any, ctx: dict) -> tuple:
        if isinstance(output, str):
            # 只标记纯错误输出，不标记包含"error"关键词的正常分析报告
            lowered = output.lower().strip()
            # 输出本身就是错误信息（很短的错误消息）
            if len(output) < 100 and any(lowered.startswith(ind) for ind in ["error", "exception", "traceback", "failed"]):
                return (False, f"输出为错误信息")
            # 输出全是堆栈跟踪
            if "traceback" in lowered and lowered.count("\n") < 5:
                return (False, "输出含堆栈跟踪")
        return (True, "")

    def _check_completeness(self, nid: str, output: Any, ctx: dict) -> dict:
        score = 1.0
        if isinstance(output, str):
            # 空或极短输出扣分
            if len(output.strip()) < 10:
                score = 0.3
            elif len(output.strip()) < 50:
                score = 0.7
        return {"passed": score > 0.5, "score": score, "issue": "" if score > 0.5 else f"输出过短({len(str(output))}字符)"}


# ─── 聚合器 ──────────────────────────────────────────────────


class Aggregator:
    """聚合器 — 对齐并行执行结果后再送质检

    职责：
      1. 收集并行节点的输出
      2. 按依赖关系对齐结果
      3. 产出统一的聚合格式
    """

    @dataclass
    class AggregatedOutput:
        """聚合后的输出"""
        task_id: str
        outputs: dict[str, Any]
        combined: str | None = None
        node_count: int = 0
        success_count: int = 0
        failed_count: int = 0

    def aggregate(self, task_id: str, graph) -> AggregatedOutput:
        """聚合 DAG 执行结果

        按拓扑顺序合并输出，叶子节点优先。
        """
        outputs = {}
        success = 0
        failed = 0

        for nid, node in graph.nodes.items():
            if node.result is not None:
                outputs[nid] = node.result
                success += 1
            else:
                failed += 1

        # 按拓扑次序合并文本输出
        parts = []
        try:
            batches = graph.topological_sort()
            for batch in batches:
                for nid in batch:
                    if nid in outputs:
                        val = outputs[nid]
                        if isinstance(val, str):
                            parts.append(f"## {nid}\n{val}")
                        elif isinstance(val, dict):
                            parts.append(f"## {nid}\n{json.dumps(val, ensure_ascii=False, indent=2)}")
                        else:
                            parts.append(f"## {nid}\n{str(val)}")
        except Exception:
            pass

        combined = "\n\n".join(parts) if parts else None

        return self.AggregatedOutput(
            task_id=task_id,
            outputs=outputs,
            combined=combined,
            node_count=len(graph.nodes),
            success_count=success,
            failed_count=failed,
        )


# ─── 完整管线 ──────────────────────────────────────────────


class FullPipeline:
    """端到端管线 — 整合 DAG + 状态机 + 仲裁器 + 记忆 + 评估

    流程：
      init → plan → execute → aggregate → critic → memory_write → eval → finalize
    """

    def __init__(self, arbiter=None, memory=None, eval_loop=None):
        from state_machine import DagStateMachine
        from arbiter import Arbiter, UtilityInput, compute_utility
        from memory import MemoryStore
        from eval_loop import EvalLoop

        self.arbiter = arbiter or Arbiter()
        self.memory = memory or MemoryStore()
        self.eval_loop = eval_loop or EvalLoop()
        self.executor = Executor()
        self.critic = Critic()
        self.aggregator = Aggregator()
        self.state_machine: DagStateMachine | None = None
        self._UtilityInput = UtilityInput
        self._compute_utility = compute_utility

        self.results: dict[str, Any] = {}

    def run(self, task_id: str, graph, runner: Callable | None = None,
            auto_validate: bool = True) -> dict[str, Any]:
        """完整执行一条任务管线

        Args:
            task_id: 任务 ID
            graph: DAG 图
            runner: 自定义节点执行器（可选）
            auto_validate: 是否自动验证 DAG

        Returns:
            {node_id: output, ...}
        """
        from state_machine import DagStateMachine, Event, RollbackLevel

        # 1. 验证 DAG
        if auto_validate:
            errors = graph.validate()
            if errors:
                raise ValueError(f"DAG 验证失败: {errors}")

        # 2. 构建状态机
        sm = DagStateMachine(task_id, graph, self.arbiter)
        self.state_machine = sm
        sm.on_event(Event.TASK_RECEIVED, "任务接收")

        # 3. Utility 评估（先完成规划阶段，使状态机进入 PLANNED）
        sm.on_event(Event.PLAN_COMPLETE, "规划完成")
        total_cost = graph.estimate_total_cost()
        utility = self._compute_utility(self._UtilityInput(
            task_value=0.6,
            cost_tokens=total_cost,
            risk_penalty=0.1,
        ))
        logger.info(f"Utility: {utility.utility:.2f} — {utility.recommendation}")

        if not utility.is_worthwhile:
            logger.warning(f"Utility={utility.utility:.2f} — {utility.recommendation}，继续执行")
            # 不拦截，继续执行

        # 4. 开始执行
        sm.on_event(Event.EXECUTE_START, "开始执行")

        # 5. 执行 DAG
        if runner:
            self.executor.register_runner("hermes", runner)

        try:
            exec_results = self.executor.execute_graph(graph)
        except Exception as e:
            sm.state_machine.rollback(RollbackLevel.LEVEL_1, str(e))
            self.results = {"_error": str(e)}
            return self.results

        # 6. 聚合
        aggregated = self.aggregator.aggregate(task_id, graph)

        # 7. 质检
        sm.on_event(Event.EXECUTE_COMPLETE, f"执行完成: {aggregated.success_count}✓ {aggregated.failed_count}✗")
        qa_result = self.critic.check(graph)

        if qa_result.passed:
            sm.on_event(Event.QA_PASS, f"质检通过 (score={qa_result.score:.2f})")
        else:
            sm.on_event(Event.QA_FAIL, f"质检问题: {len(qa_result.issues)} 项")
            self.results = {
                "_qa_failed": True,
                "_qa_issues": qa_result.issues,
                "_partial": exec_results,
                "_aggregated": aggregated.combined,
            }
            return self.results

        # 8. 记忆写入 + 评估
        self._write_to_memory(task_id, aggregated, qa_result)
        eval_score = self.eval_loop.evaluate(
            task_id, aggregated.combined or "",
            criteria=self._build_eval_criteria(aggregated, qa_result, utility),
        )
        sm.on_event(Event.MEMORY_WRITE_DONE, f"任务完成 (eval={eval_score.overall:.2f})")

        # 9. 漂移检测 (DriftReport)
        drift_stats = {
            "arbiter_stats": {
                "baseline_distribution": getattr(self.arbiter, '_distribution', {}).get('baseline', {}),
                "current_distribution": getattr(self.arbiter, '_distribution', {}).get('current', {}),
            },
            "memory_stats": {
                "temporary_entries": len(getattr(self.memory, '_l2_store', {})),
                "total_entries": max(
                    len(getattr(self.memory, '_l1_store', {})),
                    len(getattr(self.memory, '_l2_store', {})),
                    1,
                ),
            },
        }
        drift_reports = self.eval_loop.detect_drift(drift_stats)
        drift_summary = [r.to_dict() for r in drift_reports]

        # 打印执行摘要
        _print_execution_summary(task_id, graph, aggregated, qa_result, eval_score, drift_reports)

        # 10. 结果
        self.results = {
            node_id: node.result
            for node_id, node in graph.nodes.items()
            if node.result is not None
        }
        self.results["_meta"] = {
            "task_id": task_id,
            "duration": sm.state_machine.duration,
            "nodes": len(graph.nodes),
            "success": aggregated.success_count,
            "failed": aggregated.failed_count,
            "qa_score": qa_result.score,
            "qa_issues": qa_result.issues,
            "eval_score": eval_score.overall,
            "eval_passed": eval_score.passed,
            "utility": utility.utility,
            "drift_reports": drift_summary,
            "drift_detected": any(r.drifted for r in drift_reports),
            "state_history": [h.to_dict() for h in sm.state_machine.history],
        }
        return self.results

    def _write_to_memory(self, task_id: str, aggregated, qa_result) -> None:
        """将执行结果写入记忆层"""
        # L1: 任务状态
        self.memory.write_l1(f"task:{task_id}:status", "completed",
                             source="executor", ttl=3600)
        # L2: 执行摘要
        self.memory.write_l2(f"task:{task_id}:summary",
                            f"{aggregated.success_count}/{aggregated.node_count} 节点完成")
        # L2: 质检结果
        self.memory.write_l2(f"task:{task_id}:qa",
                            {"passed": qa_result.passed, "issues": qa_result.issues},
                            source="critic")
        # L4: 成果物（异步合并）
        if aggregated.combined:
            self.memory.write_l4(f"output:{task_id}", aggregated.combined[:500],
                                source="executor")

    def _build_eval_criteria(self, aggregated, qa_result, utility) -> dict:
        return {
            "quality": qa_result.score,
            "efficiency": min(1.0, aggregated.success_count / max(aggregated.node_count, 1)),
            "cost_effectiveness": max(0, utility.utility),
        }

    def summary(self) -> str:
        meta = self.results.get("_meta", {})
        if not meta:
            return "管线未运行"
        return (
            f"任务 {meta.get('task_id', '?' )[:8]}: "
            f"{meta.get('success', 0)}✓ {meta.get('failed', 0)}✗ "
            f"| QA={meta.get('qa_score', 0):.2f} EVAL={meta.get('eval_score', 0):.2f} "
            f"UTIL={meta.get('utility', 0):.2f} "
            f"| {meta.get('duration', 0):.1f}s"
        )

    def to_dict(self) -> dict:
        return self.results


# ─── 执行摘要打印 ───────────────────────────────────────────


def _print_execution_summary(task_id: str, graph, aggregated, qa_result, eval_score, drift_reports):
    """打印端到端执行摘要"""
    drift_count = sum(1 for r in drift_reports if r.drifted)
    print()
    print("═" * 55)
    print(f"  📋 管线执行摘要 — {task_id[:20]}")
    print("═" * 55)
    print(f"  DAG:       {len(graph.nodes)} 节点, {len(graph.topological_sort())} 批次")
    print(f"  执行:      {aggregated.success_count}✓ {aggregated.failed_count}✗")
    print(f"  质检 (QA): score={qa_result.score:.2f} {'✓' if qa_result.passed else '✗'}")
    if not qa_result.passed:
        for issue in qa_result.issues[:3]:
            print(f"             ⚠ {issue}")
    print(f"  评估:      score={eval_score.overall:.2f} {'✓' if eval_score.passed else '✗'}")
    print(f"  漂移检测:  {len(drift_reports)} 项, {drift_count} 项漂移")
    for r in drift_reports:
        icon = "⚠" if r.drifted else "✓"
        print(f"             {icon} {r.type.value}: {r.current_value:.2f} (阈值 {r.threshold:.2f})")
    print("═" * 55)
    print()


# ─── 测试 ──────────────────────────────────────────────────────


def test_executor():
    from dag import TaskNode, TaskGraph

    g = TaskGraph("test-exec")
    g.add_node(TaskNode(id="A", goal="任务A"))
    g.add_node(TaskNode(id="B", goal="任务B", dependencies=["A"]))

    executor = Executor()
    results = executor.execute_graph(g)
    assert len(results) == 2
    assert results["A"].success
    assert results["B"].success
    print(f"Executor: ✓ ({executor.summary()})")


def test_executor_with_runner():
    from dag import TaskNode, AgentType, TaskGraph

    calls = []

    def cc_runner(node):
        calls.append(node.id)
        return ExecutionResult(node_id=node.id, success=True, output=f"CC: {node.goal}")

    g = TaskGraph("test-runner")
    g.add_node(TaskNode(id="A", goal="写代码", agent=AgentType.CC))

    executor = Executor()
    executor.register_runner("cc", cc_runner)
    executor.execute_graph(g)
    assert "A" in calls
    print(f"Executor with runner: ✓")


def test_critic():
    critic = Critic()

    # 模拟 DAG
    from dag import TaskNode, TaskGraph
    g = TaskGraph("test-qa")
    node_a = TaskNode(id="A", goal="好的输出")
    node_a.status = type('s', (), {'value': 'completed'})()
    node_a.result = "这是一个正常的输出结果，足够长"
    g.add_node(node_a)

    node_b = TaskNode(id="B", goal="空的输出")
    node_b.status = type('s', (), {'value': 'completed'})()
    node_b.result = ""
    g.add_node(node_b)

    result = critic.check(g)
    assert not result.passed, "应有质检不通过"
    assert len(result.issues) > 0
    print(f"Critic: ✓ ({len(result.issues)} issues, score={result.score:.2f})")


def test_aggregator():
    from dag import TaskNode, TaskGraph

    g = TaskGraph("test-agg")
    n1 = TaskNode(id="A", goal="第一步")
    n1.result = "第一步结果"
    n1.status = type('s', (), {'value': 'completed'})()
    g.add_node(n1)

    n2 = TaskNode(id="B", goal="第二步", dependencies=["A"])
    n2.result = "第二步结果"
    n2.status = type('s', (), {'value': 'completed'})()
    g.add_node(n2)

    agg = Aggregator()
    result = agg.aggregate("test", g)
    assert result.success_count == 2
    assert "第一步结果" in result.combined
    assert "第二步结果" in result.combined
    print(f"Aggregator: ✓ ({result.success_count}/{result.node_count})")


def test_full_pipeline():
    from dag import TaskNode, TaskGraph

    g = TaskGraph("full-test")
    g.add_node(TaskNode(id="step1", goal="准备数据并校验完整性", cost_estimate=500))
    g.add_node(TaskNode(id="step2", goal="根据数据类型执行转换和清洗", dependencies=["step1"], cost_estimate=1000))

    pipeline = FullPipeline()

    def my_runner(node):
        time.sleep(0.01)
        return f"✔ 完成: {node.goal} — 输出通过验证，共处理 128 条记录，耗时 1.2s"

    result = pipeline.run("full_test", g, runner=my_runner)
    meta = result.get("_meta", {})
    assert meta.get("success", 0) == 2, f"success={meta.get('success')}, result keys={list(result.keys())}"
    assert meta.get("failed", 0) == 0
    assert meta.get("qa_score", 0) > 0
    assert all(step in result for step in ["step1", "step2"])
    print(f"FullPipeline: ✓ (duration={meta.get('duration', 0):.2f}s, "
          f"qa={meta.get('qa_score', 0):.2f}, eval={meta.get('eval_score', 0):.2f})")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/d")
    logging.basicConfig(level=logging.WARNING)
    print("═══ 执行器集成测试 ═══\n")
    test_executor()
    test_executor_with_runner()
    print()
    test_critic()
    print()
    test_aggregator()
    print()
    test_full_pipeline()
    print("\n═══ 全部测试通过 ═══")
