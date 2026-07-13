"""
DAG 任务图引擎 — 多Agent协作框架的基元层

核心概念：
  TaskNode     — DAG 中的一个任务节点
  TaskGraph    — 有向无环图，管理节点依赖与执行顺序
  Planner      — 将目标拆解为 TaskGraph

设计原则（四条规则替代规则堆叠）：
  1. 所有行为从基元 (Node/Edge/State/Event) 推导
  2. 不枚举规则 — 用优先级链和 Utility 函数动态判定
  3. 可并行、可回滚 — DAG 拓扑排序确保依赖正确
  4. 每个 Node 五要素：input / dependency / cost / risk / agent

Usage:
    from agent_framework.dag import Planner, TaskGraph, TaskNode

    # 自动规划（需 LLM 支持）
    graph = Planner.plan("做一个 XHS 封面图", context={...})

    # 或手动构建
    g = TaskGraph("my-task")
    g.add_node(TaskNode(id="step1", goal="...", agent="cc", ...))
    g.add_node(TaskNode(id="step2", goal="...", agent="codex", deps=["step1"], ...))
    g.validate()
    result = g.execute(runner=my_runner)
"""

from __future__ import annotations

import json
import time
import logging
from enum import Enum
from typing import Any, Callable, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("agent-framework.dag")


# ─── 基元定义 ─────────────────────────────────────────────────


class AgentType(str, Enum):
    """Agent 类型 — 执行层三通道"""
    HERMES = "hermes"       # 文本/简单任务
    CC = "cc"               # 代码/架构/长上下文
    CODEX = "codex"         # 视频/图片/短周期
    MANUAL = "manual"       # 需要用户手动操作
    CUSTOM = "custom"       # 自定义 Agent


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskNode:
    """DAG 中的一个任务节点 — 五要素完整"""
    id: str
    goal: str                                    # input: 做什么
    agent: AgentType | str = AgentType.HERMES    # agent: 谁执行
    cost_estimate: int = 0                       # cost: 预计 token 数
    risk: RiskLevel = RiskLevel.LOW              # risk: 风险等级
    dependencies: list[str] = field(default_factory=list)  # dependency: 依赖节点ID
    context: dict[str, Any] = field(default_factory=dict)  # 额外上下文
    timeout: int = 300                           # 超时秒数
    retry_count: int = 0                         # 当前重试次数
    max_retries: int = 2                         # 最大重试次数
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float | None:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    @property
    def is_leaf(self) -> bool:
        """没有出边的节点是叶子节点"""
        return True  # 由 TaskGraph 维护

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal": self.goal,
            "agent": self.agent.value if isinstance(self.agent, AgentType) else self.agent,
            "cost_estimate": self.cost_estimate,
            "risk": self.risk.value,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "error": self.error,
            "duration": self.duration,
            "retry_count": self.retry_count,
        }

    def __repr__(self) -> str:
        return f"<TaskNode {self.id} agent={self.agent.value if isinstance(self.agent, AgentType) else self.agent} status={self.status.value}>"


# ─── DAG 引擎 ─────────────────────────────────────────────────


class TaskGraph:
    """有向无环图 — 管理任务节点与依赖关系"""

    def __init__(self, name: str = "", metadata: dict | None = None):
        self.name = name
        self.nodes: dict[str, TaskNode] = {}
        self._edges: dict[str, set[str]] = {}   # node_id → set of dependents
        self._rev_edges: dict[str, set[str]] = {}  # node_id → set of dependencies (反向)
        self.metadata: dict = metadata or {}
        self.created_at: float = time.time()
        self.executed_at: float | None = None
        self.completed_at: float | None = None
        self._result: dict[str, Any] = {}

    def add_node(self, node: TaskNode) -> TaskNode:
        """添加节点到 DAG"""
        if node.id in self.nodes:
            raise ValueError(f"节点已存在: {node.id}")
        self.nodes[node.id] = node
        self._edges[node.id] = set()
        self._rev_edges[node.id] = set(node.dependencies) if node.dependencies else set()
        for dep_id in (node.dependencies or []):
            if dep_id not in self._edges:
                self._edges[dep_id] = set()
            self._edges[dep_id].add(node.id)
        return node

    def add_edge(self, from_id: str, to_id: str) -> None:
        """添加依赖边: from → to (from 必须在 to 之前完成)"""
        if from_id not in self.nodes or to_id not in self.nodes:
            raise ValueError(f"节点不存在: {from_id} 或 {to_id}")
        self._edges[from_id].add(to_id)
        self._rev_edges[to_id].add(from_id)
        # 同步更新 node.dependencies
        if from_id not in self.nodes[to_id].dependencies:
            self.nodes[to_id].dependencies.append(from_id)

    def validate(self) -> list[str]:
        """验证 DAG 合法性，返回错误列表。空列表=合法"""
        errors = []

        # 1. 检测环 (Kahn's algorithm)
        in_degree: dict[str, int] = {}
        for nid in self.nodes:
            in_degree[nid] = len(self._rev_edges[nid])

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        visited = 0

        while queue:
            nid = queue.pop(0)
            visited += 1
            for dep_id in self._edges[nid]:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    queue.append(dep_id)

        if visited != len(self.nodes):
            errors.append(f"DAG 包含环: 可遍历 {visited}/{len(self.nodes)} 节点")

        # 2. 检查依赖不存在
        for nid, node in self.nodes.items():
            for dep_id in (node.dependencies or []):
                if dep_id not in self.nodes:
                    errors.append(f"节点 {nid} 依赖不存在的节点: {dep_id}")

        return errors

    def topological_sort(self) -> list[list[str]]:
        """返回拓扑排序的并行批次 — 同一批次的节点无依赖关系，可并行执行"""
        in_degree: dict[str, int] = {}
        for nid in self.nodes:
            in_degree[nid] = len(self._rev_edges[nid])

        result: list[list[str]] = []
        remaining = set(self.nodes.keys())

        while remaining:
            batch = [nid for nid in remaining if in_degree[nid] == 0]
            if not batch:
                raise RuntimeError(f"DAG 包含环: 剩余 {len(remaining)} 个节点无法排入")
            for nid in batch:
                remaining.remove(nid)
                for dep_id in self._edges[nid]:
                    in_degree[dep_id] -= 1
            result.append(batch)

        return result

    def get_dependents(self, node_id: str) -> list[str]:
        """获取直接依赖此节点的所有节点"""
        return list(self._edges.get(node_id, set()))

    def get_dependencies(self, node_id: str) -> list[str]:
        """获取此节点的所有直接依赖"""
        return list(self._rev_edges.get(node_id, set()))

    def get_ancestors(self, node_id: str) -> set[str]:
        """获取所有祖先节点（递归）"""
        ancestors: set[str] = set()
        stack = list(self._rev_edges.get(node_id, set()))
        while stack:
            nid = stack.pop()
            if nid not in ancestors:
                ancestors.add(nid)
                stack.extend(self._rev_edges.get(nid, set()))
        return ancestors

    def get_descendants(self, node_id: str) -> set[str]:
        """获取所有子孙节点（递归）"""
        descendants: set[str] = set()
        stack = list(self._edges.get(node_id, set()))
        while stack:
            nid = stack.pop()
            if nid not in descendants:
                descendants.add(nid)
                stack.extend(self._edges.get(nid, set()))
        return descendants

    def get_leaf_nodes(self) -> list[str]:
        """叶子节点 = 没有依赖者的节点"""
        return [nid for nid in self.nodes if not self._edges.get(nid)]

    def get_root_nodes(self) -> list[str]:
        """根节点 = 没有依赖的节点"""
        return [nid for nid in self.nodes if not self._rev_edges.get(nid)]

    def estimate_total_cost(self) -> int:
        """估算 DAG 总成本（token 数）"""
        return sum(n.cost_estimate for n in self.nodes.values())

    def estimate_parallel_time(self) -> float:
        """估算并行执行时间（秒）
        假设 cost 正比于执行时间，1 token ≈ 0.01s（粗略估计）
        """
        batches = self.topological_sort()
        total = 0.0
        for batch in batches:
            max_cost = max((self.nodes[nid].cost_estimate for nid in batch), default=0)
            total += max_cost * 0.01
        return total

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "batches": self.topological_sort(),
            "total_cost": self.estimate_total_cost(),
            "node_count": len(self.nodes),
            "created_at": self.created_at,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    # ─── 执行 ────────────────────────────────────────────────

    def execute(self, runner: Callable[[TaskNode], Any]) -> dict[str, Any]:
        """执行整个 DAG，按拓扑批次顺序执行

        Args:
            runner: 执行单个节点的回调函数，接收 TaskNode 返回结果

        Returns:
            {node_id: result, ...}
        """
        self.executed_at = time.time()
        batches = self.topological_sort()

        for batch in batches:
            for nid in batch:
                node = self.nodes[nid]
                if node.status == NodeStatus.SKIPPED:
                    continue

                logger.info(f"执行节点: {nid} (agent={node.agent})")
                node.status = NodeStatus.RUNNING
                node.started_at = time.time()

                try:
                    result = runner(node)
                    node.result = result
                    node.status = NodeStatus.COMPLETED
                    node.completed_at = time.time()
                    self._result[nid] = result
                    logger.info(f"节点完成: {nid} ({node.duration:.1f}s)")
                except Exception as e:
                    node.error = str(e)
                    if node.retry_count < node.max_retries:
                        node.retry_count += 1
                        node.status = NodeStatus.PENDING
                        logger.warning(f"节点失败，重试 {node.retry_count}/{node.max_retries}: {nid} — {e}")
                        # 重新插入队列头
                        batch.insert(batch.index(nid) + 1, nid)
                        continue
                    node.status = NodeStatus.FAILED
                    node.completed_at = time.time()
                    logger.error(f"节点失败 (已达最大重试): {nid} — {e}")

        self.completed_at = time.time()
        return self._result

    def skip_node(self, node_id: str) -> None:
        """跳过某个节点（标记为 skipped）"""
        if node_id in self.nodes:
            self.nodes[node_id].status = NodeStatus.SKIPPED

    def reset(self) -> None:
        """重置所有节点状态"""
        for node in self.nodes.values():
            node.status = NodeStatus.PENDING
            node.result = None
            node.error = None
            node.started_at = None
            node.completed_at = None
            node.retry_count = 0
        self._result = {}
        self.executed_at = None
        self.completed_at = None

    def reset_node(self, node_id: str) -> None:
        """重置单个节点为 PENDING"""
        if node_id in self.nodes:
            n = self.nodes[node_id]
            n.status = NodeStatus.PENDING
            n.result = None
            n.error = None
            n.started_at = None
            n.completed_at = None
            n.retry_count = 0

    @property
    def all_completed(self) -> bool:
        return all(
            n.status in (NodeStatus.COMPLETED, NodeStatus.SKIPPED)
            for n in self.nodes.values()
        )

    @property
    def has_failures(self) -> bool:
        return any(n.status == NodeStatus.FAILED for n in self.nodes.values())

    @property
    def total_duration(self) -> float | None:
        if self.executed_at and self.completed_at:
            return self.completed_at - self.executed_at
        return None

    def summary(self) -> str:
        stats = {"completed": 0, "failed": 0, "skipped": 0, "pending": 0, "running": 0}
        for n in self.nodes.values():
            stats[n.status.value] = stats.get(n.status.value, 0) + 1
        return (
            f"TaskGraph '{self.name}': "
            f"{stats['completed']}✓ {stats['failed']}✗ {stats['skipped']}⊘ {stats['pending']}○ {stats['running']}▶ "
            f"| 总耗时: {self.total_duration:.1f}s" if self.total_duration else "| 未执行"
        )


# ─── Planner ──────────────────────────────────────────────────


class Planner:
    """Planner — 将目标拆解为 DAG

    两种模式：
    1. manual: 手动构建 DAG（当前适用）
    2. llm: 通过 LLM 自动拆解（需要外部 LLM 调用）
    """

    @staticmethod
    def create_linear(goal: str, steps: list[dict]) -> TaskGraph:
        """创建线性 DAG — 步骤依次执行

        Args:
            goal: 任务名称
            steps: 步骤列表，每项 {
                "id": str,
                "goal": str,
                "agent": AgentType | str,
                "cost_estimate": int,
                "risk": RiskLevel,
                "context": dict,
            }
        """
        graph = TaskGraph(name=goal)
        prev_id = None
        for i, step in enumerate(steps):
            node = TaskNode(
                id=step.get("id", f"step_{i}"),
                goal=step["goal"],
                agent=step.get("agent", AgentType.HERMES),
                cost_estimate=step.get("cost_estimate", 0),
                risk=step.get("risk", RiskLevel.LOW),
                dependencies=[prev_id] if prev_id else [],
                context=step.get("context", {}),
            )
            graph.add_node(node)
            prev_id = node.id
        return graph

    @staticmethod
    def create_parallel(goal: str, parallel_steps: list[dict],
                        merge_step: dict | None = None) -> TaskGraph:
        """创建并行 DAG — 多个步骤并行执行，可选合并步骤

        Args:
            goal: 任务名称
            parallel_steps: 并行步骤列表，同 create_linear 的 steps
            merge_step: 可选的合并步骤，依赖所有并行步骤
        """
        graph = TaskGraph(name=goal)
        parallel_ids = []

        for i, step in enumerate(parallel_steps):
            nid = step.get("id", f"parallel_{i}")
            node = TaskNode(
                id=nid,
                goal=step["goal"],
                agent=step.get("agent", AgentType.HERMES),
                cost_estimate=step.get("cost_estimate", 0),
                risk=step.get("risk", RiskLevel.LOW),
                dependencies=step.get("dependencies", []),
                context=step.get("context", {}),
            )
            graph.add_node(node)
            parallel_ids.append(nid)

        if merge_step:
            merge_node = TaskNode(
                id=merge_step.get("id", "merge"),
                goal=merge_step["goal"],
                agent=merge_step.get("agent", AgentType.HERMES),
                cost_estimate=merge_step.get("cost_estimate", 0),
                risk=merge_step.get("risk", RiskLevel.LOW),
                dependencies=parallel_ids,
                context=merge_step.get("context", {}),
            )
            graph.add_node(merge_node)

        return graph

    @staticmethod
    def create_from_dict(data: dict) -> TaskGraph:
        """从字典恢复 DAG"""
        graph = TaskGraph(name=data.get("name", ""))
        for node_data in data.get("nodes", []):
            deps_raw = node_data.get("dependencies", node_data.get("deps", []))
            agent_raw = node_data.get("agent", "hermes")
            try:
                agent = AgentType(agent_raw)
            except ValueError:
                agent = agent_raw
            node = TaskNode(
                id=node_data["id"],
                goal=node_data["goal"],
                agent=agent,
                cost_estimate=node_data.get("cost_estimate", 0),
                risk=RiskLevel(node_data.get("risk", "low")),
                dependencies=deps_raw,
                context=node_data.get("context", {}),
                timeout=node_data.get("timeout", 300),
                max_retries=node_data.get("max_retries", 2),
            )
            graph.add_node(node)
        return graph


# ─── 快捷构建函数 ──────────────────────────────────────────────


def linear_pipeline(name: str, steps: list[dict]) -> TaskGraph:
    """快捷：线性流水线"""
    return Planner.create_linear(name, steps)


def parallel_pipeline(name: str, branches: list[list[dict]],
                      merge: dict | None = None) -> TaskGraph:
    """快捷：多分支并行流水线"""
    graph = TaskGraph(name=name)
    all_tips = []

    for branch_steps in branches:
        prev_id = None
        for i, step in enumerate(branch_steps):
            nid = step.get("id", f"b{all_tips.count(None)}_{i}")
            deps = list(step.get("dependencies", []))
            if prev_id:
                deps.append(prev_id)
            node = TaskNode(
                id=nid,
                goal=step["goal"],
                agent=step.get("agent", AgentType.HERMES),
                cost_estimate=step.get("cost_estimate", 0),
                risk=step.get("risk", RiskLevel.LOW),
                dependencies=deps,
                context=step.get("context", {}),
            )
            graph.add_node(node)
            prev_id = nid
        if prev_id:
            all_tips.append(prev_id)

    if merge and all_tips:
        merge_node = TaskNode(
            id=merge.get("id", "merge"),
            goal=merge["goal"],
            agent=merge.get("agent", AgentType.HERMES),
            dependencies=all_tips,
        )
        graph.add_node(merge_node)

    return graph


# ─── 验证和测试 ────────────────────────────────────────────────


def test_dag():
    """基本 DAG 功能测试"""
    g = TaskGraph("test")

    g.add_node(TaskNode(id="A", goal="下载图片", agent=AgentType.CC,
                        cost_estimate=500, risk=RiskLevel.LOW))
    g.add_node(TaskNode(id="B", goal="生成文案", agent=AgentType.HERMES,
                        cost_estimate=200, risk=RiskLevel.LOW, dependencies=["A"]))
    g.add_node(TaskNode(id="C", goal="生成封面", agent=AgentType.CODEX,
                        cost_estimate=3000, risk=RiskLevel.MEDIUM, dependencies=["A"]))
    g.add_node(TaskNode(id="D", goal="合并输出", agent=AgentType.HERMES,
                        cost_estimate=100, risk=RiskLevel.LOW,
                        dependencies=["B", "C"]))

    # 验证
    errors = g.validate()
    assert not errors, f"验证失败: {errors}"

    # 拓扑排序
    batches = g.topological_sort()
    assert len(batches) >= 2, f"应该至少有2个批次: {batches}"

    print(f"节点: {len(g.nodes)}")
    print(f"并行批次: {batches}")
    print(f"根节点: {g.get_root_nodes()}")
    print(f"叶子节点: {g.get_leaf_nodes()}")
    print(f"估计总成本: {g.estimate_total_cost()} token")
    print(f"估计并行时间: {g.estimate_parallel_time():.1f}s")
    print(f"DAG JSON:\n{g.to_json()}")
    print("DAG 基础功能: ✓")


def test_planner():
    """Planner 功能测试"""
    # 线性流水线
    steps = [
        {"id": "research", "goal": "调研竞品", "agent": AgentType.HERMES, "cost_estimate": 1000},
        {"id": "draft", "goal": "写初稿", "agent": AgentType.CC, "cost_estimate": 2000},
        {"id": "review", "goal": "审查质量", "agent": AgentType.HERMES, "cost_estimate": 500},
    ]
    g = Planner.create_linear("写报告", steps)
    assert not g.validate(), f"线性 DAG 验证失败"
    assert len(g.topological_sort()) == 3, f"线性 DAG 应该 3 批"
    print(f"线性流水线: {g.topological_sort()}")

    # 并行流水线
    parallel = [
        {"id": "p1", "goal": "并行任务1", "agent": AgentType.CODEX, "cost_estimate": 2000},
        {"id": "p2", "goal": "并行任务2", "agent": AgentType.HERMES, "cost_estimate": 500},
    ]
    merge = {"id": "merge", "goal": "合并结果", "agent": AgentType.HERMES}
    g2 = Planner.create_parallel("并行测试", parallel, merge)
    assert not g2.validate(), f"并行 DAG 验证失败"
    batches = g2.topological_sort()
    assert len(batches) == 2, f"应该 2 批 (并行+合并), 得到: {batches}"
    print(f"并行流水线: {batches}")

    print("Planner 功能: ✓")


def test_cycle_detection():
    """环检测测试"""
    g = TaskGraph("cycle-test")
    g.add_node(TaskNode(id="A", goal="A"))
    g.add_node(TaskNode(id="B", goal="B", dependencies=["A"]))
    g.add_node(TaskNode(id="C", goal="C", dependencies=["B"]))
    g.add_edge("C", "A")  # 人为制造环

    errors = g.validate()
    assert len(errors) > 0, "环检测应该发现错误"
    print(f"环检测: ✓ ({errors[0]})")


def test_execute():
    """执行测试"""
    results = {}

    def runner(node: TaskNode) -> str:
        return f"完成: {node.goal}"

    g = TaskGraph("exec-test")
    g.add_node(TaskNode(id="A", goal="任务A"))
    g.add_node(TaskNode(id="B", goal="任务B", dependencies=["A"]))
    g.add_node(TaskNode(id="C", goal="任务C", dependencies=["B"]))

    g.execute(runner)
    assert g.all_completed, "所有节点应该完成"
    assert not g.has_failures, "不应有失败"
    print(f"执行结果: {g.summary()}")
    print(f"总耗时: {g.total_duration:.3f}s")
    print("DAG 执行: ✓")


def test_retry():
    """重试测试"""
    attempt = {"A": 0}

    def flaky_runner(node: TaskNode) -> str:
        attempt[node.id] = attempt.get(node.id, 0) + 1
        if node.id == "A" and attempt["A"] < 2:
            raise RuntimeError("临时失败")
        return f"OK: {node.goal}"

    g = TaskGraph("retry-test")
    g.add_node(TaskNode(id="A", goal="不稳定任务", max_retries=2))
    g.add_node(TaskNode(id="B", goal="后续任务", dependencies=["A"]))

    g.execute(flaky_runner)
    assert g.all_completed, "重试后应全部完成"
    print(f"重试测试: ✓ (任务A尝试了 {attempt['A']} 次)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("═══ DAG 引擎测试 ═══\n")
    test_dag()
    print()
    test_planner()
    print()
    test_cycle_detection()
    print()
    test_execute()
    print()
    test_retry()
    print("\n═══ 全部测试通过 ═══")
