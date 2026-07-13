# 超级楠 · 多Agent协作框架 v5

> DAG + 状态机 + 仲裁器 + 评估闭环 —— 让一群Agent像一支团队一样协作。

## 是什么

超级楠是一个多Agent协作编排框架，解决「多个AI Agent怎么配合干活」的问题。

**四层核心：**
- **DAG编排** — 任务拆解、依赖管理、并行执行
- **状态机** — 任务生命周期：INIT → PLANNED → EXECUTING → QA → FINALIZED
- **仲裁器** — 冲突裁决（5级优先级链）、Utility财务模型
- **评估闭环** — ScoreCard评分、漂移检测、策略自动调优

## 快速上手

```python
from dag import TaskGraph, TaskNode, AgentType

g = TaskGraph("我的任务")
g.add_node(TaskNode(id="research", goal="调研竞品", agent=AgentType.HERMES, cost_estimate=500, risk="low"))
g.add_node(TaskNode(id="draft", goal="写文案", agent=AgentType.CC, dependencies=["research"]))
g.validate()

batches = g.topological_sort()  # 获取并行执行批次
```

## 模块一览

| 模块 | 功能 |
|------|------|
| dag.py | DAG引擎 — TaskGraph + TaskNode + 拓扑排序 |
| state_machine.py | 状态机 — 13种事件 + Level 1-3回滚 |
| arbiter.py | 仲裁器 — 5级优先级 + Utility = TaskValue - Cost - RiskPenalty |
| memory.py | 记忆一致性 — L1强一致/L2弱一致/L4乐观锁 |
| eval_loop.py | 评估闭环 — ScoreCard + 三轴漂移检测 |
| executor.py | 集成管线 — FullPipeline + Critic质检 |
| runners.py | 三通道执行器（CC/Codex/Hermes） |
| hermes_memory.py | Hermes记忆适配层 |
| supports.py | 外围支撑 — 熔断/缓存/预算/恢复 |
| supports_ext.py | 扩展支撑 — 协议/访问控制/告警 |
| alerts.py | 告警推送（微信+控制台） |
| pipeline_stages.py | 各段逻辑实现 |

## 四通道路由

| 任务类型 | 执行通道 |
|---------|---------|
| 对话/规划/搜索/看图 | Hermes |
| 小修改（<200行） | Codex |
| 写代码/重构（≥200行） | Qwen Code（主力）→ CC（备选） |
| GitHub操作 | CC |

## License

MIT
