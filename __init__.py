"""多Agent协作框架 v5 — 四层架构的 Python 实现

四层：DAG + 状态机 + 仲裁器 + 评估闭环

目录：
  dag.py           — TaskGraph DAG 引擎 (Planner→DAG→执行)
  state_machine.py — 统一状态机 (INIT→FINALIZED)
  arbiter.py       — 统一仲裁器 (优先级链 + Utility 成本模型)
  memory.py        — 记忆一致性协议 (L1/L2/L4 三层写入)
  eval_loop.py     — 评估闭环 (Eval Loop + Drift Detection)
  executor.py      — 执行器集成 (Executor/Critic/Aggregator)
"""
