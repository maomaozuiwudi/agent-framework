"""
多Agent协作框架 v5 — 验收脚本

跑完全部模块测试 + 端到端管线验证，输出报告到 D:\acceptance-report.txt

用法：python D:\agent_framework\acceptance.py
"""

import sys, os, time, json, importlib

sys.path.insert(0, "D:")
os.chdir("D:/agent_framework")

REPORT = []
PASS = 0
FAIL = 0
TOTAL = 0


def test(name, fn):
    global PASS, FAIL, TOTAL
    TOTAL += 1
    try:
        fn()
        REPORT.append(f"  ✓ {name}")
        PASS += 1
    except Exception as e:
        REPORT.append(f"  ✗ {name}: {e}")
        FAIL += 1


def section(title):
    REPORT.append(f"\n{'='*50}")
    REPORT.append(f"  {title}")
    REPORT.append(f"{'='*50}")


# ═══════════════════════════════════════════════════════
# 1. DAG
# ═══════════════════════════════════════════════════════

def test_dag_basic():
    from dag import TaskNode, TaskGraph
    g = TaskGraph("test")
    g.add_node(TaskNode(id="A", goal="任务A"))
    g.add_node(TaskNode(id="B", goal="任务B", dependencies=["A"]))
    assert not g.validate()
    assert len(g.topological_sort()) == 2

def test_dag_parallel():
    from dag import TaskNode, TaskGraph
    g = TaskGraph("parallel")
    g.add_node(TaskNode(id="A", goal="根"))
    g.add_node(TaskNode(id="B", goal="并行1", dependencies=["A"]))
    g.add_node(TaskNode(id="C", goal="并行2", dependencies=["A"]))
    g.add_node(TaskNode(id="D", goal="合并", dependencies=["B", "C"]))
    batches = g.topological_sort()
    assert len(batches) == 3
    assert len(batches[1]) == 2

def test_dag_cycle():
    from dag import TaskNode, TaskGraph
    g = TaskGraph("cycle")
    g.add_node(TaskNode(id="A", goal="A"))
    g.add_node(TaskNode(id="B", goal="B", dependencies=["A"]))
    g.add_edge("B", "A")
    assert len(g.validate()) > 0

def test_dag_execute():
    from dag import TaskNode, TaskGraph
    g = TaskGraph("exec")
    g.add_node(TaskNode(id="A", goal="A"))
    g.add_node(TaskNode(id="B", goal="B", dependencies=["A"]))
    calls = []
    def runner(n): calls.append(n.id); return f"ok:{n.id}"
    g.execute(runner)
    assert calls == ["A", "B"]
    assert g.all_completed

def test_dag_retry():
    from dag import TaskNode, TaskGraph
    g = TaskGraph("retry")
    g.add_node(TaskNode(id="A", goal="A", max_retries=2))
    attempt = [0]
    def runner(n):
        attempt[0] += 1
        if attempt[0] < 2: raise RuntimeError("临时失败")
        return "ok"
    g.execute(runner)
    assert g.all_completed
    assert attempt[0] == 2

def test_planner_linear():
    from dag import Planner, AgentType
    g = Planner.create_linear("线性", [{"id":"a","goal":"A","agent":AgentType.HERMES},{"id":"b","goal":"B","agent":AgentType.CC}])
    assert len(g.nodes) == 2
    assert not g.validate()

def test_planner_parallel():
    from dag import Planner, AgentType
    g = Planner.create_parallel("并行", [
        {"id":"p1","goal":"并行1","agent":AgentType.CODEX},
        {"id":"p2","goal":"并行2","agent":AgentType.HERMES},
    ], {"id":"m","goal":"合并"})
    assert len(g.nodes) == 3
    batches = g.topological_sort()
    assert len(batches) == 2


# ═══════════════════════════════════════════════════════
# 2. 状态机
# ═══════════════════════════════════════════════════════

def test_sm_forward():
    from state_machine import TaskStateMachine, Event, TaskState
    sm = TaskStateMachine("t1")
    for evt in [Event.TASK_RECEIVED, Event.PLAN_COMPLETE, Event.EXECUTE_START,
                Event.EXECUTE_COMPLETE, Event.QA_PASS, Event.MEMORY_WRITE_DONE]:
        sm.transition(evt, "")
    assert sm.state == TaskState.FINALIZED and sm.is_terminal

def test_sm_rollback():
    from state_machine import TaskStateMachine, Event, TaskState, RollbackLevel
    sm = TaskStateMachine("t2")
    sm.transition(Event.TASK_RECEIVED, "")
    sm.transition(Event.PLAN_COMPLETE, "")
    sm.transition(Event.EXECUTE_START, "")
    sm.transition(Event.ERROR_OCCURRED, "")
    assert sm.state == TaskState.FAILED
    sm.transition(Event.RECOVERY_START, "")
    sm.rollback_level = RollbackLevel.LEVEL_1
    sm.transition(Event.RECOVERY_COMPLETE, "")
    assert sm.state == TaskState.EXECUTING

def test_sm_invalid():
    from state_machine import TaskStateMachine, Event, InvalidTransitionError
    sm = TaskStateMachine("t3")
    try:
        sm.transition(Event.MEMORY_WRITE_DONE, "")
        assert False, "应拒绝非法转换"
    except InvalidTransitionError:
        pass

def test_sm_history():
    from state_machine import TaskStateMachine, Event
    sm = TaskStateMachine("t4")
    sm.transition(Event.TASK_RECEIVED, "收到")
    assert len(sm.history) == 1
    assert sm.history[0].event == Event.TASK_RECEIVED


# ═══════════════════════════════════════════════════════
# 3. 仲裁器
# ═══════════════════════════════════════════════════════

def test_arbiter_safety():
    from arbiter import Arbiter, ConflictEvent
    a = Arbiter()
    d = a.resolve(ConflictEvent(type="safety_breach", source_a="a", source_b="b", claim_a="不安全", claim_b="安全"))
    assert d.resolution == "fuse_and_escalate"

def test_arbiter_qa():
    from arbiter import Arbiter, ConflictEvent
    a = Arbiter()
    d = a.resolve(ConflictEvent(type="qa_vs_executor", source_a="critic", source_b="executor", claim_a="有bug", claim_b="按设计"))
    assert d.winning_side == "critic"

def test_arbiter_agent_conflict():
    from arbiter import Arbiter, ConflictEvent
    a = Arbiter()
    d = a.resolve(ConflictEvent(type="agent_conflict", source_a="cc", source_b="hermes", claim_a="用A方案", claim_b="用B方案", context={"confidence_a":0.9,"confidence_b":0.6}))
    assert d.winning_side == "cc"

def test_utility():
    from arbiter import UtilityInput, compute_utility
    u1 = compute_utility(UtilityInput(task_value=0.9, cost_tokens=500, risk_penalty=0.0))
    assert u1.is_worthwhile
    u2 = compute_utility(UtilityInput(task_value=0.1, cost_tokens=50000, risk_penalty=0.5))
    assert not u2.is_worthwhile

def test_arbiter_rollback():
    from arbiter import Arbiter, ConflictEvent
    a = Arbiter()
    d = a.resolve(ConflictEvent(type="rollback_decision", source_a="a", source_b="b", claim_a="", claim_b="", context={"error_level": 3}))
    assert d.resolution == "rollback_level3"


# ═══════════════════════════════════════════════════════
# 4. 记忆层
# ═══════════════════════════════════════════════════════

def test_memory_l1():
    from memory import MemoryStore, ConsistencyLevel
    m = MemoryStore()
    m.write_l1("k1", "v1")
    assert m.read_l1("k1").value == "v1"

def test_memory_l2_buffer():
    from memory import MemoryStore
    m = MemoryStore()
    m.write_l2("a", "1"); m.write_l2("b", "2")
    assert len(m._l2_buffer) == 2
    m.write_l2("c", "3")
    assert len(m._l2_buffer) == 0

def test_memory_l4_version():
    from memory import MemoryStore
    m = MemoryStore()
    m.write_l4("k", {"x": 1})
    assert m.get_l4_version("k") >= 1

def test_memory_merge_human():
    from memory import MemoryStore
    m = MemoryStore()
    m.write_l4("cfg", {"theme": "dark"})
    m.write_l4("cfg", {"theme": "light"}, source="human", version=0)
    assert m.read_l4("cfg").value["theme"] == "light"

def test_memory_merge_field():
    from memory import MemoryStore
    m = MemoryStore()
    m.write_l4("u", {"a": 1, "b": 2})
    v = m.get_l4_version("u")
    m.write_l4("u", {"a": 10, "c": 3}, version=v-1)
    merged = m.read_l4("u").value
    assert merged["a"] == 10
    assert merged["b"] == 2
    assert merged["c"] == 3

def test_memory_merge_list():
    from memory import MemoryStore
    m = MemoryStore()
    m.write_l4("list", ["a", "b"])
    v = m.get_l4_version("list")
    m.write_l4("list", ["b", "c"], version=v-1)
    merged = m.read_l4("list").value
    assert "c" in merged
    assert "a" in merged

def test_memory_search():
    from memory import MemoryStore, ConsistencyLevel
    m = MemoryStore()
    m.write_l1("task:status", "running")
    m.write_l2("task:log", "started")
    results = m.search(key_prefix="task:")
    assert len(results) >= 2

def test_memory_ttl():
    from memory import MemoryStore
    m = MemoryStore()
    m.write_l1("temp", "x", ttl=0)
    import time; time.sleep(0.01)
    cleaned = m.cleanup_expired()
    assert cleaned >= 1


# ═══════════════════════════════════════════════════════
# 5. 评估闭环
# ═══════════════════════════════════════════════════════

def test_eval_basic():
    from eval_loop import EvalLoop
    e = EvalLoop()
    c = e.evaluate("t", "output", criteria={"quality": 0.9, "efficiency": 0.8})
    assert c.passed

def test_eval_low():
    from eval_loop import EvalLoop
    e = EvalLoop()
    c = e.evaluate("t", "bad", criteria={"quality": 0.1, "efficiency": 0.1})
    assert not c.passed
    assert c.needs_optimization

def test_eval_drift():
    from eval_loop import EvalLoop
    e = EvalLoop()
    reports = e.detect_drift({"prompt_structure": {"baseline_structure": "aaaa", "current_structure": "bbbb"}})
    assert len(reports) >= 1

def test_eval_policy():
    from eval_loop import EvalLoop
    e = EvalLoop()
    e.update_policy({"thresholds": {"quality_min": 0.9}})
    assert e.policy["thresholds"]["quality_min"] == 0.9


# ═══════════════════════════════════════════════════════
# 6. 执行器 + 集成管线
# ═══════════════════════════════════════════════════════

def test_executor_basic():
    from dag import TaskNode, TaskGraph
    from executor import Executor
    g = TaskGraph("t")
    g.add_node(TaskNode(id="A", goal="A"))
    g.add_node(TaskNode(id="B", goal="B", dependencies=["A"]))
    r = Executor().execute_graph(g)
    assert r["A"].success
    assert r["B"].success

def test_full_pipeline():
    from dag import TaskNode, TaskGraph
    from executor import FullPipeline
    g = TaskGraph("acceptance")
    g.add_node(TaskNode(id="s1", goal="第一步 数据处理和格式校验", cost_estimate=500))
    g.add_node(TaskNode(id="s2", goal="第二步 根据结果执行转换输出", dependencies=["s1"], cost_estimate=1000))
    p = FullPipeline()
    result = p.run("accept", g, runner=lambda n: f"完成: {n.goal} - 输出长度充足，通过验证")
    meta = result.get("_meta", {})
    assert meta.get("success", 0) == 2, f"success={meta.get('success')}"
    assert meta.get("qa_score", 0) > 0
    assert meta.get("eval_score", 0) > 0
    assert meta.get("utility", 0) > 0

def test_runners():
    from runners import HermesRunners
    from executor import Executor
    e = Executor()
    HermesRunners.register_all(e)
    assert "cc" in e.runners
    assert "codex" in e.runners
    assert "hermes" in e.runners

def test_hermes_memory_api():
    from hermes_memory import HermesMemory
    hm = HermesMemory()
    assert hm.write_l1("test:key", "val") is not None
    assert hm.write_l2("test:buf", "val") is not None
    assert isinstance(hm.search("test"), list)
    assert hm.write_l4("test:l4", {"x": 1}) is not None

def test_plan_xhs():
    from runners import plan_xhs_post
    g = plan_xhs_post("测试", "深蓝风")
    assert not g.validate()
    assert len(g.nodes) == 5


# ═══════════════════════════════════════════════════════
# 运行
# ═══════════════════════════════════════════════════════

print("开始验收...\n")

section("1. DAG 引擎")
test("基本DAG", test_dag_basic)
test("并行DAG", test_dag_parallel)
test("环检测", test_dag_cycle)
test("执行", test_dag_execute)
test("自动重试", test_dag_retry)
test("Planner线性", test_planner_linear)
test("Planner并行", test_planner_parallel)

section("2. 统一状态机")
test("正向流程", test_sm_forward)
test("回滚恢复", test_sm_rollback)
test("非法转换拒绝", test_sm_invalid)
test("历史审计", test_sm_history)

section("3. 统一仲裁器")
test("安全违规熔断", test_arbiter_safety)
test("质检覆盖执行", test_arbiter_qa)
test("Agent冲突置信度裁决", test_arbiter_agent_conflict)
test("Utility成本模型", test_utility)
test("三级回滚决策", test_arbiter_rollback)

section("4. 三层记忆协议")
test("L1强一致", test_memory_l1)
test("L2缓冲批量", test_memory_l2_buffer)
test("L4乐观锁版本", test_memory_l4_version)
test("Merge-人工覆盖", test_memory_merge_human)
test("Merge-字段级合并", test_memory_merge_field)
test("Merge-列表去重", test_memory_merge_list)
test("跨层搜索", test_memory_search)
test("TTL过期清理", test_memory_ttl)

section("5. 评估闭环")
test("基本评分", test_eval_basic)
test("低分强制优化", test_eval_low)
test("漂移检测", test_eval_drift)
test("策略更新", test_eval_policy)

section("6. 执行器 + 集成管线")
test("Executor基本执行", test_executor_basic)
test("三通道Runner注册", test_runners)
test("Hermes记忆适配器", test_hermes_memory_api)
test("XHS生产DAG规划", test_plan_xhs)

# ─── Supports tests ───

def test_sup_queue():
    from supports import TaskQueue, QueueItem, Priority
    q = TaskQueue()
    q.enqueue(QueueItem("t1", "普通"))
    q.enqueue(QueueItem("t2", "紧急", priority=Priority.CRITICAL))
    assert q.dequeue().task_id == "t2"

def test_sup_concurrency():
    from supports import TaskQueue, QueueItem
    q = TaskQueue()
    for i in range(5): q.enqueue(QueueItem(f"t{i}", "x", agent_type="cc"))
    assert sum(1 for _ in range(5) if q.dequeue("cc")) == TaskQueue.MAX_CONCURRENCY

def test_sup_circuit():
    from supports import CircuitBreaker
    cb = CircuitBreaker("t", failure_threshold=1, recovery_timeout=0.05)
    try: cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
    except: pass
    assert cb.state == "open"
    import time; time.sleep(0.06)
    assert cb.call(lambda: "ok") == "ok"

def test_sup_cache():
    from supports import TieredCache
    tc = TieredCache()
    tc.set("k", "v", tier="warm")
    tc.get("k"); tc.get("k"); tc.get("k")
    assert "k" in tc._hot

def test_sup_budget():
    from supports import BudgetMonitor
    bm = BudgetMonitor()
    assert bm.can_accept()[0]

def test_sup_feedback():
    from supports import FeedbackFilter
    ff = FeedbackFilter()
    ff.add("t", "v"); ff.add("t", "v"); ff.add("t", "v")
    assert ff.is_confirmed("t")

def test_sup_kpi():
    from supports import KPITracker
    kpi = KPITracker()
    kpi.record("success_rate", 0.9)
    assert kpi.average("success_rate") > 0

def test_sup_recovery():
    from supports import RecoveryAgent
    ra = RecoveryAgent()
    r = ra.recover("t", "err", 1)
    assert r["action"] == "retry_execute"
    r3 = ra.recover("t", "err", 3)
    assert r3["action"] == "escalate"

section("7. 外围支撑系统")
test("优先级队列", test_sup_queue)
test("并发上限", test_sup_concurrency)
test("故障熔断", test_sup_circuit)
test("缓存冷热分层", test_sup_cache)
test("预算监控", test_sup_budget)
test("负反馈抑制", test_sup_feedback)
test("KPI指标", test_sup_kpi)
test("Recovery Agent", test_sup_recovery)

# ─── supports_ext tests ───

def test_sup_ext_message():
    from supports_ext import Message, MessageType, MessageValidator, MessageBus
    msg = Message("m1", MessageType.TASK, "a", "b", {"x": 1})
    assert len(MessageValidator.validate(msg)) == 0
    bus = MessageBus()
    assert bus.send(msg)
def test_sup_ext_access():
    from supports_ext import MemoryAccessControl
    ac = MemoryAccessControl()
    assert ac.is_allowed("pref:theme", level="agent")
    assert not ac.is_allowed("secret:k", level="agent")
    assert "***" in ac.sanitize("key=sk-abc123xyz45678901234")  # 20+ chars after sk-
def test_sup_ext_intervention():
    from supports_ext import InterventionManager, InterventionPoint
    im = InterventionManager()
    ok, _ = im.intervene(InterventionPoint.AFTER_ROUTING, "t", "override_route", "x")
    assert ok
    ok, _ = im.intervene(InterventionPoint.AFTER_ROUTING, "t", "bypass_qa", "x")
    assert not ok
def test_sup_ext_cleanup():
    from supports_ext import MemoryCleanupManager
    cm = MemoryCleanupManager()
    e, a = cm.check_expired("k", "l3", time.time() - 86400*200)
    assert a == "delete"
    snap = cm.take_snapshot({"k": "v"})
    assert snap.id.startswith("snap-")
def test_sup_ext_special():
    from supports_ext import SpecialTaskAdapter
    sta = SpecialTaskAdapter()
    assert len(sta.detect_task_type(60000, True)) == 2
def test_sup_ext_discipline():
    from supports_ext import DisciplineWatcher
    dw = DisciplineWatcher()
    assert dw.watch("bypass_qa", "e", "t") is not None
    assert dw.watch("normal_run", "e", "t") is None

section("8. 扩展支撑系统（supports_ext）")
test("通信协议Schema", test_sup_ext_message)
test("记忆访问控制", test_sup_ext_access)
test("人工干预点", test_sup_ext_intervention)
test("过期清理+快照", test_sup_ext_cleanup)
test("特殊任务适配", test_sup_ext_special)
test("纪律告警", test_sup_ext_discipline)

# ─── pipeline_stages tests ───

def test_stage_cache():
    from pipeline_stages import CachePreChecker
    cpc = CachePreChecker()
    cpc.store("写一篇XHS", "x", "t1")
    assert cpc.check("写一篇XHS").exact
def test_stage_complexity():
    from pipeline_stages import ComplexityAnalyzer, ComplexityLevel
    ca = ComplexityAnalyzer()
    r = ca.analyze(estimated_tokens=60000, file_count=8, safety_required=True,
                   is_public=True, l5_error_rate=0.3, has_external_api=True)
    assert r.level in (ComplexityLevel.COMPLEX, ComplexityLevel.VERY_COMPLEX), f"got {r.level} score={r.score}"
def test_stage_review():
    from pipeline_stages import CrossReviewer
    from dag import TaskNode, TaskGraph
    cr = CrossReviewer()
    cr.register_reviewer("auto", cr.get_default_reviewer())
    g = TaskGraph("t")
    n = TaskNode(id="A", goal="正常输出通过验证")
    n.result = "足够长的验证输出内容"
    n.status = type('s', (), {'value': 'completed'})()
    g.add_node(n)
    assert len(cr.review(g)) >= 1
def test_stage_l1sync():
    from pipeline_stages import L1AutoSync
    l1 = L1AutoSync()
    l1.record_feedback("x", "v"); l1.record_feedback("x", "v"); l1.record_feedback("x", "v")
    assert l1.stats["synced"] == 1

section("9. 流水线各段逻辑")
test("段0缓存预检", test_stage_cache)
test("段2复杂度判定", test_stage_complexity)
test("段5交叉审查", test_stage_review)
test("段7 L1自动同步", test_stage_l1sync)

section("10. ⭐ 端到端 FullPipeline（核心验收）")
test("全流程：Utility→DAG执行→Critic质检→记忆写入→Eval→状态机终态", test_full_pipeline)

# 报告
section(f"验收结果: {PASS}/{TOTAL} 通过 ({FAIL} 失败)")
if FAIL == 0:
    REPORT.append("  ✅ 全部通过 — 框架核心验收合格")
else:
    REPORT.append(f"  ❌ {FAIL} 项失败，需排查")

report_text = "\n".join(REPORT)
report_path = "D:/acceptance-report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)

print(report_text)
print(f"\n报告已保存: {report_path}")
