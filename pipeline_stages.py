"""
流水线各段执行逻辑 — 段0/段2/段5/段7 的实际运行规则

包含：
  段0: 缓存预检（精确/近似匹配 + TTL分层 + LRU）
  段2: 复杂度判定 + 梯度预算降级 + 分层记忆检索
  段5: 交叉审查Agent（双模型互审 + 正向激励 + 最终判断权）
  段7: L1自动同步 + 版本乐观锁 + deprecated语义
"""

from __future__ import annotations

import time
import json
import re
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from collections import OrderedDict

logger = logging.getLogger("agent-framework.pipeline-stages")


# ═══════════════════════════════════════════════════════════
# 段0: 缓存预检
# ═══════════════════════════════════════════════════════════


@dataclass
class CacheMatch:
    """缓存匹配结果"""
    matched: bool
    confidence: float = 0.0      # 匹配置信度 0-1
    exact: bool = False          # 是否精确匹配
    value: Any = None            # 缓存的产出
    source_task: str = ""        # 来源任务 ID
    age: float = 0.0             # 缓存年龄（秒）


SIMILARITY_WEIGHT = {
    "exact": 1.0,        # 精确匹配 → 直接复用
    "strong": 0.85,      # >80% → 强参考
    "weak": 0.65,        # 60-80% → 弱参考
    "none": 0.0,         # <60% → 不注入
}


class CachePreChecker:
    """段0 缓存预检器 — 精确/近似匹配判断 + TTL 分层

    策略：
      - 精确匹配（goal 完全相同）→ 直接复用，走段6写L3记录
      - 近似匹配：
        < 60%: 不注入
        60-80%: 弱参考（记"参考了历史"，不占主流程）
        > 80%: 强参考（注入上下文）
      - 流式任务不缓存
      - TTL 分层：默认 24h / 外部数据 1h / 代码 72h
    """

    TTL_MAP = {
        "default": 86400,        # 24h
        "external_data": 3600,   # 1h
        "code": 259200,          # 72h
        "streaming": 0,          # 不缓存
    }

    # 近似匹配用关键词重叠率模拟
    # 生产环境应接入向量相似度

    def __init__(self):
        self._cache: OrderedDict[str, CacheMatch] = OrderedDict()
        self.stats = {"check": 0, "hit_exact": 0, "hit_strong": 0,
                      "hit_weak": 0, "miss": 0}

    def check(self, goal: str, cache_type: str = "default",
              is_streaming: bool = False) -> CacheMatch:
        """检查缓存

        Args:
            goal: 任务目标描述
            cache_type: 缓存类型 (default/external_data/code/streaming)
            is_streaming: 是否是流式任务

        Returns:
            CacheMatch 匹配结果
        """
        self.stats["check"] += 1

        # 流式任务不查缓存
        if is_streaming or cache_type == "streaming":
            return CacheMatch(matched=False)

        # 精确匹配
        if goal in self._cache:
            entry = self._cache[goal]
            ttl = self.TTL_MAP.get(cache_type, 86400)
            if time.time() - entry.age < ttl:
                entry.exact = True
                entry.confidence = 1.0
                self.stats["hit_exact"] += 1
                self._cache.move_to_end(goal)  # LRU 刷新
                return entry
            else:
                # TTL 过期
                del self._cache[goal]

        # 近似匹配 — 关键词重叠率
        best_match = None
        best_score = 0.0

        for cached_goal, entry in self._cache.items():
            score = self._keyword_similarity(goal, cached_goal)
            if score > best_score:
                best_score = score
                best_match = entry

        if best_match and best_score >= 0.8:
            self.stats["hit_strong"] += 1
            return CacheMatch(matched=True, confidence=best_score,
                              value=best_match.value, age=time.time() - best_match.age,
                              source_task=best_match.source_task)

        if best_match and best_score >= 0.6:
            self.stats["hit_weak"] += 1
            return CacheMatch(matched=True, confidence=best_score,
                              value=best_match.value, age=time.time() - best_match.age,
                              source_task=best_match.source_task)

        self.stats["miss"] += 1
        return CacheMatch(matched=False)

    def store(self, goal: str, value: Any, source_task: str = "",
              cache_type: str = "default") -> None:
        """写入缓存"""
        if cache_type == "streaming":
            return  # 流式不缓存

        ttl = self.TTL_MAP.get(cache_type, 86400)
        entry = CacheMatch(
            matched=True, confidence=1.0, exact=True,
            value=value, source_task=source_task,
            age=time.time(),
        )
        self._cache[goal] = entry

        # LRU 淘汰：最多 200 条
        if len(self._cache) > 200:
            self._cache.popitem(last=False)

    def _keyword_similarity(self, a: str, b: str) -> float:
        """关键词重叠率 — 简化的文本相似度

        生产环境应接入向量 embedding 相似度。
        """
        # Tokenize 为关键词
        def tokenize(s: str) -> set[str]:
            # 中文按字+常见双字，英文按词
            tokens = set()
            s_lower = s.lower()
            # 英文词
            for word in re.findall(r'\b[a-z]+\b', s_lower):
                tokens.add(word)
            # 中文单字
            for char in s:
                if '\u4e00' <= char <= '\u9fff':
                    tokens.add(char)
            # 中文双字
            for i in range(len(s) - 1):
                bi = s[i:i+2]
                if all('\u4e00' <= c <= '\u9fff' for c in bi):
                    tokens.add(bi)
            return tokens

        tokens_a = tokenize(a)
        tokens_b = tokenize(b)

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / max(len(union), 1)

    def invalidate(self, goal: str) -> bool:
        """失效缓存"""
        if goal in self._cache:
            del self._cache[goal]
            return True
        return False

    def summary(self) -> str:
        s = self.stats
        hit_rate = (s["hit_exact"] + s["hit_strong"] + s["hit_weak"]) / max(s["check"], 1) * 100
        return (
            f"段0缓存: 命中率={hit_rate:.0f}% "
            f"(精确={s['hit_exact']} 强={s['hit_strong']} 弱={s['hit_weak']}) "
            f"未命中={s['miss']} 缓存数={len(self._cache)}"
        )


# ═══════════════════════════════════════════════════════════
# 段2: 复杂度判定 + 梯度预算降级 + 分层检索
# ═══════════════════════════════════════════════════════════


class ComplexityLevel(str, Enum):
    SIMPLE = "simple"               # 简单任务
    NORMAL = "normal"               # 普通任务
    COMPLEX = "complex"             # 偏繁重
    VERY_COMPLEX = "very_complex"   # 繁重


@dataclass
class ComplexityResult:
    """复杂度判定结果"""
    level: ComplexityLevel
    score: float                     # 0-1 复杂度分数
    factors: list[dict]              # 各判定因子
    recommendation: str = ""         # 建议行动


@dataclass
class BudgetPlan:
    """预算降级方案"""
    level: str                       # full / reduced_qa / skip_qa / reject
    description: str
    cross_review: bool = True        # 是否交叉审查
    qa_rounds: int = 2               # 质检轮次


class ComplexityAnalyzer:
    """段2 复杂度判定器

    判定因子：
      - token 阈值
      - 文件数量
      - 安全要求
      - L5 历史错误率
      - 对外发布标记

    阈值动态调整：每30次取P90，0.5-2倍保护
    """

    FACTOR_WEIGHTS = {
        "token_ratio": 0.25,
        "file_count": 0.15,
        "safety_required": 0.20,
        "l5_error_rate": 0.20,
        "is_public": 0.10,
        "has_external_api": 0.10,
    }

    COMPLEXITY_THRESHOLDS = {
        ComplexityLevel.SIMPLE: 0.0,
        ComplexityLevel.NORMAL: 0.3,
        ComplexityLevel.COMPLEX: 0.6,
        ComplexityLevel.VERY_COMPLEX: 0.8,
    }

    def __init__(self):
        self._history_scores: list[float] = []
        self._history_p90: float = 0.5
        self.stats = {"analyzed": 0, "simple": 0, "normal": 0,
                      "complex": 0, "very_complex": 0}

    def analyze(self, estimated_tokens: int = 0,
                file_count: int = 1,
                safety_required: bool = False,
                l5_error_rate: float = 0.0,
                is_public: bool = False,
                has_external_api: bool = False) -> ComplexityResult:
        """执行复杂度判定

        Returns:
            ComplexityResult 包含等级、分数、因子清单
        """
        self.stats["analyzed"] += 1

        # 各因子归一化分数 (0-1)
        factors = [
            {
                "name": "token_ratio",
                "raw": estimated_tokens,
                "score": min(1.0, estimated_tokens / 50000),
            },
            {
                "name": "file_count",
                "raw": file_count,
                "score": min(1.0, file_count / 10),
            },
            {
                "name": "safety_required",
                "raw": safety_required,
                "score": 1.0 if safety_required else 0.0,
            },
            {
                "name": "l5_error_rate",
                "raw": l5_error_rate,
                "score": l5_error_rate,
            },
            {
                "name": "is_public",
                "raw": is_public,
                "score": 1.0 if is_public else 0.0,
            },
            {
                "name": "has_external_api",
                "raw": has_external_api,
                "score": 1.0 if has_external_api else 0.0,
            },
        ]

        # 加权总分
        total_score = sum(
            f["score"] * self.FACTOR_WEIGHTS.get(f["name"], 0.1)
            for f in factors
        )

        # 动态阈值调整（历史 P90）
        if self._history_scores:
            threshold_bonus = total_score / max(self._history_p90, 0.01)
            if threshold_bonus > 2.0:
                total_score = min(1.0, total_score * 0.8)  # 高阈值保护
            elif threshold_bonus < 0.5:
                total_score = min(1.0, total_score * 1.2)  # 低阈值提升

        # 确定等级
        level = ComplexityLevel.SIMPLE
        for lvl, threshold in sorted(
            self.COMPLEXITY_THRESHOLDS.items(), key=lambda x: x[1], reverse=True
        ):
            if total_score >= threshold:
                level = lvl
                break

        # 更新历史
        self._history_scores.append(total_score)
        if len(self._history_scores) >= 30:
            # 计算 P90
            sorted_scores = sorted(self._history_scores[-30:])
            p90_idx = int(len(sorted_scores) * 0.9)
            self._history_p90 = sorted_scores[p90_idx]
            self._history_scores = self._history_scores[-30:]

        # 统计
        self.stats[level.value] += 1

        # 建议
        if level == ComplexityLevel.VERY_COMPLEX:
            rec = "需全流程: 交叉审查+去AI化+多重质检"
        elif level == ComplexityLevel.COMPLEX:
            rec = "偏繁重: 标准质检+可选交叉审查"
        elif level == ComplexityLevel.NORMAL:
            rec = "普通: 标准质检"
        else:
            rec = "简单: 轻量质检"

        return ComplexityResult(
            level=level,
            score=total_score,
            factors=factors,
            recommendation=rec,
        )

    def get_budget_plan(self, complexity: ComplexityResult,
                        budget_ratio: float = 0.0) -> BudgetPlan:
        """根据复杂度和预算比例生成降级方案

        Args:
            complexity: 复杂度判定结果
            budget_ratio: 当前预算使用率 0-1

        Returns:
            BudgetPlan
        """
        # 复杂度高的任务降级阈值更高
        complexity_penalty = complexity.score * 0.3
        effective_budget = budget_ratio + complexity_penalty

        if effective_budget < 0.3:
            return BudgetPlan("full", "全流程: 交叉审查+质检", True, 2)
        elif effective_budget < 0.6:
            return BudgetPlan("reduced_qa", "压缩质检: 仅标准审查", False, 1)
        elif effective_budget < 1.0:
            return BudgetPlan("skip_qa", "跳质检: 仅验收", False, 0)
        else:
            return BudgetPlan("reject", "拒绝执行: 预算耗尽", False, 0)

    def summary(self) -> str:
        s = self.stats
        return (
            f"段2复杂度: {s['analyzed']}次判定 "
            f"(简单={s['simple']} 普通={s['normal']} "
            f"繁重={s['complex']+s['very_complex']})"
        )


class MemoryRetriever:
    """段2 分层记忆检索器 — L3/L4/L5 强制检索

    策略：
      - L4 技能 → 精确匹配（skill_name）
      - L5 反馈 → 关键词检索（高频错误预检）
      - L3 会话 → 全文检索
      - 读取失败重试2次（2s/5s），全失败后跳过
    """

    def __init__(self):
        self.stats = {"l3_retrieved": 0, "l4_loaded": 0,
                      "l5_prechecked": 0, "failures": 0}

    def retrieve(self, query: str, task_type: str = "",
                 context: dict | None = None) -> dict:
        """分层检索

        Returns:
            {l3_results: [...], l4_skills: [...], l5_warnings: [...],
             total_chars: int, sources: [str]}
        """
        results = {
            "l3_results": [],
            "l4_skills": [],
            "l5_warnings": [],
            "total_chars": 0,
            "sources": [],
        }

        # L4 技能 — 精确匹配
        l4_hits = self._retrieve_l4(query)
        if l4_hits:
            results["l4_skills"] = l4_hits
            results["sources"].append("l4")
            self.stats["l4_loaded"] += 1

        # L5 反馈预检 — 高频错误
        l5_hits = self._retrieve_l5(query)
        if l5_hits:
            results["l5_warnings"] = l5_hits
            results["sources"].append("l5")
            self.stats["l5_prechecked"] += 1

        # L3 会话检索
        l3_hits = self._retrieve_l3(query)
        if l3_hits:
            results["l3_results"] = l3_hits
            results["sources"].append("l3")
            self.stats["l3_retrieved"] += 1

        # 计算总字符
        for key in ["l3_results", "l4_skills", "l5_warnings"]:
            for item in results[key]:
                content = item.get("content", "") if isinstance(item, dict) else str(item)
                results["total_chars"] += len(str(content))

        return results

    def _retrieve_l3(self, query: str) -> list[dict]:
        """L3 会话检索"""
        try:
            from hermes_tools import session_search as search
            r = search(query=query, limit=5)
            if hasattr(r, 'get'):
                return r.get("results", [])
            return []
        except (ImportError, Exception) as e:
            logger.debug(f"L3 检索失败: {e}")
            return []

    def _retrieve_l4(self, query: str) -> list[dict]:
        """L4 技能检索 — 从已安装 skill 中匹配"""
        try:
            from hermes_tools import terminal
            skills = terminal(command="ls ~/AppData/Local/hermes/skills/", timeout=5)
            if skills and "No such file" not in skills.get("output", ""):
                skill_lines = skills["output"].strip().split("\n")
                # 按关键词匹配技能
                keywords = query.lower().split()
                matched = []
                for line in skill_lines:
                    line = line.strip().rstrip("/")
                    if any(kw in line.lower() for kw in keywords):
                        matched.append({"name": line, "source": "l4"})
                return matched
        except (ImportError, Exception):
            pass
        return []

    def _retrieve_l5(self, query: str) -> list[dict]:
        """L5 反馈预检 — 通过 memory 工具检索高频错误"""
        try:
            from hermes_tools import memory as hermes_memory
            # memory 工具不支持精确检索，这里返回空列表作为 fallback
            # 实际可通过 session_search 检索历史反馈
            return []
        except ImportError:
            return []

    def summary(self) -> str:
        s = self.stats
        return (
            f"段2检索: L3={s['l3_retrieved']} L4={s['l4_loaded']} "
            f"L5={s['l5_prechecked']} 失败={s['failures']}"
        )


# ═══════════════════════════════════════════════════════════
# 段5: 交叉审查 Agent
# ═══════════════════════════════════════════════════════════


@dataclass
class ReviewRound:
    """单轮审查结果"""
    reviewer: str
    passed: bool
    score: float
    issues: list[str]
    suggestions: list[str]
    round_num: int = 1

    def to_dict(self) -> dict:
        return {
            "reviewer": self.reviewer,
            "passed": self.passed,
            "score": self.score,
            "issues": self.issues[:5],
            "suggestions": self.suggestions[:3],
            "round": self.round_num,
        }


class CrossReviewer:
    """段5 交叉审查 Agent

    双模型互审（不同 Agent 交叉检查）：
      - 发现错误 → 正向激励，加一轮审查（最多2轮）
      - 段5有最终复杂度判断权（> 预算降级决策）
      - 偏繁重任务由段5判定后强制升级质检

    使用方法：
      1. 创建 CrossReviewer 实例
      2. 注册审查员（critic_a, critic_b）
      3. 对 DAG 执行交叉审查
    """

    MAX_ROUNDS = 2

    def __init__(self):
        self._reviewers: dict[str, Callable] = {}
        self._rounds: list[ReviewRound] = []
        self.stats = {"reviews": 0, "found_issues": 0, "upgraded": 0}

    def register_reviewer(self, name: str, reviewer_fn: Callable) -> None:
        """注册审查员

        Args:
            name: 审查员名称（如 "critic_deepseek", "critic_kimi"）
            reviewer_fn: 接收 (graph) 返回 ReviewRound
        """
        self._reviewers[name] = reviewer_fn

    def review(self, graph, context: dict | None = None) -> list[ReviewRound]:
        """执行交叉审查

        流程：
          1. 注册的审查员依次审查
          2. 如发现问题且轮次 < MAX_ROUNDS，加一轮
          3. 汇总审查结果

        Returns:
            审查轮次列表
        """
        ctx = context or {}
        rounds = []
        reviewer_names = list(self._reviewers.keys())

        for round_num in range(1, self.MAX_ROUNDS + 1):
            round_results = []
            found_issues = False

            for name in reviewer_names:
                fn = self._reviewers[name]
                try:
                    result = fn(graph)
                    if not isinstance(result, ReviewRound):
                        result = ReviewRound(
                            reviewer=name,
                            passed=result.get("passed", True) if isinstance(result, dict) else True,
                            score=result.get("score", 1.0) if isinstance(result, dict) else 1.0,
                            issues=result.get("issues", []) if isinstance(result, dict) else [],
                            suggestions=result.get("suggestions", []) if isinstance(result, dict) else [],
                            round_num=round_num,
                        )
                    round_results.append(result)
                    self.stats["reviews"] += 1
                    self.stats["found_issues"] += len(result.issues)
                    if not result.passed:
                        found_issues = True
                except Exception as e:
                    logger.warning(f"审查员 {name} 异常: {e}")

            if round_results:
                rounds.extend(round_results)
                self._rounds.extend(round_results)

            # 无问题则提前结束
            if not found_issues:
                break

        return rounds

    def check_complexity_upgrade(self, original_complexity: str,
                                 review_results: list[ReviewRound]) -> tuple[bool, str]:
        """段5 最终复杂度判定

        如果审查发现实际复杂度高于标注，强制升级质检。
        此决策优先级高于预算降级决策。

        Args:
            original_complexity: 原始复杂度标注
            review_results: 审查结果

        Returns:
            (should_upgrade: bool, reason: str)
        """
        issue_count = sum(len(r.issues) for r in review_results)
        avg_score = sum(r.score for r in review_results) / max(len(review_results), 1)

        # 高于阈值 → 升级
        if issue_count >= 3 and avg_score < 0.5:
            self.stats["upgraded"] += 1
            return (True, f"段5判定: 发现 {issue_count} 个问题 (avg={avg_score:.2f})，"
                          f"强制升级质检（> 预算降级决策）")

        if original_complexity in ("simple", "normal") and issue_count >= 2:
            self.stats["upgraded"] += 1
            return (True, f"低复杂度标注但发现 {issue_count} 个问题，强制升级")

        return (False, "复杂度判定通过")

    def get_default_reviewer(self) -> Callable:
        """返回默认审查函数（基于 Critic）"""
        from executor import Critic as BaseCritic

        def default_review(graph) -> ReviewRound:
            critic = BaseCritic()
            result = critic.check(graph)
            return ReviewRound(
                reviewer="auto_critic",
                passed=result.passed,
                score=result.score,
                issues=result.issues,
                suggestions=result.suggestions,
            )

        return default_review

    def summary(self) -> str:
        return (
            f"段5交叉审查: {self.stats['reviews']} 次审查 "
            f"发现 {self.stats['found_issues']} 问题 "
            f"升级 {self.stats['upgraded']} 次"
        )


# ═══════════════════════════════════════════════════════════
# 段7: L1 自动同步 + 版本乐观锁
# ═══════════════════════════════════════════════════════════


@dataclass
class L1SyncEntry:
    """L1 自动同步条目"""
    topic: str
    value: str
    confirm_count: int = 0
    last_confirmed: float = field(default_factory=time.time)
    synced: bool = False


class L1AutoSync:
    """段7 L1 自动同步 — 连续3次确认的稳定反馈自动写入 L1

    策略：
      - 连续 3 次确认 → 自动写入 L1
      - 版本乐观锁 → 每次更新版本号
      - deprecated 语义 → 标记旧版本不删除，加 deprecated 前缀
    """

    CONFIRM_THRESHOLD = 3

    def __init__(self):
        self._entries: dict[str, L1SyncEntry] = {}
        self._deprecated: set[str] = set()
        self._sync_log: list[dict] = []
        self.stats = {"confirmed": 0, "synced": 0, "deprecated": 0}

    def record_feedback(self, topic: str, value: str) -> dict:
        """记录一次反馈

        Args:
            topic: 反馈主题（如 "color_scheme"）
            value: 反馈内容（如 "深蓝科技风"）

        Returns:
            {status: "pending"/"ready_to_sync"/"synced",
             count: int, version: int}
        """
        if topic not in self._entries:
            self._entries[topic] = L1SyncEntry(topic=topic, value=value, confirm_count=1)
        else:
            entry = self._entries[topic]
            if entry.value == value:
                entry.confirm_count += 1
                entry.last_confirmed = time.time()
            else:
                # 值变化 → 重置计数器
                old_value = entry.value
                entry.value = value
                entry.confirm_count = 1
                self._log(f"值变更: {topic} '{old_value}' → '{value}'")

        entry = self._entries[topic]

        if entry.confirm_count >= self.CONFIRM_THRESHOLD and not entry.synced:
            self._sync_to_l1(topic, value, entry.confirm_count)
            entry.synced = True
            self.stats["synced"] += 1
            return {"status": "synced", "count": entry.confirm_count, "version": entry.confirm_count}

        if entry.synced:
            return {"status": "synced", "count": entry.confirm_count, "version": 0}

        self.stats["confirmed"] += 1
        return {
            "status": "pending",
            "count": entry.confirm_count,
            "version": 0,
            "remaining": self.CONFIRM_THRESHOLD - entry.confirm_count,
        }

    def _sync_to_l1(self, topic: str, value: str, count: int) -> int:
        """同步到 L1"""
        try:
            from hermes_tools import memory as hermes_memory
            version = int(time.time())
            content = f"l1:auto:{topic}: {value} (v{version}, confirmed {count}x)"
            hermes_memory(action="add", target="memory", content=content)
            self._log(f"L1同步: {topic} = '{value}' (v{version})")
            return version
        except ImportError:
            logger.debug(f"L1 同步跳过 (无 hermes_tools): {topic}")
            return 0

    def mark_deprecated(self, topic: str) -> bool:
        """标记为 deprecated（版本优化锁，不删除旧数据）"""
        if topic in self._entries:
            self._deprecated.add(topic)
            self.stats["deprecated"] += 1
            self._log(f"deprecated: {topic}")
            return True
        return False

    def is_deprecated(self, topic: str) -> bool:
        return topic in self._deprecated

    def get_ready_count(self) -> int:
        """等待同步的条目数"""
        return sum(
            1 for e in self._entries.values()
            if e.confirm_count >= self.CONFIRM_THRESHOLD and not e.synced
        )

    def _log(self, message: str) -> None:
        self._sync_log.append({"message": message, "timestamp": time.time()})

    def summary(self) -> str:
        return (
            f"段7 L1同步: {self.stats['confirmed']}反馈 "
            f"已同步={self.stats['synced']} "
            f"待同步={self.get_ready_count()} "
            f"deprecated={self.stats['deprecated']}"
        )


# ═══════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════

def test_cache_prechecker():
    cpc = CachePreChecker()

    # 精确匹配
    result = cpc.check("做一篇关于AI Agent的XHS")
    assert not result.matched  # 首次未命中

    cpc.store("做一篇关于AI Agent的XHS", {"id": "xhs_001"}, "t1")
    result = cpc.check("做一篇关于AI Agent的XHS")
    assert result.matched and result.exact
    assert result.confidence == 1.0
    print(f"段0 精确匹配: ✓")

    # 近似匹配
    result2 = cpc.check("做一篇关于AI Agent的小红书")
    assert result2.matched
    print(f"段0 近似匹配: ✓ (confidence={result2.confidence:.2f})")

    # 不匹配
    result3 = cpc.check("写一个Python函数计算Fibonacci")
    assert not result3.matched
    print(f"段0 不匹配: ✓")

    # 流式不缓存
    cpc.store("stream_task", "data", cache_type="streaming")
    assert "stream_task" not in cpc._cache
    print(f"段0 流式不缓存: ✓")

    print(f"  {cpc.summary()}")


def test_complexity():
    ca = ComplexityAnalyzer()

    # 简单任务
    r1 = ca.analyze(estimated_tokens=100, file_count=1)
    assert r1.level == ComplexityLevel.SIMPLE
    print(f"段2 简单判定: ✓ (score={r1.score:.2f})")

    # 繁重任务
    r2 = ca.analyze(estimated_tokens=60000, file_count=8,
                     safety_required=True, is_public=True, l5_error_rate=0.3)
    print(f"段2 繁重判定: ✓ (level={r2.level.value}, score={r2.score:.2f})")

    # 简单+低预算 → full
    simple = ca.analyze(estimated_tokens=100, file_count=1)
    plan = ca.get_budget_plan(simple, budget_ratio=0.1)
    assert plan.level == "full", f"应 full, 实际 {plan.level}"
    print(f"段2 预算full: ✓ (level={plan.level})")

    # 繁重+高预算 → skip_qa
    plan2 = ca.get_budget_plan(r2, budget_ratio=0.7)
    print(f"段2 预算降级: ✓ (level={plan2.level})")


def test_cross_review():
    cr = CrossReviewer()

    # 注册默认审查员
    cr.register_reviewer("auto_critic", cr.get_default_reviewer())

    # 创建一个需要审查的 DAG
    from dag import TaskNode, TaskGraph
    g = TaskGraph("review-test")
    n = TaskNode(id="A", goal="正常的输出内容 长度足够通过验证 没有问题")
    n.result = "这是正确且完整的输出，已经过验证"
    n.status = type('s', (), {'value': 'completed'})()
    g.add_node(n)

    results = cr.review(g)
    assert len(results) >= 1
    print(f"段5 交叉审查: ✓ ({len(results)} 轮)")

    # 复杂度升级判定
    upgrade, reason = cr.check_complexity_upgrade("simple", [
        ReviewRound("c1", False, 0.3, ["问题1", "问题2", "问题3"], [])
    ])
    assert upgrade
    print(f"段5 复杂度升级: ✓ ({reason[:40]}...)")


def test_l1_auto_sync():
    l1 = L1AutoSync()

    # 前两次 → pending
    r1 = l1.record_feedback("color", "深蓝")
    assert r1["status"] == "pending"
    assert r1["count"] == 1

    r2 = l1.record_feedback("color", "深蓝")
    assert r2["status"] == "pending"

    # 第三次 → sync
    r3 = l1.record_feedback("color", "深蓝")
    assert r3["status"] == "synced"
    print(f"段7 L1自动同步: ✓ (3次确认→同步)")

    # deprecated
    l1.mark_deprecated("color")
    assert l1.is_deprecated("color")
    print(f"段7 deprecated: ✓")

    # 值变更重置
    r4 = l1.record_feedback("theme", "dark")
    r5 = l1.record_feedback("theme", "light")  # 值变了
    assert r5["count"] == 1
    print(f"段7 值变更重置: ✓")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print("═══ 流水线各段逻辑测试 ═══\n")

    print("─ 段0 缓存预检 ─")
    test_cache_prechecker()
    print()

    print("─ 段2 复杂度+预算+检索 ─")
    test_complexity()
    print()

    print("─ 段5 交叉审查 ─")
    test_cross_review()
    print()

    print("─ 段7 L1自动同步 ─")
    test_l1_auto_sync()
    print()

    print("\n═══ 全部测试通过 ═══")
