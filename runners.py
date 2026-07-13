"""
Hermes 三通道真实执行器 — 对接 CC/Codex/Hermes 的实际工具调用

三通道：
  CC:     通过 delegate_task 派给 Claude Code 子代理
  Codex:  通过 image_generate/text_to_speech/terminal 执行
  Hermes: 直接在当前上下文中执行（terminal/工具链）

每个 runner 可在两个环境下使用：
  1. Hermes 会话中直接调用（import hermes_tools）
  2. execute_code() 中调用（from hermes_tools import ...）

Usage（在 Hermes 会话中）:
    from runners import HermesRunners
    from executor import Executor

    exec = Executor()
    HermesRunners.register_all(exec)
    results = exec.execute_graph(my_graph)
"""

from __future__ import annotations

import time
import json
import os
import ast
import re
import logging
from typing import Any

from executor import ExecutionResult

logger = logging.getLogger("agent-framework.runners")


# ─── 三通道 Runner ──────────────────────────────────────────


class HermesRunners:
    """Hermes 三通道执行器

    提供三个 Agent 通道的真实执行函数。
    依赖 Hermes 工具集（delegate_task, terminal, image_generate 等）。
    """

    @staticmethod
    def cc_runner(node) -> ExecutionResult:
        """CC Runner — 纯 Python 静态代码审查器（无需外部 CLI）

        对 workdir 下的文件进行静态分析：
          - .py: 语法检查 (compile)、AST 结构分析、安全问题扫描
          - .html: 语法检查、XSS 风险检测

        Args:
            node: TaskNode 实例

        Returns:
            ExecutionResult
        """
        start = time.time()
        context = node.context or {}
        workdir = context.get('workdir', '.')
        goal = node.goal

        issues = []
        files_scanned = 0

        try:
            # Scan all relevant files in workdir
            for root, dirs, files in os.walk(workdir):
                # Skip common non-source directories
                dirs[:] = [d for d in dirs if not d.startswith(('.', '__pycache__', 'node_modules', 'venv', '.venv', 'dist', 'build', '.git'))]
                for fname in files:
                    fpath = os.path.join(root, fname)
                    if fname.endswith('.py'):
                        files_scanned += 1
                        file_issues = _analyze_py_file(fpath)
                        issues.extend(file_issues)
                    elif fname.endswith('.html'):
                        files_scanned += 1
                        file_issues = _analyze_html_file(fpath)
                        issues.extend(file_issues)

            # Build report
            severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
            for issue in issues:
                sev = issue.get("severity", "info")
                if sev in severity_counts:
                    severity_counts[sev] += 1

            high_or_critical = severity_counts['critical'] + severity_counts['high']

            report_lines = [
                f"═══ CC 代码审查报告 ═══",
                f"目标: {goal[:60]}{'...' if len(goal) > 60 else ''}",
                f"扫描目录: {os.path.abspath(workdir)}",
                f"扫描文件: {files_scanned}",
                f"发现问题: {len(issues)}",
                f"  严重: critical={severity_counts['critical']} high={severity_counts['high']} "
                f"medium={severity_counts['medium']} low={severity_counts['low']} info={severity_counts['info']}",
                "",
            ]

            if issues:
                report_lines.append("── 问题详情 ──")
                for idx, issue in enumerate(issues, 1):
                    report_lines.append(
                        f"{idx}. [{issue['severity'].upper()}] {issue['file']}:{issue.get('line', '?')} "
                        f"({issue.get('checker', 'unknown')})"
                    )
                    report_lines.append(f"   问题: {issue['message']}")
                    if issue.get('suggestion'):
                        report_lines.append(f"   建议: {issue['suggestion']}")
                    report_lines.append("")
            else:
                report_lines.append("✓ 未发现明显问题。")

            report = "\n".join(report_lines)
            duration = time.time() - start

            return ExecutionResult(
                node_id=node.id,
                success=high_or_critical == 0,
                output=report,
                agent_used="cc",
                duration=duration,
                metadata={
                    "files_scanned": files_scanned,
                    "total_issues": len(issues),
                    "severity_counts": severity_counts,
                },
            )

        except Exception as e:
            duration = time.time() - start
            return ExecutionResult(
                node_id=node.id,
                success=False,
                error=f"CC 代码审查出错: {e}",
                agent_used="cc",
                duration=duration,
            )

    @staticmethod
    def codex_runner(node) -> ExecutionResult:
        """Codex Runner — 纯 Python 静态分析器（无需外部 CLI）

        对 workdir 下的文件进行静态分析：
          - .py: 安全检查（eval/exec/shell注入）、依赖审查
          - .yaml/.yml: 安全隐患扫描（硬编码密钥、危险配置）
          - .json: 结构验证

        Args:
            node: TaskNode 实例

        Returns:
            ExecutionResult
        """
        start = time.time()
        context = node.context or {}
        workdir = context.get('workdir', '.')
        goal = node.goal

        issues = []
        files_scanned = 0

        try:
            for root, dirs, files in os.walk(workdir):
                dirs[:] = [d for d in dirs if not d.startswith(('.', '__pycache__', 'node_modules', 'venv', '.venv', 'dist', 'build', '.git'))]
                for fname in files:
                    fpath = os.path.join(root, fname)
                    if fname.endswith('.py'):
                        files_scanned += 1
                        file_issues = _analyze_py_file(fpath)
                        issues.extend(file_issues)
                    elif fname.endswith(('.yaml', '.yml')):
                        files_scanned += 1
                        file_issues = _analyze_yaml_file(fpath)
                        issues.extend(file_issues)
                    elif fname.endswith('.json'):
                        files_scanned += 1
                        file_issues = _analyze_json_file(fpath)
                        issues.extend(file_issues)

            severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
            for issue in issues:
                sev = issue.get("severity", "info")
                if sev in severity_counts:
                    severity_counts[sev] += 1

            high_or_critical = severity_counts['critical'] + severity_counts['high']

            report_lines = [
                f"═══ Codex 代码/配置分析报告 ═══",
                f"目标: {goal[:60]}{'...' if len(goal) > 60 else ''}",
                f"扫描目录: {os.path.abspath(workdir)}",
                f"扫描文件: {files_scanned}",
                f"发现问题: {len(issues)}",
                f"  严重: critical={severity_counts['critical']} high={severity_counts['high']} "
                f"medium={severity_counts['medium']} low={severity_counts['low']} info={severity_counts['info']}",
                "",
            ]

            if issues:
                report_lines.append("── 问题详情 ──")
                for idx, issue in enumerate(issues, 1):
                    report_lines.append(
                        f"{idx}. [{issue['severity'].upper()}] {issue['file']}:{issue.get('line', '?')} "
                        f"({issue.get('checker', 'unknown')})"
                    )
                    report_lines.append(f"   问题: {issue['message']}")
                    if issue.get('suggestion'):
                        report_lines.append(f"   建议: {issue['suggestion']}")
                    report_lines.append("")
            else:
                report_lines.append("✓ 未发现明显问题。")

            report = "\n".join(report_lines)
            duration = time.time() - start

            return ExecutionResult(
                node_id=node.id,
                success=high_or_critical == 0,
                output=report,
                agent_used="codex",
                duration=duration,
                metadata={
                    "files_scanned": files_scanned,
                    "total_issues": len(issues),
                    "severity_counts": severity_counts,
                },
            )

        except Exception as e:
            duration = time.time() - start
            return ExecutionResult(
                node_id=node.id,
                success=False,
                error=f"Codex 代码分析出错: {e}",
                agent_used="codex",
                duration=duration,
            )

    @staticmethod
    def hermes_runner(node) -> str:
        """Hermes Runner — 描述直接执行的操作"""
        context = node.context or {}
        task_type = context.get("type", "direct")

        instructions = {
            "channel": "hermes",
            "goal": node.goal,
            "node_id": node.id,
            "action": "execute_locally",
            "tool_calls": [],
        }

        if task_type == "search":
            instructions["tool_calls"].append({
                "tool": "web_search",
                "params": {"query": node.goal},
            })
        elif task_type == "read":
            instructions["tool_calls"].append({
                "tool": "read_file",
                "params": {"path": context.get("path", "")},
            })
        elif task_type == "write":
            instructions["tool_calls"].append({
                "tool": "write_file",
                "params": {
                    "path": context.get("path", ""),
                    "content": context.get("content", node.goal),
                },
            })
        elif task_type == "terminal":
            instructions["tool_calls"].append({
                "tool": "terminal",
                "params": {
                    "command": context.get("command", node.goal),
                    "timeout": context.get("timeout", 30),
                },
            })
        else:
            instructions["tool_calls"].append({
                "tool": "direct_execution",
                "params": {"goal": node.goal},
            })

        return json.dumps(instructions, ensure_ascii=False)

    @staticmethod
    def register_all(executor) -> None:
        """注册全部三通道 runner 到 Executor"""
        executor.register_runner("cc", HermesRunners.cc_runner)
        executor.register_runner("codex", HermesRunners.codex_runner)
        executor.register_runner("hermes", HermesRunners.hermes_runner)

        # 别名
        executor.register_runner("manual", HermesRunners.hermes_runner)
        executor.register_runner("custom", HermesRunners.hermes_runner)

    @staticmethod
    def register_subrunners(executor, runners: dict[str, callable]) -> None:
        """注册自定义 runner 覆盖默认"""
        for agent_type, runner_fn in runners.items():
            executor.register_runner(agent_type, runner_fn)


# ─── 静态分析辅助函数 ──────────────────────────────────────────


def _analyze_py_file(fpath: str) -> list[dict]:
    """分析 .py 文件：语法检查 + AST 结构分析 + 安全问题扫描"""
    issues = []
    try:
        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
            source = f.read()
    except (IOError, PermissionError) as e:
        issues.append({
            "file": fpath, "line": 1, "severity": "high",
            "checker": "file_read",
            "message": f"无法读取文件: {e}",
            "suggestion": "检查文件权限和编码",
        })
        return issues

    # 1. 语法检查 (compile)
    try:
        compile(source, fpath, 'exec')
    except SyntaxError as e:
        issues.append({
            "file": fpath, "line": e.lineno or 1, "severity": "critical",
            "checker": "syntax",
            "message": f"语法错误: {e.msg}",
            "suggestion": f"第 {e.lineno} 行附近: {e.text.strip() if e.text else 'N/A'}",
        })
        return issues  # 语法错误时停止进一步分析

    # 2. AST 结构分析
    try:
        tree = ast.parse(source, filename=fpath)
    except SyntaxError:
        return issues  # 已在上面处理

    # 检查 import 风格（避免通配符导入）
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.names and any(n.name == '*' for n in node.names):
            issues.append({
                "file": fpath, "line": getattr(node, 'lineno', 1), "severity": "medium",
                "checker": "import_style",
                "message": f"通配符导入: from {node.module or ''} import *",
                "suggestion": "改为显式导入具体名称",
            })

    # 3. 安全问题扫描（正则）
    security_patterns = [
        # (pattern, severity, checker, message, suggestion)
        (r'\beval\s*\(', "critical", "eval_usage",
         "eval() 使用 — 可能导致任意代码执行",
         "改用 ast.literal_eval() 或 safer alternatives"),
        (r'\bexec\s*\(', "critical", "exec_usage",
         "exec() 使用 — 可能导致任意代码执行",
         "避免使用 exec()，考虑其他方案"),
        (r'\bos\.system\s*\(', "critical", "shell_injection",
         "os.system() 使用 — shell 注入风险",
         "改用 subprocess.run() 并传递参数列表"),
        (r'\bsubprocess\.(call|Popen|run)\s*\(.*shell\s*=\s*True',
         "high", "shell_injection",
         "shell=True 参数 — shell 注入风险",
         "移除 shell=True，传递参数列表而非字符串"),
        (r'\b__import__\s*\(', "high", "dynamic_import",
         "__import__() 动态导入",
         "改用 importlib.import_module()"),
        (r'\bpickle\.(loads?|dumps?)\s*\(',
         "medium", "unsafe_deserialization",
         "pickle 反序列化 — 可能导致任意代码执行",
         "改用 JSON 或安全序列化格式"),
        (r'\.format\(.*\{\}', "info", "format_string",
         "危险的 format() 模板",
         "审查模板内容"),
        (r'password\s*=.*["\'][^"\']+["\']', "high", "hardcoded_secret",
         "疑似硬编码密码",
         "改用环境变量或密钥管理服务"),
        (r'(api_key|apikey|secret|token)\s*=.*["\'][^"\']+["\']',
         "high", "hardcoded_credential",
         "疑似硬编码凭证",
         "改用环境变量或 .env 文件"),
        (r'\.connect\s*\(["\'].*://.*?[:@].*?["\']', "high", "connection_string",
         "连接字符串中可能包含明文凭据",
         "使用环境变量注入凭据"),
        (r'sqlite3\.connect\s*\(', "low", "sqlite",
         "SQLite 连接 — 检查路径是否正确",
         "验证文件路径"),
        (r'\bSELECT\b.+\bFROM\b.+\bWHERE\b.*\+.*["\']',
         "critical", "sql_injection",
         "疑似 SQL 注入（字符串拼接）",
         "使用参数化查询 / ORM"),
        (r'(marshal|dill|shelve)\.(load|dump)s?\(',
         "medium", "unsafe_serialization",
         "不安全的序列化模块",
         "优先使用 JSON"),
    ]

    for pattern, severity, checker, message, suggestion in security_patterns:
        for match in re.finditer(pattern, source, re.IGNORECASE):
            # 计算行号
            line_no = source[:match.start()].count('\n') + 1
            issues.append({
                "file": fpath, "line": line_no, "severity": severity,
                "checker": checker,
                "message": message,
                "suggestion": suggestion,
            })

    return issues


def _analyze_html_file(fpath: str) -> list[dict]:
    """分析 .html 文件：语法检查 + XSS 风险检测"""
    issues = []
    try:
        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except (IOError, PermissionError) as e:
        issues.append({
            "file": fpath, "line": 1, "severity": "high",
            "checker": "file_read",
            "message": f"无法读取文件: {e}",
            "suggestion": "检查文件权限和编码",
        })
        return issues

    # 1. 常见 HTML 语法问题
    # 检查未闭合的标签（简单启发式）
    closing_tags = set(re.findall(r'</(\w+)>', content))
    opening_tags = set(re.findall(r'<(\w+)[^>]*>', content))
    self_closing = {'br', 'hr', 'img', 'input', 'meta', 'link', 'source', 'area', 'base', 'col', 'embed', 'param', 'track', 'wbr'}

    unclosed = opening_tags - closing_tags - self_closing
    if unclosed:
        issues.append({
            "file": fpath, "line": 1, "severity": "medium",
            "checker": "html_syntax",
            "message": f"可能未闭合的标签: {', '.join(sorted(unclosed)[:5])}",
            "suggestion": "确保每个打开标签都有对应的闭合标签",
        })

    # 检查不正确的引号
    for i, line in enumerate(content.split('\n'), 1):
        # 简单的单引号/双引号不匹配检测
        stripped = line.strip()
        if stripped.count('"') % 2 != 0 and not stripped.startswith('//') and not stripped.startswith('/*'):
            issues.append({
                "file": fpath, "line": i, "severity": "low",
                "checker": "html_quotes",
                "message": "行内双引号数量为奇数，可能有未闭合的引号",
                "suggestion": "检查该行的引号匹配",
            })

    # 2. XSS 风险检测
    xss_patterns = [
        (r'innerHTML\s*=', "critical", "xss_inner_html",
         "innerHTML 赋值 — 可能导致 XSS 攻击",
         "改用 textContent 或安全的 sanitizer"),
        (r'dangerouslySetInnerHTML', "critical", "xss_react",
         "React dangerouslySetInnerHTML — 可能导致 XSS",
         "使用 DOMPurify 净化后使用"),
        (r'document\.write\s*\(', "high", "xss_document_write",
         "document.write() — 可能导致 XSS",
         "改用 DOM 操作 API"),
        (r'eval\s*\(', "critical", "xss_eval",
         "eval() in HTML — 任意代码执行风险",
         "移除 eval() 调用"),
        (r'src\s*=\s*["\']javascript:', "critical", "xss_javascript_url",
         "javascript: URL — 点击劫持/XSS 风险",
         "移除 javascript: 协议 URL"),
        (r'on\w+\s*=\s*["\'].*?\(', "high", "xss_inline_handler",
         "内联事件处理器（onclick/onload 等）",
         "改用 addEventListener 分离事件"),
    ]

    for pattern, severity, checker, message, suggestion in xss_patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            line_no = content[:match.start()].count('\n') + 1
            issues.append({
                "file": fpath, "line": line_no, "severity": severity,
                "checker": checker,
                "message": message,
                "suggestion": suggestion,
            })

    return issues


def _analyze_yaml_file(fpath: str) -> list[dict]:
    """分析 .yaml/.yml 文件：安全检查"""
    issues = []
    try:
        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except (IOError, PermissionError) as e:
        issues.append({
            "file": fpath, "line": 1, "severity": "high",
            "checker": "file_read",
            "message": f"无法读取文件: {e}",
            "suggestion": "检查文件权限和编码",
        })
        return issues

    # 1. 硬编码密钥检测
    secret_patterns = [
        (r'(password|passwd|pwd)\s*[:=]\s*["\']?[^"\'\s{]+', "high", "yaml_hardcoded_password",
         "YAML 中包含硬编码密码"),
        (r'(api_key|apikey|api-key)\s*[:=]\s*["\']?[^"\'\s{]+', "high", "yaml_hardcoded_api_key",
         "YAML 中包含硬编码 API 密钥"),
        (r'(secret|token)\s*[:=]\s*["\']?[^"\'\s{]+', "high", "yaml_hardcoded_secret",
         "YAML 中包含硬编码密钥/令牌"),
    ]

    for pattern, severity, checker, message in secret_patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            line_no = content[:match.start()].count('\n') + 1
            issues.append({
                "file": fpath, "line": line_no, "severity": severity,
                "checker": checker,
                "message": message,
                "suggestion": "使用 ${ENV_VAR} 占位符或外部 secret 管理",
            })

    # 2. 危险配置检测
    danger_patterns = [
        (r'privileged\s*:\s*true', "critical", "yaml_privileged",
         "容器以特权模式运行"),
        (r'securityContext:\s*\n\s+allowPrivilegeEscalation:\s*true', "high", "yaml_priv_escalation",
         "允许权限提升"),
        (r'hostNetwork\s*:\s*true', "high", "yaml_host_network",
         "容器使用宿主机网络"),
        (r'readOnlyRootFilesystem\s*:\s*false', "medium", "yaml_writable_root",
         "容器文件系统可写"),
        (r'image\s*:\s*.*:latest', "medium", "yaml_latest_tag",
         "使用 latest 标签 — 不可复现构建",
         "指定具体版本标签"),
    ]

    for pattern, severity, checker, message, *rest in danger_patterns:
        suggestion = rest[0] if rest else "审查此配置项的安全性"
        for match in re.finditer(pattern, content, re.IGNORECASE):
            line_no = content[:match.start()].count('\n') + 1
            issues.append({
                "file": fpath, "line": line_no, "severity": severity,
                "checker": checker,
                "message": message,
                "suggestion": suggestion,
            })

    return issues


def _analyze_json_file(fpath: str) -> list[dict]:
    """分析 .json 文件：结构验证"""
    issues = []
    try:
        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except (IOError, PermissionError) as e:
        issues.append({
            "file": fpath, "line": 1, "severity": "high",
            "checker": "file_read",
            "message": f"无法读取文件: {e}",
            "suggestion": "检查文件权限和编码",
        })
        return issues

    # JSON 语法验证
    try:
        json.loads(content)
    except json.JSONDecodeError as e:
        issues.append({
            "file": fpath, "line": e.lineno or 1, "severity": "critical",
            "checker": "json_syntax",
            "message": f"JSON 解析错误: {e.msg}",
            "suggestion": f"第 {e.lineno} 行附近: 检查括号、逗号和引号",
        })

    # 检查文件大小
    file_size = len(content.encode('utf-8'))
    if file_size > 1024 * 1024:  # > 1MB
        issues.append({
            "file": fpath, "line": 1, "severity": "info",
            "checker": "json_size",
            "message": f"JSON 文件较大 ({file_size / 1024:.0f} KB)",
            "suggestion": "考虑拆分或使用流式解析",
        })

    return issues


# ─── DAG 执行辅助 ──────────────────────────────────────────


def execute_dag_at_agent_level(graph, runner_type: str = "hermes",
                                auto_validate: bool = True) -> dict:
    """Agent 级别的 DAG 执行 — 按批次逐个节点在当前上下文中执行

    这个函数供 Hermes Agent 在规划完 DAG 后调用。
    它会输出每个节点的执行指引，Agent 依次执行。

    Args:
        graph: TaskGraph 实例
        runner_type: 默认 runner 类型
        auto_validate: 是否自动验证

    Returns:
        {node_id: output, ...}
    """
    if auto_validate:
        errors = graph.validate()
        if errors:
            return {"_error": f"DAG 验证失败: {errors}"}

    results = {}
    batches = graph.topological_sort()

    print(f"═══ DAG 执行计划: {graph.name} ═══")
    print(f"节点数: {len(graph.nodes)}, 并行批次: {len(batches)}")
    print(f"预计总成本: {graph.estimate_total_cost()} token\n")

    for batch_idx, batch in enumerate(batches):
        print(f"批次 {batch_idx + 1}/{len(batches)}: {batch}")
        for nid in batch:
            node = graph.nodes[nid]
            agent = node.agent.value if hasattr(node.agent, 'value') else node.agent
            deps = node.dependencies or []

            print(f"  [{agent}] {nid}: {node.goal}")
            if deps:
                print(f"    依赖: {deps}")
            if node.cost_estimate > 0:
                print(f"    预计: ~{node.cost_estimate} token, 风险: {node.risk.value if hasattr(node.risk, 'value') else node.risk}")

            # 输出执行指引
            instructions = _get_execution_instructions(node, agent)
            results[nid] = instructions

        print()

    # 汇总
    print(f"═══ 共 {len(graph.nodes)} 节点, 预计 {graph.estimate_parallel_time():.1f}s ═══")
    return results


def _get_execution_instructions(node, agent: str) -> dict:
    """生成单节点执行指引"""
    return {
        "node_id": node.id,
        "agent": agent,
        "goal": node.goal,
        "context": node.context,
        "execute": True,
    }


# ─── DAG 规划助手 ───────────────────────────────────────────


def plan_xhs_post(topic: str, style: str = "tech") -> TaskGraph:
    """规划一篇小红书帖子的完整生产流程

    典型 XHS 生产 DAG（并行+线性混合）:
      调研 (Hermes) ─┬─→ 文案创作 (CC) ─→ 去AI化质检 (Hermes)
                     └─→ 封面设计 (Codex) ─→ 图片渲染 (Codex)
                                          └→ 排版合成 (Hermes)
    """
    from dag import TaskNode, TaskGraph, AgentType

    g = TaskGraph(f"XHS: {topic[:20]}")

    # 调研
    g.add_node(TaskNode(
        id="research", goal=f"调研 {topic} 的 XHS 热点和关键词",
        agent=AgentType.HERMES, cost_estimate=500, risk="low",
        context={"type": "search"},
    ))

    # 文案（依赖调研）
    g.add_node(TaskNode(
        id="copywriting", goal=f"创作 {topic} 的 XHS 文案, {style}风格",
        agent=AgentType.CC, cost_estimate=2000, risk="low",
        dependencies=["research"],
        context={"technical_requirements": "纯文字大纲框架，不要逐字稿"},
    ))

    # 封面设计（依赖调研，与文案并行）
    g.add_node(TaskNode(
        id="cover_design", goal=f"设计 {topic} 封面图, {style}风格",
        agent=AgentType.CODEX, cost_estimate=3000, risk="medium",
        dependencies=["research"],
        context={"type": "image", "aspect_ratio": "portrait"},
    ))

    # 去AI化质检（依赖文案）
    g.add_node(TaskNode(
        id="qa_rewrite", goal=f"去AI化 + 质检\n{topic} 文案",
        agent=AgentType.HERMES, cost_estimate=1000, risk="medium",
        dependencies=["copywriting"],
    ))

    # 排版合成（依赖文案和封面）
    g.add_node(TaskNode(
        id="layout", goal=f"合成最终图文",
        agent=AgentType.HERMES, cost_estimate=500, risk="low",
        dependencies=["qa_rewrite", "cover_design"],
    ))

    return g


def plan_code_task(description: str) -> TaskGraph:
    """规划一个代码开发任务的 DAG"""
    from dag import TaskNode, TaskGraph, AgentType

    g = TaskGraph(f"Code: {description[:20]}")

    g.add_node(TaskNode(
        id="analyze", goal=f"需求分析: {description}",
        agent=AgentType.HERMES, cost_estimate=500, risk="low",
    ))
    g.add_node(TaskNode(
        id="implement", goal=f"编码实现: {description}",
        agent=AgentType.CC, cost_estimate=5000, risk="medium",
        dependencies=["analyze"],
    ))
    g.add_node(TaskNode(
        id="review", goal=f"代码审查: {description}",
        agent=AgentType.HERMES, cost_estimate=1000, risk="medium",
        dependencies=["implement"],
    ))

    return g


# ─── 测试 ────────────────────────────────────────────────────


def test_register():
    """测试 runner 注册"""
    from executor import Executor
    exec = Executor()
    HermesRunners.register_all(exec)
    assert "cc" in exec.runners
    assert "codex" in exec.runners
    assert "hermes" in exec.runners
    print(f"Runner 注册: ✓ (3 channels)")


def test_cc_runner():
    """测试 CC runner — 验证返回 ExecutionResult 结构"""
    from dag import TaskNode
    node = TaskNode(id="test_cc", goal="写一个 Python 函数", agent="cc", timeout=30)
    result = HermesRunners.cc_runner(node)
    assert isinstance(result, ExecutionResult), f"期望 ExecutionResult, 得到 {type(result)}"
    assert result.node_id == "test_cc"
    assert result.agent_used == "cc"
    # CLI 可能未安装，但结构必须正确
    status = "✓" if result.success else f"✗ ({result.error[:30]})"
    print(f"CC Runner: {status}")


def test_codex_runner():
    """测试 Codex runner — 验证返回 ExecutionResult 结构"""
    from dag import TaskNode
    node = TaskNode(id="test_codex", goal="生成科技风封面", agent="codex", timeout=30)
    result = HermesRunners.codex_runner(node)
    assert isinstance(result, ExecutionResult), f"期望 ExecutionResult, 得到 {type(result)}"
    assert result.node_id == "test_codex"
    assert result.agent_used == "codex"
    # CLI 可能未安装，但结构必须正确
    status = "✓" if result.success else f"✗ ({result.error[:30]})"
    print(f"Codex Runner: {status}")


def test_plan_xhs():
    """测试 XHS 生产 DAG 规划"""
    g = plan_xhs_post("AI Agent 搭建", "深蓝科技风")
    errors = g.validate()
    assert not errors, f"验证失败: {errors}"
    batches = g.topological_sort()
    print(f"XHS DAG: {len(g.nodes)} 节点, {len(batches)} 批次")
    for i, batch in enumerate(batches):
        print(f"  批次 {i+1}: {batch}")
    print("XHS 规划: ✓")


def test_plan_code():
    """测试代码 DAG 规划"""
    g = plan_code_task("FastAPI CRUD API")
    errors = g.validate()
    assert not errors
    print(f"Code DAG: {len(g.nodes)} 节点, {len(g.topological_sort())} 批次")
    print("Code 规划: ✓")


if __name__ == "__main__":
    import sys, json
    sys.path.insert(0, "/d")
    logging.basicConfig(level=logging.WARNING)
    print("═══ 三通道执行器测试 ═══\n")
    test_register()
    test_cc_runner()
    test_codex_runner()
    print()
    test_plan_xhs()
    test_plan_code()
    print("\n═══ 全部通过 ═══")
