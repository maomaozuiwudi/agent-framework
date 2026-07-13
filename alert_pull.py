"""告警轮询推送 — 由 cron job 每 5 分钟执行一次
读取 memory 中 alert:wechat: 前缀的新条目，汇总推微信。

部署：
  cronjob action=create name="告警推送" schedule="every 5m" \
    script="D:\\agent_framework\\alert_pull.py" no_agent=True
"""

import sys, json
sys.path.insert(0, "D:")

try:
    from agent_framework.hermes_memory import l3_search

    results = l3_search("alert:wechat:", limit=10)
    if not results:
        exit(0)  # 无告警静默

    alerts = []
    for r in results:
        snippet = r.get("snippet", "")
        try:
            json_str = snippet.split(": ", 1)[1] if ": " in snippet else snippet
            data = json.loads(json_str)
            alerts.append(data)
        except (json.JSONDecodeError, IndexError):
            alerts.append({"title": "未知告警", "message": snippet[:100]})

    if alerts:
        print(f"📢 告警汇总 ({len(alerts)} 条)")
        print()
        for a in alerts[-5:]:
            icon = {"WARNING": "⚠️", "ERROR": "❌", "CRITICAL": "🚨"}.get(
                a.get("severity", "WARNING"), "🔹")
            print(f"{icon} {a.get('title', '未知')}")
            print(f"  {a.get('message', '')}")
            print()

except Exception as e:
    print(f"❌ 告警轮询异常: {e}")
    exit(1)
