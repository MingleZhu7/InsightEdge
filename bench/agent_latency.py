"""Agent latency baseline over different tool routes (fresh session per question)."""
import base64
import json
import statistics
import time
import urllib.request

URL = "http://127.0.0.1:8848/api/chat"
IMG = "/home/cix/rexseek-studio/web/assets/samples/helmet_all.jpg"
img_b64 = "data:image/jpeg;base64," + base64.b64encode(open(IMG, "rb").read()).decode()

QUESTIONS = [
    "画面里有几个人？",
    "左边的人戴安全帽了吗？",
    "现在危险区有人吗？",
    "画面里的人在哪里？",
    "这个人穿什么颜色的衣服？",
    "描述一下这张图",
    "你好",
    "画面里有几个人？",          # repeat -> cache
    "左边的人戴安全帽了吗？",      # repeat -> cache
]


def ask(q, sid):
    payload = {"session_id": sid, "messages": [{"role": "user", "content": q}],
               "media": {"data_url": img_b64}}
    body = json.dumps(payload).encode()
    t0 = time.perf_counter()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    return (time.perf_counter() - t0) * 1000, r


rows = []
for i, q in enumerate(QUESTIONS):
    try:
        ms, r = ask(q, f"s_speed{i:02d}")
    except Exception as e:  # noqa: BLE001
        print(f"{q:24s} ERROR {type(e).__name__}: {e}")
        continue
    tool = "-"
    for step in r.get("trace", []):
        if step.get("type") == "tool":
            tool = step.get("tool", "-")
        elif step.get("type") == "plan" and step.get("steps"):
            tool = step["steps"][0].get("tool", "-")
    ans = (r.get("reply") or "")[:46]
    rows.append((q, ms, tool))
    print(f"{q:24s} {ms:8.1f} ms  tool={tool:12s} {ans}")

cold = [r[1] for r in rows[:6]]
print(f"\n非缓存 {len(cold)} 条：mean {statistics.mean(cold):.0f} ms | "
      f"median {statistics.median(cold):.0f} ms | min {min(cold):.0f} | max {max(cold):.0f}")
if len(rows) > 6:
    print(f"重复同一问题（缓存命中）：{[round(r[1],1) for r in rows[6:]]} ms")

