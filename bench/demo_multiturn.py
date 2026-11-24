"""Check multi-turn reference memory (same session) and a couple of cleaner phrasings."""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8849/api/chat"


def ask(sample, q, sid):
    body = json.dumps({"session_id": sid, "messages": [{"role": "user", "content": q}],
                       "media": {"sample_id": sample}}).encode()
    t = time.perf_counter()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        BASE, data=body, headers={"Content-Type": "application/json"}), timeout=300))
    ms = (time.perf_counter() - t) * 1000
    tool = "-"
    for st in r.get("trace", []):
        if st.get("type") == "tool":
            tool = st.get("tool", "-")
        elif st.get("type") == "plan" and st.get("steps"):
            tool = st["steps"][0].get("tool", "-")
    return ms, tool, (r.get("reply") or "").replace("\n", " ")[:100], len(r.get("objects") or [])


print("=== 多轮指代（同一 session）===")
sid = "s_multi_test_01"
for q in ["穿蓝色裙子的女士在哪里？", "她右边是谁？", "她是谁？"]:
    ms, tool, reply, n = ask("proposal", q, sid)
    print(f"  {q:16s} {ms:7.0f} ms tool={tool:10s} obj={n} | {reply}")

print("=== 番茄（换问法，避开旧缓存）===")
sid2 = "s_multi_test_02"
for q in ["成熟的红番茄有几颗？", "这些番茄里哪些还没熟？"]:
    ms, tool, reply, n = ask("tomatoes", q, sid2)
    print(f"  {q:16s} {ms:7.0f} ms tool={tool:10s} obj={n} | {reply}")
