"""Dry-run the demo question set so the recording script only contains verified answers."""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8849/api/chat"
CASES = [
    ("tomatoes", "哪些番茄已经熟了？"),
    ("tomatoes", "画面里一共有几颗番茄？"),
    ("tomatoes", "最左边那颗番茄熟了吗？"),
    ("proposal", "穿蓝色裙子的女士在哪里？"),
    ("proposal", "她右边是谁？"),
    ("webinar", "讲者里有哪几位女性？"),
    ("team", "穿橙色连衣裙的女士在哪？"),
    ("team", "画面里有狗吗？"),
    ("highway", "画面里有哪些车？"),
    ("highway", "找出那辆白色大巴"),
]


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
    return ms, tool, (r.get("reply") or "").replace("\n", " ")[:90], len(r.get("objects") or [])


for i, (sample, q) in enumerate(CASES):
    try:
        ms, tool, reply, n = ask(sample, q, f"s_demo{i:02d}")
        print(f"[{sample:9s}] {q:18s} {ms:7.0f} ms tool={tool:11s} obj={n:2d} | {reply}")
    except Exception as e:  # noqa: BLE001
        print(f"[{sample:9s}] {q:18s} ERROR {type(e).__name__}: {e}")
