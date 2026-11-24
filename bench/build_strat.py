"""Build a domain-stratified HumanRef subset: 20 refs per domain (6 domains = 120).

Writes /home/cix/cand120/annot_strat120.json and symlinks the images into /home/cix/strat120/.
Only refs whose image exists in /home/cix/bench/imgs are taken.
"""
import collections
import json
import os

SRC = "/home/cix/HumanRef/annotations.jsonl"
IMG = "/home/cix/bench/imgs"
OUT_ANN = "/home/cix/cand120/annot_strat120.json"
OUT_IMG = "/home/cix/strat120"
PER = 20

rows = [json.loads(line) for line in open(SRC, encoding="utf-8")]
by_dom = collections.defaultdict(list)
for r in rows:
    if os.path.exists(os.path.join(IMG, r["image_name"])):
        by_dom[r["domain"]].append(r)

os.makedirs(OUT_IMG, exist_ok=True)
IMAGES_STAY_IN_PLACE = True   # container sees the same files at /cix/bench/imgs
sel = []
for dom in ["attribute", "position", "interaction", "reasoning", "celebrity", "rejection"]:
    take = by_dom[dom][:PER]
    print(f"{dom:12s} available {len(by_dom[dom]):5d} -> take {len(take)}")
    sel.extend(take)

out = []
for i, r in enumerate(sel):
    pass
    out.append({
        "id": i,
        "image_name": r["image_name"],
        "referring": r["referring"],
        "chinese_referring": r.get("chinese_referring", ""),
        "domain": r["domain"],
        "sub_domain": r.get("sub_domain", ""),
        "answer_boxes": r["answer_boxes"],
        "candidate_boxes": r.get("candidate_boxes", []),
    })

json.dump(out, open(OUT_ANN, "w"), ensure_ascii=False)
print(f"\nwrote {OUT_ANN}: {len(out)} refs | images {len(os.listdir(OUT_IMG))}")
print("domain counts:", dict(collections.Counter(x["domain"] for x in out)))
print("avg GT boxes:", round(sum(len(x["answer_boxes"]) for x in out) / len(out), 2))

