"""Build the v1-format annotations.jsonl from the stratified 120 set."""
import json

rows = json.load(open("/home/cix/cand120/annot_strat120.json"))
with open("/home/cix/v1/annotations.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        rec = {
            "id": r["id"],
            "image_name": r["image_name"],
            "referring": r["referring"],
            "candidate_boxes": r.get("candidate_boxes", []),
            "answer_boxes": r.get("answer_boxes", []),
            "domain": r.get("domain", ""),
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("wrote", len(rows), "refs;",
      "avg candidates", round(sum(len(r.get("candidate_boxes", [])) for r in rows) / len(rows), 2))
