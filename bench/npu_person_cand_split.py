"""NPU person candidates from the split-output INT8 model (box / score / mask / proto separate).

Run on the host: LD_LIBRARY_PATH=/usr/share/cix/lib python3 npu_person_cand_split.py [conf]
"""
import json
import sys
import time

import numpy as np
from NOE_Engine import EngineInfer

CONF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
ENGINE = sys.argv[2] if len(sys.argv) > 2 else "/home/cix/yoloe26l_person_split_i8.cix"
OUT = sys.argv[3] if len(sys.argv) > 3 else "/home/cix/cand120/annot_npu_split_i8.json"
IOU = 0.7

xs = np.load("/home/cix/xs_strat120.npy")
meta = json.load(open("/home/cix/xs_strat120_meta.json"))
rows = json.load(open("/home/cix/cand120/annot_strat120.json"))

m = EngineInfer(ENGINE)
for i in range(3):
    m.forward(xs[i:i + 1])

t0 = time.perf_counter()
raw = []
for i in range(len(xs)):
    outs = [np.asarray(o, np.float32).ravel() for o in m.forward(xs[i:i + 1])]
    raw.append(outs)
m.clean()
ms = (time.perf_counter() - t0) / len(xs) * 1000
print(f"NPU {ms:.2f} ms/img | outputs {len(raw[0])} sizes {[o.size for o in raw[0]]}", flush=True)


def pick(outs, n):
    for o in outs:
        if o.size == n:
            return o.reshape(-1)
    return None


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1); inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


out, counts, nraw = [], [], []
for i, r in enumerate(rows):
    box = pick(raw[i], 4 * 8400)
    score = pick(raw[i], 8400)
    if box is None or score is None:
        raise RuntimeError(f"missing box/score in {[o.size for o in raw[i]]}")
    box = box.reshape(4, 8400).T
    bx = np.stack([box[:, 0] - box[:, 2] / 2, box[:, 1] - box[:, 3] / 2,
                   box[:, 0] + box[:, 2] / 2, box[:, 1] + box[:, 3] / 2], 1)
    md = meta[i]
    w, h, sc, dw, dh = md["w"], md["h"], md["sc"], md["dw"], md["dh"]
    bx[:, [0, 2]] = (bx[:, [0, 2]] - dw) / sc
    bx[:, [1, 3]] = (bx[:, [1, 3]] - dh) / sc
    bx[:, [0, 2]] = bx[:, [0, 2]].clip(0, w)
    bx[:, [1, 3]] = bx[:, [1, 3]].clip(0, h)

    idx = np.where(score >= CONF)[0]
    nraw.append(len(idx))
    if idx.size:
        sb = [bx[k].tolist() for k in idx]
        ss = score[idx].tolist()
        order = np.argsort(-np.asarray(ss, np.float32)); keep = []
        for j in order:
            if all(iou(sb[j], sb[q]) <= IOU for q in keep):
                keep.append(int(j))
        boxes = [sb[j] for j in keep]; scores = [ss[j] for j in keep]
    else:
        boxes, scores = [], []
    r2 = dict(r); r2["candidate_boxes"] = boxes; r2["cand_scores"] = scores
    out.append(r2); counts.append(len(boxes))

json.dump(out, open(OUT, "w"), ensure_ascii=False)
print(f"wrote {OUT} | boxes/img {np.mean(counts):.2f} (min {min(counts)}, max {max(counts)}) | "
      f"above-conf/img {np.mean(nraw):.1f}")
