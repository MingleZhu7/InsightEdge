"""Rebuild YOLOE RepRTA text embeddings without the `clip` package.

tokenize (HF CLIPTokenizer) -> mobileclip2_b.ts -> YOLOEDetect.get_tpe -> W[N,512]
Validate against /cix/W200.npy on celebrity refs; write /cix/W_all.npy for a given ann file.
"""
import json
import sys

import numpy as np
import torch
from transformers import CLIPTokenizer
from ultralytics import YOLOE

ANN = sys.argv[1] if len(sys.argv) > 1 else "/cix/recall200/ann.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/cix/W_all.npy"
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 0

torch.set_grad_enabled(False)
yolo = YOLOE("/cix/yoloe-26l-seg.pt")
head = yolo.model.model[-1]
print("head:", type(head).__name__)
tok = CLIPTokenizer("/cix/clipL/vocab.json", "/cix/clipL/merges.txt")
DEV = "cuda" if "--cuda" in sys.argv else "cpu"
yolo.model.to(DEV).eval()
enc = torch.jit.load("/cix/mobileclip2_b.ts", map_location=DEV).eval()


def embed(texts):
    ids = tok(list(texts), padding="max_length", max_length=77, truncation=True,
              return_tensors="pt")["input_ids"].to(DEV)
    feats = enc(ids)                      # [N, 512], already L2-normalised
    for shape in ("BND", "ND"):
        x = feats.unsqueeze(0) if shape == "BND" else feats
        try:
            tpe = head.get_tpe(x)
            out = tpe.reshape(len(texts), -1)
            if out.shape[1] == 512:
                return out
        except Exception as e:  # noqa: BLE001
            print(f"  get_tpe({shape}) failed: {type(e).__name__}: {str(e)[:80]}")
    raise RuntimeError("no working get_tpe shape")


rows = json.load(open(ANN))
if LIMIT:
    rows = rows[:LIMIT]
texts = [r["referring"] for r in rows]
print(f"ann={ANN} refs={len(texts)}")

W = torch.cat([embed(texts[i:i + 20]) for i in range(0, len(texts), 20)], 0).float().cpu().numpy()
np.save(OUT, W)
print("W", W.shape, "norm", float(np.linalg.norm(W[0])))

# ---- validation against the known-good W200
try:
    W200 = np.load("/cix/W200.npy").astype(np.float32)
    if len(W) >= 4 and W200.shape[0] >= 4:
        for i in range(4):
            a, b = W[i], W200[i]
            cosn = float((a / np.linalg.norm(a)) @ (b / np.linalg.norm(b)))
            print(f"  validate[{i}] {rows[i]['referring'][:40]!r}: cos {cosn:+.5f}")
except Exception as e:  # noqa: BLE001
    print("validation skipped:", e)

