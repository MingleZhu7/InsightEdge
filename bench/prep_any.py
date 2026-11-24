"""Letterbox 640 preprocessing for an arbitrary annotation file (container side).

Usage: python3 prep_any.py <ann.json> <img_dir> <out_xs.npy> <out_meta.json>
"""
import json
import sys

import numpy as np
from PIL import Image

ann_path, img_dir, out_xs, out_meta = sys.argv[1:5]
rows = json.load(open(ann_path))
xs, meta = [], []
for r in rows:
    im = Image.open(f"{img_dir}/{r['image_name']}").convert("RGB")
    w, h = im.size
    sc = min(640 / w, 640 / h)
    nw, nh = int(round(w * sc)), int(round(h * sc))
    im2 = im.resize((nw, nh), Image.Resampling.BILINEAR)
    cv = Image.new("RGB", (640, 640), (114, 114, 114))
    cv.paste(im2, ((640 - nw) // 2, (640 - nh) // 2))
    xs.append((np.asarray(cv, np.float32) / 255.0).transpose(2, 0, 1))
    meta.append({"w": w, "h": h, "sc": sc, "dw": (640 - nw) // 2, "dh": (640 - nh) // 2})
np.save(out_xs, np.stack(xs).astype(np.float32))
json.dump(meta, open(out_meta, "w"))
print("prep done", np.stack(xs).shape)
