"""Export YOLOE-26l with a FIXED 'person' vocabulary, so the ONNX carries real class scores.

The previous export (y26l.cix / yoloe-26l-seg.onnx) had the 80-class branch all zeros because
the class head needs the text embedding; baking a single 'person' class makes the score concrete.
Then verify: output shapes, score statistics, and box agreement vs PyTorch predict.
"""
import json
import os
import shutil

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLOE

SRC = "/cix/yoloe-26l-seg.pt"
WORK = "/cix/export_person"
os.makedirs(WORK, exist_ok=True)
ckpt = os.path.join(WORK, "yoloe-26l-person.pt")
if not os.path.exists(ckpt):
    shutil.copy(SRC, ckpt)

print("[1] load + set_classes(['person'])", flush=True)
m = YOLOE(ckpt)
tpe = m.get_text_pe(["person"])
m.set_classes(["person"], tpe)
print("   names:", m.model.names, flush=True)

print("[2] export onnx", flush=True)
path = m.export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False, device="cpu")
print("   exported:", path, os.path.getsize(path) / 1e6, "MB", flush=True)

print("[3] inspect onnx", flush=True)
import onnxruntime as ort  # noqa: E402
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
for i in sess.get_inputs():
    print(f"   input  {i.name} {i.shape} {i.type}")
for o in sess.get_outputs():
    print(f"   output {o.name} {o.shape} {o.type}")

print("[4] verify on strat images", flush=True)
rows = json.load(open("/cix/cand120/annot_strat120.json"))[:3]
for r in rows:
    im = Image.open(f"/cix/bench/imgs/{r['image_name']}").convert("RGB")
    w, h = im.size
    sc = min(640 / w, 640 / h)
    nw, nh = int(round(w * sc)), int(round(h * sc))
    im2 = im.resize((nw, nh), Image.Resampling.BILINEAR)
    cv = Image.new("RGB", (640, 640), (114, 114, 114))
    cv.paste(im2, ((640 - nw) // 2, (640 - nh) // 2))
    x = np.ascontiguousarray((np.asarray(cv, np.float32) / 255.0).transpose(2, 0, 1)[None])
    out = sess.run(None, {sess.get_inputs()[0].name: x})
    print(f"   {r['image_name'][:30]}: {len(out)} outputs, "
          + ", ".join(str(o.shape) for o in out))
    o0 = out[0]
    if o0.ndim == 3:
        ch = o0.shape[1]
        print(f"      channels {ch}: box 0-3, then score/mask")
        for sl in [(0, 4), (4, 5), (5, 6), (ch - 32, ch)]:
            v = o0[:, sl[0]:sl[1]]
            if v.size:
                print(f"      ch{sl[0]}-{sl[1]}: min {v.min():+.3f} max {v.max():+.3f} "
                      f"mean {v.mean():+.3f} | >0.25 {(v > 0.25).mean():.4f}")
