"""Split the mixed-range 37-channel output into separate tensors.

The AIPU compiler zeroes the person-score channel when box/score/mask share one INT8 Concat
(their value ranges differ by 3 orders of magnitude). Emitting box / score / mask as separate
graph outputs gives each its own quantization scale.

Outputs after the split: box [1,4,8400], score [1,1,8400], mask [1,32,8400], plus the mask
prototypes (original output1).
"""
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

SRC = "/cix/export_person/yoloe-26l-person.onnx"
DST = "/cix/export_person/yoloe-26l-person-split.onnx"

m = onnx.load(SRC)
g = m.graph
names = {"output0", "output1"}
cat = None
for n in g.node:
    if "output0" in n.output:
        cat = n
if cat is None:
    raise SystemExit("output0 producer not found")
print("concat node:", cat.name, cat.op_type, "inputs:", list(cat.input), flush=True)

shapes = {"box_out": [1, 4, 8400], "score_out": [1, 1, 8400], "mask_out": [1, 32, 8400]}
out_names = ["box_out", "score_out", "mask_out"]
for inp, name in zip(cat.input, out_names):
    g.node.append(helper.make_node("Identity", [inp], [name]))
    g.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shapes[name]))

# drop the old 37-channel output, keep output1 (mask prototypes)
keep = [o for o in g.output if o.name != "output0"]
del g.output[:]
g.output.extend(keep)
onnx.checker.check_model(m)
onnx.save(m, DST)
print("saved", DST, flush=True)

# sanity: split scores must equal channel 4 of the original concat
s1 = ort.InferenceSession(SRC, providers=["CPUExecutionProvider"])
s2 = ort.InferenceSession(DST, providers=["CPUExecutionProvider"])
from PIL import Image  # noqa: E402

im = Image.open("/cix/bench/imgs/1120068788.jpg").convert("RGB")
w, h = im.size
sc = min(640 / w, 640 / h)
nw, nh = int(round(w * sc)), int(round(h * sc))
im2 = im.resize((nw, nh), Image.Resampling.BILINEAR)
cv = Image.new("RGB", (640, 640), (114, 114, 114))
cv.paste(im2, ((640 - nw) // 2, (640 - nh) // 2))
x = np.ascontiguousarray((np.asarray(cv, np.float32) / 255.0).transpose(2, 0, 1)[None])

a = s1.run(None, {s1.get_inputs()[0].name: x})[0]
outs = {o.name: o for o in s2.get_outputs()}
b = s2.run(None, {s2.get_inputs()[0].name: x})
d = dict(zip([o.name for o in s2.get_outputs()], b))
print("split outputs:", [(k, v.shape) for k, v in d.items()])
print("box  max diff", float(np.abs(d["box_out"] - a[:, :4]).max()))
print("score max diff", float(np.abs(d["score_out"].reshape(1, -1) - a[:, 4]).max()))
print("mask max diff", float(np.abs(d["mask_out"] - a[:, 5:]).max()))
print("score stats: max %.4f, n>0.25 %d" % (d["score_out"].max(), int((d["score_out"] > 0.25).sum())))
