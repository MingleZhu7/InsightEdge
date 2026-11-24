"""Unified RexSeek benchmark: one protocol, three candidate sources, L towers + Qwen.

Measures per stage (ms) and accuracy (R/P/DF1@0.5, @0.5:0.95) on the same 120 HumanRef
images, so the detector comparison is apples-to-apples:

  --cand gd         Grounding DINO graft448 IxRT engine (v1 deployed path), live
  --cand yoloe_gpu  YOLOE-26l IxRT fp16 engine, live
  --cand yoloe_npu  YOLOE-26l NPU INT8 candidates (precomputed NMS/top-6)

Stages: prep (decode+resize+normalize+box encode+tokenize) / det / nms /
        clip(L14-336) / convnext(L@768) / proj / qwen ; total = serial sum.
"""
import argparse
import json
import re
import statistics
import sys
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, "/cix/third_party/RexSeek")
from rexseek.builder import load_rexseek_model
from rexseek.utils.inference_utils import (modify_processor_resolution, expand2square,
                                           pad_boxes, resize_boxes, split_special_strings)
from rexseek.utils.constants import (IMAGE_TOKEN_INDEX, DEFAULT_OBJECT_INDEX,
                                     DEFAULT_IMAGE_TOKEN, DEFAULT_OBJECT_FEATURE_TOKEN)

sys.path.insert(0, "/work/dino")
from common import create_engine_context, get_io_bindings  # noqa: E402
from load_ixrt_plugin import load_ixrt_plugin  # noqa: E402
import cuda.cudart as cudart  # noqa: E402
import tensorrt  # noqa: E402

IMG_DIR = "/cix/recall200"
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
DINO_W, DINO_H = 672, 448
MAXT = 128
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
TOPK = 6   # overridden by --topk


def sync():
    torch.cuda.synchronize()


def gpu_sync():
    (err,) = cudart.cudaDeviceSynchronize()
    assert err == cudart.cudaError_t.cudaSuccess


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1); inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(boxes, scores, iou_th=0.7, k=TOPK, prefilter=300):
    """Vectorised greedy NMS (same result as the serial version, ~100x faster)."""
    if len(boxes) == 0:
        return []
    b = np.asarray(boxes, np.float32); s = np.asarray(scores, np.float32)
    order = np.argsort(-s)[:prefilter]
    b = b[order]
    keep = []
    while len(keep) < k and b.shape[0] > 0:
        keep.append(int(order[0]))
        if b.shape[0] == 1:
            break
        x1 = np.maximum(b[1:, 0], b[0, 0]); y1 = np.maximum(b[1:, 1], b[0, 1])
        x2 = np.minimum(b[1:, 2], b[0, 2]); y2 = np.minimum(b[1:, 3], b[0, 3])
        iw = np.maximum(0.0, x2 - x1); ih = np.maximum(0.0, y2 - y1)
        inter = iw * ih
        a1 = (b[0, 2] - b[0, 0]) * (b[0, 3] - b[0, 1])
        a2 = (b[1:, 2] - b[1:, 0]) * (b[1:, 3] - b[1:, 1])
        ious = inter / np.maximum(a1 + a2 - inter, 1e-9)
        order = order[1:][ious <= iou_th]
        b = b[1:][ious <= iou_th]
    return [boxes[i] for i in keep]


# ------------------------------------------------------------------ detectors
class GdIxrt:
    """Grounding DINO graft448 (official RexSeek front-end) via IxRT engine."""

    def __init__(self, path="/work/weights/graft448_fc.engine"):
        from tokenizers import Tokenizer
        load_ixrt_plugin()
        logger = tensorrt.Logger(tensorrt.Logger.ERROR)
        self.eng, self.ctx = create_engine_context(path, logger)
        self.t_in, self.t_out, self.alloc = get_io_bindings(self.eng)
        self.tk = Tokenizer.from_file("/work/dino/tokenizer_config/tokenizer.json")
        self.special = [i for i in (self.tk.token_to_id("[CLS]"), self.tk.token_to_id("[SEP]"),
                                    self.tk.token_to_id("."), self.tk.token_to_id("?"))
                        if i is not None]

    def _masks(self, ids_row, n=MAXT):
        mask = np.zeros((n, n), bool); pos = np.zeros(n, np.int64); prev = 0
        for col in np.nonzero(np.isin(ids_row, self.special))[0]:
            if col == 0 or col == n - 1:
                mask[col, col] = True; pos[col] = 0
            else:
                mask[prev + 1:col + 1, prev + 1:col + 1] = True
                pos[prev + 1:col + 1] = np.arange(0, col - prev)
            prev = col
        return mask[None], pos[None]

    def prepare(self, pil, text):
        img = pil.resize((DINO_W, DINO_H), Image.Resampling.BILINEAR)
        a = np.asarray(img, np.float32) / 255.0
        a = (a - np.array(DINO_MEAN, np.float32)) / np.array(DINO_STD, np.float32)
        xd = np.ascontiguousarray(a.transpose(2, 0, 1)[None])
        enc = self.tk.encode(text)
        ids = np.zeros((1, MAXT), np.int64)
        ids[0, :min(len(enc.ids), MAXT)] = enc.ids[:MAXT]
        tmask, tpos = self._masks(ids[0])
        return {"img": xd, "input_ids": ids,
                "attention_mask": np.ones((1, MAXT), np.int64),
                "token_type_ids": np.zeros((1, MAXT), np.int64),
                "pixel_mask": np.ones((1, DINO_H, DINO_W), np.int64),
                "text_token_mask": tmask, "position_ids": tpos}

    def run(self, feeds):
        for b in self.t_in:
            arr = np.ascontiguousarray(feeds[b["name"]])
            cudart.cudaMemcpy(b["allocation"], arr, b["nbytes"],
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        gpu_sync(); t0 = time.perf_counter()
        self.ctx.execute_v2(self.alloc)
        gpu_sync()
        ms = (time.perf_counter() - t0) * 1000
        bufs = []
        for spec in self.t_out:
            buf = np.zeros(spec["shape"], spec["dtype"])
            cudart.cudaMemcpy(buf, spec["allocation"], spec["nbytes"],
                              cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            bufs.append(buf)
        return bufs[0], bufs[1], ms


class YoloeGpu:
    """YOLOE-26l IxRT fp16 engine: box + 512-d embedding, referring-expression scoring."""

    def __init__(self, engine="/cix/emb_fp16.engine", W=None):
        load_ixrt_plugin()
        logger = tensorrt.Logger(tensorrt.Logger.ERROR)
        self.eng, self.ctx = create_engine_context(engine, logger)
        self.t_in, self.t_out, self.alloc = get_io_bindings(self.eng)
        self.W = W

    @staticmethod
    def prepare(pil):
        w, h = pil.size
        sc = min(640 / w, 640 / h)
        nw, nh = int(round(w * sc)), int(round(h * sc))
        im2 = pil.resize((nw, nh), Image.Resampling.BILINEAR)
        cv = Image.new("RGB", (640, 640), (114, 114, 114))
        cv.paste(im2, ((640 - nw) // 2, (640 - nh) // 2))
        x = np.ascontiguousarray((np.asarray(cv, np.float32) / 255.0).transpose(2, 0, 1)[None])
        return x, sc, nw, nh

    def run(self, x):
        cudart.cudaMemcpy(self.t_in[0]["allocation"], x, self.t_in[0]["nbytes"],
                          cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        gpu_sync(); t0 = time.perf_counter()
        self.ctx.execute_v2(self.alloc)
        gpu_sync()
        ms = (time.perf_counter() - t0) * 1000
        buf = np.zeros(self.t_out[0]["shape"], self.t_out[0]["dtype"])
        cudart.cudaMemcpy(buf, self.t_out[0]["allocation"], self.t_out[0]["nbytes"],
                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        return np.asarray(buf).reshape(1, 516, 8400).astype(np.float32), ms


def yoloe_boxes(out, sc, nw, nh, w, h, Wvec):
    o = out[0]
    raw = o[:4].T
    cx, cy, wd, ht = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    bx = np.stack([cx - wd / 2, cy - ht / 2, cx + wd / 2, cy + ht / 2], 1)
    dw, dh = (640 - nw) // 2, (640 - nh) // 2
    bx[:, [0, 2]] = (bx[:, [0, 2]] - dw) / sc
    bx[:, [1, 3]] = (bx[:, [1, 3]] - dh) / sc
    bx[:, [0, 2]] = bx[:, [0, 2]].clip(0, w)
    bx[:, [1, 3]] = bx[:, [1, 3]].clip(0, h)
    s = Wvec @ o[4:]
    return bx.tolist(), s.tolist()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, choices=["gd", "yoloe_gpu", "yoloe_npu", "ann", "hf_gd"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default=None)
    ap.add_argument("--wsub", type=int, default=40, help="images used for independent tower timing")
    ap.add_argument("--det-ms", type=float, default=0.0, help="external detector time for yoloe_npu")
    ap.add_argument("--ann", default="/cix/cand120/annot_gd120.json")
    ap.add_argument("--img-dir", default="/cix/recall200")
    ap.add_argument("--w", default="/cix/W200.npy")
    ap.add_argument("--npu-cand", default="/cix/cand120/annot_npu120.json")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--gd-text", default="refer", choices=["refer", "person"])
    ap.add_argument("--thr", type=float, default=None, help="score threshold: keep ALL boxes above it (no K cap)")
    a = ap.parse_args()

    global TOPK
    TOPK = a.topk
    rows = json.load(open(a.ann))[:a.n]
    out_path = a.out or f"/cix/bench_unified_{a.cand}.json"
    print(f"[bench] cand={a.cand} n={len(rows)}", flush=True)

    print("[bench] loading RexSeek-3B (official, L towers) ...", flush=True)
    tokenizer, model, ip, _ = load_rexseek_model("/cix/models_orig/IDEA-Research__RexSeek-3B")
    ip, crop = modify_processor_resolution(ip)
    model.eval()
    VT, VA = model.get_vision_tower(), model.get_vision_tower_aux()

    W200 = np.load(a.w).astype(np.float32)
    if a.cand == "gd":
        det = GdIxrt()
    elif a.cand == "yoloe_gpu":
        det = YoloeGpu(W=W200)
    elif a.cand == "hf_gd":
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        det = None
        gd_proc = AutoProcessor.from_pretrained("/cix/rexseek-studio/models/grounding-dino-tiny")
        gd_hf = AutoModelForZeroShotObjectDetection.from_pretrained(
            "/cix/rexseek-studio/models/grounding-dino-tiny").to("cuda").eval()
        print("[bench] stock HF GroundingDINO-tiny loaded", flush=True)
    else:
        det = None
        npu_rows = json.load(open(a.npu_cand))

    def build_tower_inputs(image, cand):
        ow, oh = TF.get_image_size(image)
        im = expand2square(image, tuple(int(x * 255) for x in ip.image_mean))
        pw, ph = TF.get_image_size(im)
        aux = np.asarray(im.resize((768, 768), Image.BILINEAR), np.float32) / 255.0
        aux = torch.from_numpy(((aux - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1))
        pv = torch.nn.functional.interpolate(aux[None], size=[crop["height"], crop["width"]],
                                             mode="bilinear", align_corners=False)
        bb = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in cand]
        bb = pad_boxes(bb, (ow, oh))
        bb = resize_boxes(bb, (pw, ph), (768, 768))
        gtb = torch.tensor([[x, y, x + w, y + h] for x, y, w, h in bb]).unsqueeze(0)
        return pv, aux[None], gtb.to(dtype=torch.float16, device="cuda")

    def tok(prompt, tk):
        bos = getattr(tk, "bos_token_id", None)
        ids = [bos] if bos is not None else []
        for ch in split_special_strings(prompt, [DEFAULT_IMAGE_TOKEN, DEFAULT_OBJECT_FEATURE_TOKEN]):
            if ch == DEFAULT_IMAGE_TOKEN:
                ids.append(IMAGE_TOKEN_INDEX)
            elif ch == DEFAULT_OBJECT_FEATURE_TOKEN:
                ids.append(DEFAULT_OBJECT_INDEX)
            else:
                r = tk(ch, add_special_tokens=False)["input_ids"]
                if len(r) > 0 and isinstance(r[0], list):
                    r = r[0]
                ids.extend(int(v) for v in r)
        return ids

    # warmup (model + engines + towers)
    print("[bench] warmup ...", flush=True)
    for r in rows[:2]:
        image = Image.open(f"{a.img_dir}/{r['image_name']}").convert("RGB")
        text = (r["referring"].rstrip(".") + ".").lower()
        if a.cand == "gd":
            feeds = det.prepare(image, text)
            _, _, _ = det.run(feeds)
            cand = [[10, 10, image.size[0] * .5, image.size[1] * .5]]
        elif a.cand == "yoloe_gpu":
            x, sc, nw, nh = det.prepare(image)
            o, _ = det.run(x)
            bx, s = yoloe_boxes(o, sc, nw, nh, *image.size, W200[0])
            cand = [[10, 10, image.size[0] * .5, image.size[1] * .5]]
        else:
            cand = [[10, 10, image.size[0] * .5, image.size[1] * .5]]
        pv, aux, gtb = build_tower_inputs(image, cand)
        ids = torch.tensor(tok("<|im_start|>user\n" + DEFAULT_IMAGE_TOKEN + "\n<obj0>" +
                               DEFAULT_OBJECT_FEATURE_TOKEN + "\nPlease detect x. Answer the question with object indexes.<|im_end|>\n<|im_start|>assistant\n",
                               tokenizer)).unsqueeze(0).cuda()
        with torch.inference_mode(), torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            model.generate(ids, pixel_values=pv.cuda().to(torch.float16), pixel_values_aux=aux.cuda().to(torch.float16),
                           gt_boxes=gtb, do_sample=False, max_new_tokens=8, temperature=0.0, top_p=None)

    tm = {}
    preds = []
    t_all = time.perf_counter()
    n_tower = 0
    towers = []

    for i, r in enumerate(rows):
        wall0 = time.perf_counter()
        path = f"{a.img_dir}/{r['image_name']}"
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {r['image_name']}: {e}", flush=True)
            continue
        w, h = image.size
        text = (r["referring"].rstrip(".") + ".").lower()

        # -------- detector + its preprocessing
        t0 = time.perf_counter()
        det_ms = 0.0
        if a.cand == "gd":
            gd_text = text if a.gd_text == "refer" else "person."
            feeds = det.prepare(image, gd_text)
            prep_det = (time.perf_counter() - t0) * 1000
            logits, boxes, det_ms = det.run(feeds)
            sc_ = 1.0 / (1.0 + np.exp(-logits.reshape(-1, logits.shape[-1]).astype(np.float64)))
            score = sc_.max(axis=1)
            # same candidate budget as YOLOE: top-6 by score, no hard threshold
            keep = np.argsort(-score)[:200]
            bx = boxes.reshape(-1, 4)
            # engine boxes are normalized (cx, cy, w, h)
            cand_all = [[(bx[k][0] - bx[k][2] / 2) * w, (bx[k][1] - bx[k][3] / 2) * h,
                         (bx[k][0] + bx[k][2] / 2) * w, (bx[k][1] + bx[k][3] / 2) * h]
                        for k in keep]
            sc_all = [float(score[k]) for k in keep]
        elif a.cand == "yoloe_gpu":
            x, scl, nw, nh = det.prepare(image)
            prep_det = (time.perf_counter() - t0) * 1000
            o, det_ms = det.run(x)
            Wv = W200[i] if i < len(W200) else W200[0]
            cand_all, sc_all = yoloe_boxes(o, scl, nw, nh, w, h, Wv)
        elif a.cand == "hf_gd":
            gd_text = "person." if a.gd_text == "person" else text
            inp = gd_proc(images=image, text=gd_text, return_tensors="pt").to("cuda")
            prep_det = (time.perf_counter() - t0) * 1000
            sync(); t_det = time.perf_counter()
            with torch.inference_mode():
                out = gd_hf(**inp)
            sync(); det_ms = (time.perf_counter() - t_det) * 1000
            res = gd_proc.post_process_grounded_object_detection(
                out, inp.input_ids, box_threshold=0.25, text_threshold=0.25,
                target_sizes=[image.size[::-1]])[0]
            cand_all = res["boxes"].tolist()
            sc_all = res["scores"].tolist()
        elif a.cand == "ann":
            # dataset-provided candidate boxes (the setting v1 used): no detector, no NMS
            prep_det = (time.perf_counter() - t0) * 1000
            det_ms = 0.0
            cand_all = r.get("candidate_boxes", [])[:12]
            sc_all = [1.0 - 0.001 * k for k in range(len(cand_all))]
        else:
            # NPU path: candidates come from the NPU model; re-do its letterbox 640 here so the
            # preprocessing cost is counted, and take the measured NPU run time from --det-ms.
            YoloeGpu.prepare(image)
            prep_det = (time.perf_counter() - t0) * 1000
            det_ms = a.det_ms
            cand_all = npu_rows[i]["candidate_boxes"]
            sc_all = npu_rows[i].get("cand_scores", [1.0] * len(cand_all))

        t1 = time.perf_counter()
        if a.cand == "ann":
            cand = cand_all
        elif a.thr is not None:
            idx = [i for i, s_ in enumerate(sc_all) if s_ >= a.thr]
            cand = nms([cand_all[i] for i in idx], [sc_all[i] for i in idx], k=1000000) if idx else []
        else:
            cand = nms(cand_all, sc_all, k=TOPK) if cand_all else []
        nms_ms = (time.perf_counter() - t1) * 1000

        if not cand:
            preds.append({"id": r["id"], "image_name": r["image_name"], "raw_response": "",
                          "extracted_predictions": [], "n_cand": 0,
                          "answer_boxes": r["answer_boxes"], "referring": r["referring"]})
            continue

        # -------- L towers + Qwen
        t0 = time.perf_counter()
        pv, aux, gtb = build_tower_inputs(image, cand)
        obj_tok = [f"<obj{k}>" for k in range(len(cand))]
        s = obj_tok[0] + DEFAULT_OBJECT_FEATURE_TOKEN
        for tt in obj_tok[1:]:
            s += tt + DEFAULT_OBJECT_FEATURE_TOKEN
        q = DEFAULT_IMAGE_TOKEN + "\n" + s + "\n" + \
            f"Please detect {r['referring']} in this image. Answer the question with object indexes."
        prompt = "<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n"
        input_ids = torch.tensor(tok(prompt, tokenizer)).unsqueeze(0).cuda()
        prep_tower = (time.perf_counter() - t0) * 1000
        prep_ms = prep_det + prep_tower

        pv16, aux16 = pv.cuda().to(torch.float16), aux.cuda().to(torch.float16)

        # independent tower timing on a subset (warm, no graph capture)
        if n_tower < a.wsub:
            with torch.inference_mode(), torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                for _ in range(2):
                    _ = VT(pv16); _ = VA(aux16)
                sync()
                t0 = time.perf_counter(); _ = VT(pv16); sync(); clip_ms = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter(); _ = VA(aux16); sync(); cn_ms = (time.perf_counter() - t0) * 1000
            n_tower += 1
            towers.append((clip_ms, cn_ms))
        clip_ms = float(np.mean([t[0] for t in towers])) if towers else 0.0
        cn_ms = float(np.mean([t[1] for t in towers])) if towers else 0.0

        sync(); t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            o = model.generate(input_ids, pixel_values=pv16, pixel_values_aux=aux16, gt_boxes=gtb,
                               do_sample=False, max_new_tokens=64, temperature=0.0, top_p=None)
        sync(); gen_ms = (time.perf_counter() - t0) * 1000
        ans = tokenizer.batch_decode(o, skip_special_tokens=False)[0].strip()

        proj_ms = 0.0   # projector cost is inside generate; not separable here
        qwen_ms = max(gen_ms - clip_ms - cn_ms, 0.0)

        idxs = [int(m) for m in re.findall(r"<obj(\d+)", ans)]
        pred_boxes = [cand[j] for j in idxs if 0 <= j < len(cand)]
        wall = (time.perf_counter() - wall0) * 1000

        rec = {"id": r["id"], "image_name": r["image_name"], "domain": r.get("domain", ""), "raw_response": ans,
               "extracted_predictions": pred_boxes, "n_cand": len(cand),
               "answer_boxes": r["answer_boxes"], "referring": r["referring"],
               "t": {"prep": prep_ms, "det": det_ms, "nms": nms_ms, "clip": clip_ms,
                     "convnext": cn_ms, "proj": proj_ms, "qwen": qwen_ms,
                     "gen_total": gen_ms, "wall": wall}}
        preds.append(rec)
        for k, v in rec["t"].items():
            tm.setdefault(k, []).append(v)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(rows)}  wall {(time.perf_counter()-t_all)/(i+1):.2f}s/条", flush=True)

    wall_all = time.perf_counter() - t_all

    def stat(k):
        v = tm.get(k, [])
        return (float(statistics.mean(v)), float(statistics.median(v)), len(v)) if v else (0.0, 0.0, 0)

    keys = ["prep", "det", "nms", "clip", "convnext", "proj", "qwen", "gen_total"]
    summary = {k: {"mean": stat(k)[0], "median": stat(k)[1], "n": stat(k)[2]} for k in keys}
    summary["serial_total_ms"] = sum(summary[k]["mean"] for k in keys if k != "gen_total")
    summary["wall_per_item_s"] = wall_all / max(len(preds), 1)
    summary["n_items"] = len(preds)
    summary["cand"] = a.cand

    print(f"\n===== {a.cand} 分阶段（均值 ms）=====", flush=True)
    for k in keys:
        print(f"  {k:10s} {summary[k]['mean']:9.2f}  中位 {summary[k]['median']:9.2f}  n={summary[k]['n']}", flush=True)
    print(f"  {'串行合计':10s} {summary['serial_total_ms']:9.2f}", flush=True)
    print(f"  墙钟/条   {summary['wall_per_item_s']*1000:9.2f}", flush=True)

    def evaluate(th, items=None):
        rs, ps = [], []
        for p in (preds if items is None else items):
            gts = p.get("answer_boxes") or []
            pds = p["extracted_predictions"] or []
            if not gts:
                continue
            hit = sum(1 for g in gts if any(iou(g, q_) >= th for q_ in pds))
            rs.append(hit / len(gts))
            ps.append((sum(1 for q_ in pds if any(iou(g, q_) >= th for g in gts)) / len(pds)) if pds else 0.0)
        R = float(np.mean(rs)) if rs else 0.0
        P = float(np.mean(ps)) if ps else 0.0
        F1 = 2 * R * P / (R + P) if (R + P) > 0 else 0.0
        return R, P, F1, len(rs)

    acc = {}
    print(f"\n===== {a.cand} 精度 =====", flush=True)
    for th in (0.5, 0.75):
        R, P, F1, n = evaluate(th)
        acc[f"@{th}"] = {"R": R, "P": P, "DF1": F1, "n": n}
        print(f"  IoU {th}:  R={R:.4f}  P={P:.4f}  DF1={F1:.4f}  (n={n})", flush=True)
    Rs, Ps, Fs = [], [], []
    for th in np.arange(0.5, 0.96, 0.05):
        R, P, F1, _ = evaluate(float(th))
        Rs.append(R); Ps.append(P); Fs.append(F1)
    acc["@0.5:0.95"] = {"R": float(np.mean(Rs)), "P": float(np.mean(Ps)), "DF1": float(np.mean(Fs))}
    print(f"  R/P/DF1 @0.5:0.95:  R={acc['@0.5:0.95']['R']:.4f}  P={acc['@0.5:0.95']['P']:.4f}  "
          f"DF1={acc['@0.5:0.95']['DF1']:.4f}", flush=True)

    # ---- per-domain + rejection
    per_domain = {}
    for dom in sorted({p.get("domain", "") for p in preds}):
        items = [p for p in preds if p.get("domain", "") == dom]
        if not items:
            continue
        gtd = sum(1 for p in items if p.get("answer_boxes"))
        if gtd == 0:            # rejection: score = correctly outputs nothing
            empty = sum(1 for p in items if not p["extracted_predictions"])
            per_domain[dom] = {"n": len(items), "kind": "rejection",
                               "correct_reject": empty / len(items)}
            print(f"  [{dom}] n={len(items)} 正确拒答 {empty}/{len(items)} = {empty/len(items):.4f}", flush=True)
            continue
        R5, P5, F5, n5 = evaluate(0.5, items)
        Rs2, Ps2, Fs2 = [], [], []
        for th in np.arange(0.5, 0.96, 0.05):
            R_, P_, F_, _ = evaluate(float(th), items)
            Rs2.append(R_); Ps2.append(P_); Fs2.append(F_)
        per_domain[dom] = {"n": n5, "kind": "grounding",
                           "R@0.5": R5, "P@0.5": P5, "DF1@0.5": F5,
                           "DF1@0.5:0.95": float(np.mean(Fs2))}
        print(f"  [{dom}] n={n5} R@0.5 {R5:.4f} P@0.5 {P5:.4f} DF1@0.5 {F5:.4f} "
              f"DF1@.5:.95 {np.mean(Fs2):.4f}", flush=True)

    acc["per_domain"] = per_domain
    gr = [v for v in per_domain.values() if v["kind"] == "grounding"]
    if gr:
        acc["mean_over_domains"] = {
            "n_domains": len(gr),
            "R@0.5": float(np.mean([v["R@0.5"] for v in gr])),
            "P@0.5": float(np.mean([v["P@0.5"] for v in gr])),
            "DF1@0.5": float(np.mean([v["DF1@0.5"] for v in gr])),
            "DF1@0.5:0.95": float(np.mean([v["DF1@0.5:0.95"] for v in gr])),
        }
        print(f"  [域平均] DF1@0.5 {acc['mean_over_domains']['DF1@0.5']:.4f}  "
              f"DF1@.5:.95 {acc['mean_over_domains']['DF1@0.5:0.95']:.4f}", flush=True)

    json.dump({"summary": summary, "accuracy": acc, "preds": preds}, open(out_path, "w"), ensure_ascii=False)
    print(f"\n[bench] wrote {out_path} | {wall_all/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()








