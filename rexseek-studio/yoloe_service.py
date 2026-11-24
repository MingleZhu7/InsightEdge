#!/usr/bin/env python3
"""Persistent YOLOE-26L NPU service (board host).

POST /detect  {"image": "data:image/jpeg;base64,..."} -> candidates
"""
from __future__ import annotations

import base64, io, json, os, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from NOE_Engine import EngineInfer

# 模型路径可用环境变量覆盖：YOLOE_CIX=/path/to/model.cix
CIX = os.environ.get("YOLOE_CIX", "/home/cix/yoloe26l_person_split_i8.cix")
MODEL_NAME = os.path.basename(CIX)
S = 640
CONF = 0.25
NMS = 0.45
MAX_BOX = 30
COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def load_image(data_url: str) -> Image.Image:
    raw = data_url.split(",", 1)[1] if "," in data_url else data_url
    return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")


def letterbox(im: Image.Image):
    w, h = im.size
    scale = min(S / w, S / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = im.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (S, S), (114, 114, 114))
    px, py = (S - nw) // 2, (S - nh) // 2
    canvas.paste(resized, (px, py))
    x = np.asarray(canvas, np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(x), scale, px, py, w, h


def nms(boxes, scores, iou_thr=0.45):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


class Service:
    def __init__(self):
        print(f"[yoloe] model={CIX}", flush=True)
        self.engine = EngineInfer(CIX)
        # warmup
        dummy = np.zeros((1, 3, S, S), np.float32)
        for _ in range(2):
            self.engine.forward(dummy)

    def detect(self, tensor_b64: str) -> dict:
        t0 = time.perf_counter()
        raw = base64.b64decode(tensor_b64.split(",", 1)[-1])
        x = np.load(io.BytesIO(raw)).astype(np.float32)
        if x.ndim == 3:
            x = x[None]
        if x.shape != (1, 3, S, S):
            raise ValueError(f"tensor must be (1,3,{S},{S}), got {x.shape}")
        t_prep = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        outs = self.engine.forward(x)
        t_npu = (time.perf_counter() - t1) * 1000
        if not isinstance(outs, (list, tuple)):
            outs = [outs]
        arrs = [np.asarray(o).reshape(-1) for o in outs]
        box_t = next((a for a in arrs if a.size == 4 * 8400), None)
        score_t = next((a for a in arrs if a.size == 8400), None)
        proto = next((a for a in arrs if a.size == 32 * 160 * 160), None)
        if box_t is None or score_t is None:
            raise RuntimeError(f"unexpected YOLOE outputs: {[a.size for a in arrs]}")
        # 拆分输出：box [4,8400] (cx,cy,w,h 像素) / score [8400] (person 置信度, 已 sigmoid)
        xywh = box_t.reshape(4, 8400).T.astype(np.float32)
        scores = score_t.reshape(-1).astype(np.float32)
        cls_ids = np.zeros(len(scores), np.int64)
        keep = scores > CONF
        xywh, scores, cls_ids = xywh[keep], scores[keep], cls_ids[keep]
        cx, cy, bw, bh = xywh.T
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, S)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, S)
        # YOLOE 的类别头不可信：做 class-agnostic NMS，避免同一目标被多个错误类别重复保留
        out = []
        for i in nms(boxes, scores):
            bw = boxes[i][2] - boxes[i][0]
            bh = boxes[i][3] - boxes[i][1]
            if bw < 2 or bh < 2:
                continue
            # 去掉几乎覆盖整图的巨型碎片框
            if bw > S * 0.92 or bh > S * 0.92:
                continue
            c = int(cls_ids[i])
            out.append({
                "box": [round(float(v), 2) for v in boxes[i]],
                "score": round(float(scores[i]), 4),
                "label": COCO[c] if c < len(COCO) else str(c),
                "class_id": c,
            })
        out.sort(key=lambda r: -r["score"])
        out = out[:MAX_BOX]
        return {
            "coords": "640", "size": [S, S], "boxes": [o["box"] for o in out],
            "scores": [o["score"] for o in out], "labels": [o["label"] for o in out],
            "objects": out, "prep_ms": round(t_prep, 2), "npu_ms": round(t_npu, 2),
            "total_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    svc = None

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] " + fmt % args, flush=True)

    def _json(self, obj, code=200):
        raw = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path in ("/health", "/api/health"):
            return self._json({"ok": True, "service": "yoloe-26l-npu", "model": MODEL_NAME})
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path not in ("/detect", "/api/detect"):
            return self._json({"error": "not found"}, 404)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n).decode())
            data = payload.get("tensor")
            if not data:
                return self._json({"error": "tensor (base64 npy) required"}, 400)
            return self._json(self.svc.detect(data))
        except Exception as exc:
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


if __name__ == "__main__":
    H.svc = Service()
    srv = ThreadingHTTPServer(("0.0.0.0", 8090), H)
    print("yoloe-26l NPU service on :8090", flush=True)
    srv.serve_forever()
