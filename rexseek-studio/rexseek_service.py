#!/usr/bin/env python3
"""RexSeek-3B 真实推理服务（供 RexSeek Studio 的 proxy 模式调用）。

链路：
    GroundingDINO-tiny 出候选框  ->  RexSeek-3B 选 object index  ->  返回框 + 回答

启动（需要 GPU 与 rexseek conda 环境）：
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      /home/mingle/miniconda3/envs/rexseek/bin/python rexseek_service.py --port 9000

然后在另一个终端启动 UI：
    python3 server.py --engine proxy --endpoint http://127.0.0.1:9000

接口与 engine/proxy_engine.py 的约定一致：POST /api/chat
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# 模型与视觉塔都在本地，禁止联网兜底
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import (
    AutoConfig,  # noqa: E402
    TextIteratorStreamer,  # noqa: E402
    AutoModelForCausalLM,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    AutoTokenizer,
)

ROOT = Path(__file__).resolve().parent
WS = ROOT.parent
REPO = Path(os.environ.get("REXSEEK_REPO", WS / "dist/rexseek_hybrid_server/repo"))
MODEL_PATH = Path(os.environ.get("REXSEEK_MODEL", WS / "models/RexSeek-3B-local"))
GDINO_PATH = Path(os.environ.get("REXSEEK_GDINO", WS / "models/grounding-dino-tiny"))
CHAT_MODEL = Path(os.environ.get("REXSEEK_CHAT_MODEL", WS / "Project/board/models/Qwen__Qwen2.5-1.5B-Instruct"))
SAMPLES_PATH = Path(os.environ.get("REXSEEK_SAMPLES", str(ROOT / "data/samples.json")))
SAMPLES = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
SAMPLE_BY_ID = {s["id"]: s for s in SAMPLES["samples"]}
WEB = ROOT / "web"

sys.path.insert(0, str(ROOT))
from agent import RexSeekAgent
from session_store import SESSION_RE, SessionStore, new_session_id

sys.path.insert(0, str(REPO))
from rexseek.utils import xywh_to_xyxy, xyxy_to_xywh  # noqa: E402
from rexseek.utils.constants import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_OBJECT_FEATURE_TOKEN,
    DEFAULT_OBJECT_TOKEN,
)
from rexseek.utils.inference_utils import (  # noqa: E402
    expand2square,
    modify_processor_resolution,
    pad_boxes,
    prepare_input_for_rexseek,
    resize_boxes,
    tokenizer_image_object_token,
)

AGENT_SYSTEM = """你是 RexSeek Studio 的视觉对话助手，名字叫 RexSeek Agent。
你可以和用户自然对话，也可以根据系统提供的 RexSeek 检测结果，说明画面里目标的位置。
你看不到原始像素，只能依据检测结果回答；不要编造检测结果之外的目标、数量、位置或动作。
回答要简洁自然，通常 1-3 句中文；不要输出 <ground>、<objects>、<objN> 之类的标记。
如果用户只是打招呼、感谢或询问能力，直接自然回应，不要强行提定位。
你的能力包括：定位图片/视频中的目标并画框、回答目标的位置、进行简短对话。
打招呼时简短介绍自己是 RexSeek Agent，并说明可以帮用户在画面里定位目标。
回答定位结果时必须包含目标名称和位置，例如「穿蓝色裙子的女士在画面左侧」，不要只回答位置。"""

LABEL_ZH = {"person": "人", "tomato": "番茄", "bus": "大巴", "car": "汽车", "motorcycle": "摩托车", "truck": "卡车", "bicycle": "自行车"}

PALETTE = ["#35e0ff", "#ffb547", "#a78bfa", "#b6f36b", "#ff7a6b", "#7cc4ff", "#ff8fd8", "#ffd166"]
TEMPLATE = dict(
    SYSTEM="<|im_start|>system\n{system}<|im_end|>\n",
    INSTRUCTION="<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n",
    SUFFIX="<|im_end|>",
    SUFFIX_AS_EOS=True,
    SEP="\n",
    STOP_WORDS=["<|im_end|>", "<|endoftext|>"],
)


# --------------------------------------------------------------------------- utils
def _norm(text: str) -> str:
    text = (text or "").lower()
    return re.sub(r"[\s，。！？、,.!?~…]+", "", text)


def last_user_text(payload: dict) -> str:
    for msg in reversed(payload.get("messages") or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return (msg.get("content") or "").strip()
    return ""


def match_intent(sample: dict | None, query: str) -> dict | None:
    if not sample:
        return None
    q = _norm(query)
    best, best_score = None, 0
    for intent in sample.get("intents", []):
        score = sum(max(2, len(_norm(k))) for k in intent.get("keywords", []) if _norm(k) and _norm(k) in q)
        if score > best_score:
            best, best_score = intent, score
    return best


def nms(boxes, scores, iou_thr: float = 0.6):
    """极简 NMS，避免引入 torchvision 依赖。"""
    order = sorted(range(len(boxes)), key=lambda i: float(scores[i]), reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        bi = boxes[i]
        rest = []
        for j in order:
            bj = boxes[j]
            x1, y1 = max(bi[0], bj[0]), max(bi[1], bj[1])
            x2, y2 = min(bi[2], bj[2]), min(bi[3], bj[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_i = max(0.0, bi[2] - bi[0]) * max(0.0, bi[3] - bi[1])
            area_j = max(0.0, bj[2] - bj[0]) * max(0.0, bj[3] - bj[1])
            union = area_i + area_j - inter
            if union > 0 and inter / union <= iou_thr:
                rest.append(j)
        order = rest
    return keep


def spatial(box) -> str:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    h = "左侧" if cx < 0.34 else ("右侧" if cx > 0.66 else "中部")
    v = "偏上" if cy < 0.34 else ("偏下" if cy > 0.66 else "中部")
    if h == "中部" and v == "中部":
        return "画面中央"
    if h == "中部":
        return f"画面{v}"
    if v == "中部":
        return f"画面{h}"
    return f"画面{h}{v}"


def decode_data_url(data_url: str) -> Image.Image:
    if not data_url:
        raise ValueError("empty data_url")
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    return Image.open(io.BytesIO(raw)).convert("RGB")


# --------------------------------------------------------------------------- runtime
class Runtime:
    def __init__(self):
        self.lock = threading.RLock()
        t0 = time.time()
        self.yoloe_url = os.environ.get("YOLOE_URL") or ""
        self.yoloe_score_min = float(os.environ.get("YOLOE_SCORE_MIN", "0.6"))
        if self.yoloe_url:
            print(f"[rexseek] using YOLOE-26L NPU at {self.yoloe_url}", flush=True)
            self.gdino = None
            self.gdino_processor = None
        else:
            print("[rexseek] loading GroundingDINO-tiny ...", flush=True)
            self.gdino_processor = AutoProcessor.from_pretrained(str(GDINO_PATH))
            self.gdino = AutoModelForZeroShotObjectDetection.from_pretrained(str(GDINO_PATH)).to("cuda").eval()
        self._gdino_lock = threading.Lock()

        print("[rexseek] loading RexSeek-3B ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), use_fast=False)
        for token in [f"<obj{i}>" for i in range(100)] + ["<ground>", "</ground>", "<objects>", "</objects>"]:
            if token not in self.tokenizer.get_vocab():
                self.tokenizer.add_tokens([token], special_tokens=True)
        self.model = self._load_rexseek(str(MODEL_PATH)).eval()
        if os.environ.get("REXSEEK_COMPILE") == "1":
            try:
                self.model.vision_tower = torch.compile(self.model.vision_tower, mode="default")
                self.model.vision_tower_aux = torch.compile(self.model.vision_tower_aux, mode="default")
                print("[rexseek] compiled CLIP / ConvNeXt towers", flush=True)
            except Exception as exc:
                print(f"[rexseek] compile skipped: {exc}", flush=True)
        self.processor = self.model.get_vision_tower().image_processor
        self.processor, self.crop_size_raw = modify_processor_resolution(self.processor)

        self.qwen_url = os.environ.get("QWEN_URL") or ""
        self._tr_cache: dict = {}
        if self.qwen_url:
            print(f"[rexseek] using llama.cpp Qwen3.5-0.8B CPU at {self.qwen_url}", flush=True)
            self.chat_tokenizer = None
            self.chat_model = None
        else:
            print("[rexseek] loading Qwen2.5-1.5B agent ...", flush=True)
            self.chat_tokenizer = AutoTokenizer.from_pretrained(str(CHAT_MODEL))
            self.chat_model = AutoModelForCausalLM.from_pretrained(
                str(CHAT_MODEL), torch_dtype=torch.float16, device_map="cuda"
            ).eval()
        self.device = torch.cuda.get_device_name(0)
        self.use_vllm = os.environ.get("REXSEEK_USE_VLLM") == "1"
        self.vllm = None
        self.vllm_sp = None
        self.EmbedsPrompt = None
        if self.use_vllm:
            from vllm import LLM, SamplingParams
            from vllm.inputs import EmbedsPrompt
            self.EmbedsPrompt = EmbedsPrompt
            lm_path = os.environ.get("REXSEEK_LM", "/models/Qwen2.5-3B-RexSeek-LM")
            print(f"[rexseek] loading RexSeek LM vLLM {lm_path} ...", flush=True)
            self.vllm = LLM(
                model=lm_path, enable_prompt_embeds=True, dtype="bfloat16",
                gpu_memory_utilization=float(os.environ.get("VLLM_GPU_UTIL", "0.42")),
                max_model_len=4096, max_num_seqs=8, max_num_batched_tokens=1024,
                compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"},
            )
            self.vllm_sp = SamplingParams(temperature=0.0, max_tokens=24)
            print("[rexseek] vLLM LM ready", flush=True)
        self.session_store = SessionStore(ROOT / "data" / "sessions.db")
        try:
            self.session_store.cleanup()
        except Exception:
            pass
        self.agent = RexSeekAgent(self, SAMPLE_BY_ID, {}, self.session_store)
        # 同图缓存：解码图 / GDINO 候选框 / 视觉塔输出
        self._image_cache = {}
        self._candidate_cache = {}
        self._visual_cache = {}
        self._prep_cache = {}
        self._current_image_key = None
        self._prof = {}
        self._install_prof("clip", self.model.vision_tower)
        self._install_prof("cn", self.model.vision_tower_aux)
        self._install_prof("proj", self.model.mm_projector)
        print("[prof] hooks installed", flush=True)
        self._orig_encode_images = self.model.encode_images

        def _cached_encode_images(images, images_aux):
            key = self._current_image_key
            if key and key in self._visual_cache:
                return self._visual_cache[key]
            out = self._orig_encode_images(images, images_aux)
            if key:
                if len(self._visual_cache) >= 8:
                    self._visual_cache.pop(next(iter(self._visual_cache)))
                self._visual_cache[key] = out
            return out

        self.model.encode_images = _cached_encode_images
        print(f"[rexseek] ready in {time.time() - t0:.1f}s on {self.device}", flush=True)

    # ------------------------------------------------------------------ models
    @staticmethod
    def _wanted_labels(prompt: str):
        q = (prompt or "").lower()
        wanted = set()
        if any(w in q for w in ("person", "woman", "man", "people", "speaker")):
            wanted.add("person")
        for label in ("tomato", "bus", "car", "truck", "motorcycle", "bicycle", "dog", "cat",
                      "bird", "horse", "elephant", "bear", "chair", "bottle", "cup"):
            if label in q:
                wanted.add(label)
        if "tomatoes" in q:
            wanted.add("tomato")
        if "vehicles" in q:
            wanted.update(("car", "truck", "bus", "motorcycle"))
        return wanted or None

    def _yoloe_candidates(self, image: Image.Image, prompt: str, box_threshold: float):
        """调用 host NPU 的 YOLOE-26L；输入 640x640 letterbox tensor，返回原图坐标。"""
        w, h = image.size
        scale = min(640 / w, 640 / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = image.resize((nw, nh), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (640, 640), (114, 114, 114))
        px, py = (640 - nw) // 2, (640 - nh) // 2
        canvas.paste(resized, (px, py))
        x = np.asarray(canvas, np.float32).transpose(2, 0, 1)[None] / 255.0
        buf = io.BytesIO()
        np.save(buf, np.ascontiguousarray(x))
        payload = json.dumps({"tensor": base64.b64encode(buf.getvalue()).decode()}).encode()
        req = urllib.request.Request(self.yoloe_url.rstrip("/") + "/detect", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        boxes, scores, labels = [], [], []
        for obj in data.get("objects", []):
            x1, y1, x2, y2 = obj["box"]
            boxes.append([
                max(0.0, min(w, (x1 - px) / scale)),
                max(0.0, min(h, (y1 - py) / scale)),
                max(0.0, min(w, (x2 - px) / scale)),
                max(0.0, min(h, (y2 - py) / scale)),
            ])
            scores.append(float(obj["score"]))
            labels.append(str(obj["label"]).lower())
        wanted = None if os.environ.get("YOLOE_CLASS_AGNOSTIC") == "1" else self._wanted_labels(prompt)
        if wanted:
            keep = [i for i, label in enumerate(labels) if label in wanted]
            boxes = [boxes[i] for i in keep]
            scores = [scores[i] for i in keep]
            labels = [labels[i] for i in keep]
        if not boxes:
            return torch.zeros((0, 4)), torch.zeros((0,)), []
        keep_score = [i for i, score in enumerate(scores) if score >= self.yoloe_score_min]
        boxes = [boxes[i] for i in keep_score]
        scores = [scores[i] for i in keep_score]
        labels = [labels[i] for i in keep_score]
        if not boxes:
            return torch.zeros((0, 4)), torch.zeros((0,)), []
        b = torch.tensor(boxes, dtype=torch.float32)
        sc = torch.tensor(scores, dtype=torch.float32)
        keep = sc >= float(box_threshold)
        return b[keep], sc[keep], [l for l, k in zip(labels, keep.tolist()) if k]

    def _ensure_gdino(self):
        """开放词表（非 person）提示需要 GDINO；按需加载，避免 YOLOE 模式下常驻。"""
        if self.gdino_processor is not None:
            return
        with self._gdino_lock:
            if self.gdino_processor is not None:
                return
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            print("[rexseek] lazy-loading GroundingDINO-tiny ...", flush=True)
            self.gdino_processor = AutoProcessor.from_pretrained(str(GDINO_PATH))
            self.gdino = AutoModelForZeroShotObjectDetection.from_pretrained(
                str(GDINO_PATH)).to("cuda").eval()
            print("[rexseek] GroundingDINO ready", flush=True)

    def _person_prompt(self, prompt: str) -> bool:
        """只有"纯 person"提示才走 NPU YOLOE；含其它类名（tomato/helmet/...）时走 GDINO。"""
        toks = [t for t in re.split(r"[^a-zA-Z]+", (prompt or "").lower()) if t]
        allow = {"person", "persons", "people", "worker", "workers", "object", "objects", "human", "humans"}
        return bool(toks) and all(t in allow for t in toks)

    def candidates(self, image: Image.Image, prompt: str, roi=None, box_threshold: float = 0.2):
        image_key = getattr(image, "_rexseek_hash", None)
        cache_key = (image_key, prompt, tuple(roi) if roi else None, round(float(box_threshold), 3))
        if image_key and cache_key in self._candidate_cache:
            return self._candidate_cache[cache_key]
        _t_cand = time.perf_counter()
        use_yoloe = bool(self.yoloe_url) and self._person_prompt(prompt)
        if use_yoloe:
            boxes, scores, labels = self._yoloe_candidates(image, prompt, box_threshold)
        else:
            self._ensure_gdino()
            inputs = self.gdino_processor(images=image, text=prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = self.gdino(**inputs)
            try:
                res = self.gdino_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=box_threshold,
                    text_threshold=0.2,
                    target_sizes=[image.size[::-1]],
                )[0]
            except TypeError:
                # transformers 5.x renamed box_threshold -> threshold
                res = self.gdino_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=box_threshold,
                    text_threshold=0.2,
                    target_sizes=[image.size[::-1]],
                )[0]
            boxes = res["boxes"].float().cpu()
            scores = res["scores"].float().cpu()
            labels = [str(x).lower() for x in (res.get("labels") or ["object"] * len(boxes))]
        self._prof["cand"] = (time.perf_counter() - _t_cand) * 1000
        if len(boxes) == 0:
            return boxes, scores, labels

        w, h = image.size
        idx = list(range(len(boxes)))
        if roi:
            idx = [
                i for i in idx
                if (boxes[i][0] + boxes[i][2]) / 2 / w >= roi[0]
                and (boxes[i][1] + boxes[i][3]) / 2 / h >= roi[1]
            ]
        if use_yoloe:
            # YOLOE 标签是 COCO 闭集且不可信，按全类做 NMS
            keep = nms([boxes[i].tolist() for i in idx], [scores[i].tolist() for i in idx], 0.4)
            keep = [idx[j] for j in keep]
        else:
            keep = []
            for label in set(labels[i] for i in idx):
                sub = [i for i in idx if labels[i] == label]
                sub_keep = nms([boxes[i].tolist() for i in sub], [scores[i].tolist() for i in sub], 0.4)
                keep.extend(sub[j] for j in sub_keep)
        keep.sort(key=lambda i: -float(scores[i]))
        result = (boxes[keep], scores[keep], [labels[i] for i in keep])
        if image_key:
            if len(self._candidate_cache) >= 32:
                self._candidate_cache.pop(next(iter(self._candidate_cache)))
            self._candidate_cache[cache_key] = result
        return result

    def prepare_cached(self, image: Image.Image, boxes, question: str) -> dict:
        """prepare_input_for_rexseek 的图像侧产物按 image hash 缓存。"""
        key = getattr(image, "_rexseek_hash", None)
        base = self._prep_cache.get(key) if key else None
        if base is None:
            ori_w, ori_h = image.size
            square = expand2square(image, tuple(int(x * 255) for x in self.processor.image_mean))
            pad_w, pad_h = square.size
            image_aux = self.processor.preprocess(square, return_tensors="pt")["pixel_values"][0]
            resize_h, resize_w = image_aux.shape[-2:]
            pixel_values_aux = image_aux.unsqueeze(0)
            image_336 = torch.nn.functional.interpolate(
                image_aux.clone()[None],
                size=[self.crop_size_raw["height"], self.crop_size_raw["width"]],
                mode="bilinear",
                align_corners=False,
            )[0]
            base = {
                "ori_w": ori_w, "ori_h": ori_h,
                "pad_w": pad_w, "pad_h": pad_h,
                "resize_h": resize_h, "resize_w": resize_w,
                "pixel_values_aux": pixel_values_aux,
                "pixel_values": image_336.unsqueeze(0),
            }
            if key:
                if len(self._prep_cache) >= 16:
                    self._prep_cache.pop(next(iter(self._prep_cache)))
                self._prep_cache[key] = base

        data = dict(base)
        bbox = xyxy_to_xywh(boxes)
        bbox = pad_boxes(bbox, (base["ori_w"], base["ori_h"]))
        bbox = resize_boxes(bbox, (base["pad_w"], base["pad_h"]), (base["resize_h"], base["resize_w"]))
        data["gt_boxes"] = torch.tensor(xywh_to_xyxy(bbox)).unsqueeze(0)

        total_num_boxes = len(bbox)
        obj_tokens = [DEFAULT_OBJECT_TOKEN.replace("<i>", str(i)) for i in range(total_num_boxes)]
        obj_tokens = DEFAULT_OBJECT_FEATURE_TOKEN.join(obj_tokens) + DEFAULT_OBJECT_FEATURE_TOKEN
        question = question.replace(DEFAULT_IMAGE_TOKEN, "")
        question = DEFAULT_IMAGE_TOKEN + "\n" + obj_tokens + "\n" + question
        inputs = TEMPLATE["INSTRUCTION"].format(input=question, round=1)
        data["input_ids"] = torch.tensor(tokenizer_image_object_token(inputs, self.tokenizer)).unsqueeze(0)
        return data

    def _generate_answer(self, data, max_tokens: int = 128) -> str:
        if self.vllm is not None:
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                _, _, _, _, embeds, _ = self.model.prepare_inputs_labels_for_multimodal(
                    data["input_ids"].to("cuda"),
                    None, None, None, None,
                    data["pixel_values"].to("cuda", dtype=torch.bfloat16),
                    data["pixel_values_aux"].to("cuda", dtype=torch.bfloat16),
                    data["gt_boxes"].to("cuda", dtype=torch.bfloat16),
                )
            outs = self.vllm.generate([self.EmbedsPrompt(prompt_embeds=embeds[0].contiguous())], self.vllm_sp)
            return self.tokenizer.decode(list(outs[0].outputs[0].token_ids), skip_special_tokens=False).strip()
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            out = self.model.generate(
                data["input_ids"].to("cuda"),
                pixel_values=data["pixel_values"].to("cuda", dtype=torch.float16),
                pixel_values_aux=data["pixel_values_aux"].to("cuda", dtype=torch.float16),
                gt_boxes=data["gt_boxes"].to("cuda", dtype=torch.float16),
                do_sample=False,
                max_new_tokens=max_tokens,
                temperature=0.0,
                top_p=None,
            )
        return self.tokenizer.batch_decode(out, skip_special_tokens=False)[0].strip()

    def vqa(self, image: Image.Image, boxes, question: str, max_tokens: int = 96) -> str:
        """让 RexSeek 直接看图回答开放问题（不只是输出目标编号）。"""
        self._current_image_key = getattr(image, "_rexseek_hash", None)
        data = self.prepare_cached(image, boxes, question)
        return self._generate_answer(data, max_tokens)

    def ground(self, image: Image.Image, boxes, query: str, max_tokens: int = 128) -> str:
        question = f"Please detect {query} in this image. Answer the question with object indexes."
        self._current_image_key = getattr(image, "_rexseek_hash", None)
        data = self.prepare_cached(image, boxes, question)
        if self.vllm is not None:
            for k in ("clip", "cn", "proj", "cand"):
                self._prof.pop(k, None)
            _t0 = time.perf_counter()
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                _, _, _, _, embeds, _ = self.model.prepare_inputs_labels_for_multimodal(
                    data["input_ids"].to("cuda"),
                    None, None, None, None,
                    data["pixel_values"].to("cuda", dtype=torch.bfloat16),
                    data["pixel_values_aux"].to("cuda", dtype=torch.bfloat16),
                    data["gt_boxes"].to("cuda", dtype=torch.bfloat16),
                )
            _t_vis = (time.perf_counter() - _t0) * 1000
            _t1 = time.perf_counter()
            outs = self.vllm.generate([self.EmbedsPrompt(prompt_embeds=embeds[0].contiguous())], self.vllm_sp)
            _t_llm = (time.perf_counter() - _t1) * 1000
            print(f"[prof] cand={self._prof.get('cand', 0):.1f} clip={self._prof.get('clip', 0):.1f} "
                  f"cn={self._prof.get('cn', 0):.1f} proj={self._prof.get('proj', 0):.1f} "
                  f"vis_total={_t_vis:.1f} llm3b={_t_llm:.1f} nboxes={data['gt_boxes'].shape[1]}", flush=True)
            return self.tokenizer.decode(list(outs[0].outputs[0].token_ids), skip_special_tokens=False).strip()
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            out = self.model.generate(
                data["input_ids"].to("cuda"),
                pixel_values=data["pixel_values"].to("cuda", dtype=torch.float16),
                pixel_values_aux=data["pixel_values_aux"].to("cuda", dtype=torch.float16),
                gt_boxes=data["gt_boxes"].to("cuda", dtype=torch.float16),
                do_sample=False,
                max_new_tokens=max_tokens,
                temperature=0.0,
                top_p=None,
            )
        return self.tokenizer.batch_decode(out, skip_special_tokens=False)[0].strip()

    def _llm_chat(self, messages: list[dict], max_new_tokens: int = 96, on_token=None) -> str:
        url = self.qwen_url.rstrip("/")
        if not url.endswith("/v1/chat/completions"):
            url += "/v1/chat/completions"
        body = {
            "model": "qwen35-0.8b-cpu",
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": 0,
            "stream": bool(on_token),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode(),
                                     headers={"Content-Type": "application/json"})
        if on_token:
            parts = []
            with urllib.request.urlopen(req, timeout=300) as resp:
                for line in resp:
                    line = line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        delta = json.loads(raw)["choices"][0].get("delta", {}).get("content") or ""
                    except Exception:
                        delta = ""
                    if delta:
                        parts.append(delta)
                        on_token(delta)
            return "".join(parts).strip()
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode())
        msg = data["choices"][0]["message"]
        return (msg.get("content") or msg.get("reasoning_content") or "").strip()

    def chat(self, history: list[dict], context: str, max_new_tokens: int = 96, on_token=None) -> str:
        system = AGENT_SYSTEM + ("\n\n" + context if context else "")
        msgs = [{"role": "system", "content": system}]
        for m in history[-8:]:
            role, content = m.get("role"), (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})
        if len(msgs) == 1:
            return ""
        if self.qwen_url:
            reply = self._llm_chat(msgs, max_new_tokens, on_token)
            return re.sub(r"</?ground>|</?objects>|<obj\d+>", "", reply).strip()
        text = self.chat_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.chat_tokenizer(text, return_tensors="pt").to("cuda")
        kwargs = dict(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=self.chat_tokenizer.eos_token_id,
        )
        if on_token:
            streamer = TextIteratorStreamer(self.chat_tokenizer, skip_prompt=True, skip_special_tokens=True)
            kwargs["streamer"] = streamer

            def _gen():
                with torch.inference_mode():
                    self.chat_model.generate(**kwargs)

            thread = threading.Thread(target=_gen, daemon=True)
            thread.start()
            parts = []
            for chunk in streamer:
                if chunk:
                    parts.append(chunk)
                    on_token(chunk)
            thread.join(timeout=30)
            reply = "".join(parts).strip()
        else:
            with torch.inference_mode():
                out = self.chat_model.generate(**kwargs)
            reply = self.chat_tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        reply = re.sub(r"</?ground>|</?objects>|<obj\d+>", "", reply).strip()
        return reply

    def extract_target(self, query: str) -> str:
        """让 Qwen 把中文问题里的目标翻成最简单的英文类别词，供 GDINO 出候选框。"""
        cache_key = ("noun", query)
        if cache_key in self._tr_cache:
            return self._tr_cache[cache_key]
        msgs = [
            {"role": "system", "content": "把用户问题里的目标物体翻译成最具体的英文名词，只输出这个名词，不要类别、不要解释。例如：大象→elephant，狗→dog，穿蓝裙子的女士→woman。"},
            {"role": "user", "content": query},
        ]
        if self.qwen_url:
            raw = self._llm_chat(msgs, 24)
        else:
            text = self.chat_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = self.chat_tokenizer(text, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                out = self.chat_model.generate(
                    **enc, max_new_tokens=24, do_sample=False, pad_token_id=self.chat_tokenizer.eos_token_id
                )
            raw = self.chat_tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        cleaned = re.sub(r"[^a-zA-Z .]", "", raw).strip(" .")
        cleaned = " ".join(cleaned.split())[:80]
        if len(self._tr_cache) < 256:
            self._tr_cache[cache_key] = cleaned
        return cleaned

    def extract_refexp(self, query: str) -> str:
        """把中文指代翻成"保留修饰"的英文指代表达，专门给 RexSeek 用。

        extract_target() 是给 GroundingDINO 出候选框用的名词（会把"蓝色衣服"抹掉），
        不能拿它当 RexSeek 的查询，否则"穿蓝色衣服的人"会退化成"woman"，两个人都被选中。
        """
        cache_key = ("ref", query)
        if cache_key in self._tr_cache:
            return self._tr_cache[cache_key]
        msgs = [
            {"role": "system", "content": "把用户的中文指代翻译成简短英文指代表达，必须保留颜色、穿着、动作、位置、"
                                          "人物关系等全部修饰。只输出英文短语，不要解释。例如："
                                          "穿蓝色衣服的人→person wearing blue clothes；"
                                          "左边戴帽子的人→person on the left wearing a hat；"
                                          "成熟的番茄→ripe tomatoes；画面中的女人→the woman in the image。"},
            {"role": "user", "content": query},
        ]
        if self.qwen_url:
            raw = self._llm_chat(msgs, 48)
        else:
            text = self.chat_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = self.chat_tokenizer(text, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                out = self.chat_model.generate(
                    **enc, max_new_tokens=48, do_sample=False, pad_token_id=self.chat_tokenizer.eos_token_id
                )
            raw = self.chat_tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        cleaned = re.sub(r"[^a-zA-Z .,'\-]", "", raw).strip(" .")
        cleaned = " ".join(cleaned.split())[:120]
        if len(self._tr_cache) < 256:
            self._tr_cache[cache_key] = cleaned
        return cleaned

    # ----------------------------------------------------------------- routing
    @staticmethod
    def route(query: str, intent: dict | None) -> str:
        q = _norm(query)
        if not q:
            return "chat"
        if any(w in q for w in ("你好", "您好", "hi", "hello", "嗨", "在吗", "早上好", "晚上好")):
            return "chat"
        if any(w in q for w in ("谢谢", "感谢", "thanks", "thankyou")):
            return "chat"
        if any(w in q for w in ("你能做什么", "你会什么", "怎么用", "帮助", "help", "你是谁", "介绍一下你自己", "自我介绍")):
            return "chat"
        if any(w in q for w in ("描述", "介绍一下", "画面里有什么", "有什么东西", "整体", "总结一下")):
            return "describe"
        if intent:
            return "grounding"
        cues = ("在哪", "哪里", "找出", "找一找", "定位", "框", "哪些", "几个", "几颗", "多少",
                "谁", "哪位", "哪个", "左边", "右边", "上面", "下面", "穿", "颜色", "背景",
                "前景", "目标", "人", "车", "番茄", "大巴", "摩托", "有没有", "有吗", "吗")
        if any(w in q for w in cues):
            return "grounding"
        return "chat"

    def classify_tool(self, query: str) -> str:
        """Qwen 受约束工具分类（只输出一个词）。"""
        system = (
            "你是工具调度器。根据用户问题只输出一个工具名，不要解释："
            "ground=定位某个目标；count=统计某类目标数量；describe=概述画面；"
            "relative=结合上文看某目标的左/右/上/下方；resolve=解析“她/他/它/第二个/刚才那个”；"
            "clarify=信息不足需要追问；chat=打招呼/闲聊/能力询问。"
        )
        examples = [
            ("画面里有几个人？", "count"),
            ("找出穿蓝裙子的女士", "ground"),
            ("描述一下这张图", "describe"),
            ("她右边是谁？", "relative"),
            ("她是谁？", "resolve"),
            ("你好", "chat"),
        ]
        msgs = [{"role": "system", "content": system}]
        for u, a in examples:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": query})
        if self.qwen_url:
            raw = self._llm_chat(msgs, 6).strip().lower()
        else:
            text = self.chat_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = self.chat_tokenizer(text, return_tensors="pt").to("cuda")
            with self.lock, torch.inference_mode():
                out = self.chat_model.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=self.chat_tokenizer.eos_token_id)
            raw = self.chat_tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
        word = re.sub(r"[^a-z]", "", raw)
        return word

    def tool_call(self, query: str, context: str = "") -> dict | None:
        """Qwen 输出结构化 tool call（精简目录），Agent 侧再做 schema 校验与护栏。"""
        system = (
            "你是 RexSeek Agent 的工具选择器。只输出一个 JSON：{\"tool\":\"...\",\"args\":{...}}\n"
            "工具：\n"
            "guard_events 台账/几次/历史 {}\n"
            "guard_segment 某起事件的起止/时长/人数/概况 {}\n"
            "guard_check 现在区域里有没有人 {}\n"
            "zone_query 某人/某描述在不在危险区或值守区 {target}\n"
            "ground 找目标/在哪/哪个/是谁 {target}\n"
            "count 数数量 {category}\n"
            "ppe_check 戴没戴安全帽/反光背心 {query}\n"
            "attribute 那个人穿什么颜色 {question}\n"
            "describe 描述整张画面 {}\n"
            "relative 有上文主语时问左右上下 {direction,target,ordinal}\n"
            "resolve 指代（她/他/刚才那个）{phrase}\n"
            "clarify 有指代但没有上文主语 {question}\n"
            "chat 寒暄/确认/闲聊 {}\n"
            "规则：装备→ppe_check；事件时间/人数→guard_segment；在不在区域→zone_query；找人→ground；闲聊→chat。\n"
            "上下文里【事件】=正在看某起事件，【画面】=视频或图片。"
        )
        examples = [
            ("今天有几起告警？", '{"tool":"guard_events","args":{}}'),
            ("这起事件几点开始、几点结束？", '{"tool":"guard_segment","args":{}}'),
            ("这个人戴安全帽了吗？", '{"tool":"ppe_check","args":{"query":"这个人戴安全帽了吗？"}}'),
            ("穿蓝色衣服的人在不在危险区？", '{"tool":"zone_query","args":{"target":"穿蓝色衣服的人"}}'),
            ("画面里的人在哪里？", '{"tool":"ground","args":{"target":"画面里的人"}}'),
            ("你好", '{"tool":"chat","args":{}}'),
        ]
        msgs = [{"role": "system", "content": system}]
        for u, a in examples:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        user = (f"上下文：{context}\n用户问题：{query}" if context else query)
        msgs.append({"role": "user", "content": user})
        if self.qwen_url:
            raw = self._llm_chat(msgs, 120).strip()
        else:
            text = self.chat_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = self.chat_tokenizer(text, return_tensors="pt").to("cuda")
            with self.lock, torch.inference_mode():
                out = self.chat_model.generate(
                    **enc, max_new_tokens=120, do_sample=False, pad_token_id=self.chat_tokenizer.eos_token_id
                )
            raw = self.chat_tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        tool = data.get("tool")
        args = data.get("args") or {}
        if tool not in ("ground", "count", "describe", "relative", "resolve", "clarify", "chat"):
            return None
        if not isinstance(args, dict):
            args = {}
        return {"tool": tool, "args": args}

    # ------------------------------------------------------------------ reply
    def respond(self, payload: dict) -> dict:
        return self.agent.handle(payload)

    def _load_rexseek(self, path: str):
        """优先 from_pretrained；板卡 transformers 5.x 的 meta 初始化限制下退回 from_config。"""
        if os.environ.get("REXSEEK_FROM_CONFIG") != "1":
            try:
                return AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16)
            except RuntimeError as exc:
                if "meta device" not in str(exc):
                    raise
                print("[rexseek] from_pretrained hit meta-device guard; falling back to from_config", flush=True)
        from safetensors.torch import load_file
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        with torch.device("cuda"):
            model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, dtype=torch.bfloat16)
        def fix(sd):
            return {k.replace(".vision_model.", ".", 1): v for k, v in sd.items()}
        for f in sorted(Path(path).glob("model*.safetensors")):
            model.load_state_dict(fix(load_file(str(f))), strict=False, assign=True)
        return model.to("cuda").eval()

    def _install_prof(self, name, mod):
        def pre(m, inp):
            torch.cuda.synchronize()
            self._prof[name + "_t0"] = time.perf_counter()

        def post(m, inp, out):
            torch.cuda.synchronize()
            t0 = self._prof.get(name + "_t0")
            if t0 is not None:
                self._prof[name] = self._prof.get(name, 0.0) + (time.perf_counter() - t0) * 1000

        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)

    def _load_image(self, media: dict, sample: dict | None) -> Image.Image:
        # 视频必须使用当前帧；图片样本可用磁盘原文件保证 hash 稳定。
        data_url = media.get("data_url")
        if sample and sample.get("kind") == "video" and data_url:
            raw = base64.b64decode(data_url.split(",", 1)[1]) if "," in data_url else base64.b64decode(data_url)
        elif sample:
            path = WEB / (sample.get("poster") or sample["file"]) if sample.get("kind") == "video" else WEB / sample["file"]
            raw = path.read_bytes()
        else:
            if not data_url:
                raise ValueError("media.data_url or valid sample_id is required")
            raw = base64.b64decode(data_url.split(",", 1)[1]) if "," in data_url else base64.b64decode(data_url)
        key = hashlib.sha256(raw).hexdigest()
        if key in self._image_cache:
            return self._image_cache[key]
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        image._rexseek_hash = key
        if len(self._image_cache) >= 16:
            self._image_cache.pop(next(iter(self._image_cache)))
        self._image_cache[key] = image
        return image

    def index_media(self, media: dict) -> dict:
        """把一张图预处理 + GDINO + 视觉塔结果预热进缓存。"""
        sample = SAMPLE_BY_ID.get(media.get("sample_id") or "")
        t0 = time.perf_counter()
        image = self._load_image(media, sample)
        key = getattr(image, "_rexseek_hash", None)
        prompt = (sample or {}).get("gdino_prompt") or media.get("gdino_prompt") or "person. object."
        roi = (sample or {}).get("roi")
        threshold = (sample or {}).get("box_threshold", 0.2)
        already = bool(key and key in self._visual_cache and any(k[0] == key for k in self._candidate_cache))
        with self.lock:
            # 先准备图像 tensor；YOLOE NPU 和 ViT/ConvNeXt GPU 并行
            data = self.prepare_cached(image, torch.tensor([[0.0, 0.0, 16.0, 16.0]]), "dummy")
            self._current_image_key = key

            def _visual():
                with torch.inference_mode():
                    self.model.encode_images(
                        data["pixel_values"].to("cuda", dtype=torch.bfloat16),
                        data["pixel_values_aux"].to("cuda", dtype=torch.bfloat16),
                    )

            import threading
            vt = threading.Thread(target=_visual, daemon=True)
            vt.start()
            boxes, scores, labels = self.candidates(image, prompt, roi, threshold)
            vt.join(timeout=120)
        print(
            f"[index] cached={already} ms={int((time.perf_counter() - t0) * 1000)} "
            f"candidates={len(boxes)} image={key[:12]}",
            flush=True,
        )
        return {
            "image_id": key,
            "cached": already,
            "index_ms": int((time.perf_counter() - t0) * 1000),
            "candidates": int(len(boxes)),
            "size": [image.width, image.height],
        }


# --------------------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    server_version = "RexSeekService/1.0"
    protocol_version = "HTTP/1.1"
    runtime: Runtime = None  # type: ignore

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path in ("/health", "/api/health"):
            return self._json(
                {"ok": True, "engine": "rexseek", "model": "RexSeek-3B + Qwen2.5-1.5B",
                 "chat_model": ("Qwen3.5-0.8B-Q4_K_M (CPU)" if self.runtime.qwen_url else "Qwen2.5-1.5B-Instruct"), "llm": ("vLLM" if self.runtime.use_vllm else "HF"), "demo": False, "device": self.runtime.device}
            )
        if path.startswith("/api/session/"):
            sid = path[len("/api/session/"):]
            if not SESSION_RE.match(sid):
                return self._json({"error": "invalid session id"}, 400)
            data = self.runtime.session_store.load(sid)
            if not data:
                return self._json({"session_id": sid, "exists": False}, 404)
            return self._json({
                "session_id": sid,
                "exists": True,
                "turns": len(data.get("history") or []) // 2,
                "updated": data.get("updated"),
                "image_key": data.get("image_key"),
                "history": (data.get("history") or [])[-6:],
            })
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urllib.parse.urlsplit(self.path).path
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return self._json({"error": "empty body"}, 400)
        if length > 32 * 1024 * 1024:
            return self._json({"error": "payload too large"}, 413)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return self._json({"error": "invalid json"}, 400)

        if path == "/api/session":
            sid = str(payload.get("session_id") or "")
            if not SESSION_RE.match(sid):
                sid = new_session_id()
            existing = self.runtime.session_store.load(sid)
            if not existing:
                self.runtime.session_store.save(sid, {
                    "last_objects": [], "last_subject": None, "last_query": "",
                    "history": [], "image_key": None,
                })
            return self._json({"session_id": sid, "resumed": bool(existing), "turns": len((existing or {}).get("history") or []) // 2})

        if path == "/api/index":
            try:
                return self._json(self.runtime.index_media(payload.get("media") or payload))
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                return self._json({"error": str(exc)}, 500)

        if path == "/api/chat/stream":
            return self._stream(payload)

        if path != "/api/chat":
            return self._json({"error": "not found"}, 404)

        started = time.perf_counter()
        try:
            result = self.runtime.respond(payload)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            return self._json({"error": str(exc)}, 500)
        result.setdefault("meta", {})["server_latency_ms"] = int((time.perf_counter() - started) * 1000)
        return self._json(result)

    def _stream(self, payload):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def emit(event):
            try:
                self.wfile.write(("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                raise

        try:
            self.runtime.agent.handle(payload, emit)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            try:
                emit({"type": "error", "error": str(exc)})
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    Handler.runtime = Runtime()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"[rexseek] serving on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("bye")


if __name__ == "__main__":
    main()
