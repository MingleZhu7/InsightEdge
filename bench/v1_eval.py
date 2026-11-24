"""v1 (GroundingDINO/GD-candidates + ViT-B + ConvNeXt-B + Qwen2.5-3B) accuracy on the board.

Same feature files produced by e2e/{prep_img,vit_stage,cn_stage}.py; this script does the RoI /
projector / prompt-embed / vLLM decode step and dumps per-ref predictions so we can compute
R/P/F1 with the same evaluator as the v3 deployment.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import EmbedsPrompt

MODEL = "/models/Qwen2.5-3B-RexSeek-LM"
OUT = "/cix/v1/out"
ANN = "/cix/v1/annotations.jsonl"
PROJ = "/cix/e2e/proj"
RES = "/cix/v1/v1_preds.json"

SYS = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
Q = ("\nPlease detect {ref} in this image. Answer the question with object indexes."
     "<|im_end|>\n<|im_start|>assistant\n<ground>{ref}</ground><objects>")
OBJ_RE = re.compile(r"<obj(\d+)")


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def mlp2x(x, w0, b0, w2, b2):
    return gelu(x @ w0.T + b0) @ w2.T + b2


def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL)
    with safe_open(os.path.join(MODEL, "model.safetensors"), framework="pt") as f:
        w = f.get_tensor("model.embed_tokens.weight")
    emb = torch.nn.Embedding(*w.shape, dtype=w.dtype)
    emb.weight.data.copy_(w)
    emb = emb.cuda()

    w0 = np.load(os.path.join(PROJ, "mm_projector_0_weight.npy"))
    b0 = np.load(os.path.join(PROJ, "mm_projector_0_bias.npy"))
    w2 = np.load(os.path.join(PROJ, "mm_projector_2_weight.npy"))
    b2 = np.load(os.path.join(PROJ, "mm_projector_2_bias.npy"))

    anns = [json.loads(line) for line in open(ANN)]
    print(f"anns {len(anns)}", flush=True)

    def to_dev(a):
        t = torch.frombuffer(bytearray(np.ascontiguousarray(a, np.float32).tobytes()),
                             dtype=torch.float32)
        return t.reshape(a.shape).cuda().bfloat16()

    t0 = time.perf_counter()
    llm = LLM(model=MODEL, enable_prompt_embeds=True, dtype="bfloat16",
              gpu_memory_utilization=0.55, max_model_len=4096,
              compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"})
    print(f"[init] {time.perf_counter() - t0:.1f}s", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=8)

    rows, lat = [], []
    for ann in anns:
        stem = f"{ann['id']:05d}"
        try:
            vit = np.load(os.path.join(OUT, stem + ".vit.npy")).astype(np.float32)
            l3 = np.load(os.path.join(OUT, stem + ".l3.npy")).astype(np.float32)
            obj = to_dev(np.load(os.path.join(OUT, stem + ".obj.npy")).astype(np.float32))
        except FileNotFoundError:
            continue
        img = to_dev(mlp2x(np.concatenate([vit, l3], -1), w0, b0, w2, b2))[None]
        parts = [emb(tok(SYS, return_tensors="pt").input_ids.cuda()), img]
        for i in range(obj.shape[0]):
            parts.append(emb(tok(f"<obj{i}>", return_tensors="pt").input_ids.cuda()))
            parts.append(obj[i:i + 1][None])
        parts.append(emb(tok(Q.format(ref=ann["referring"]), return_tensors="pt").input_ids.cuda()))
        ctx = torch.cat(parts, dim=1)[0].contiguous()

        t = time.perf_counter()
        o = llm.generate([EmbedsPrompt(prompt_embeds=ctx)], sp)[0]
        lat.append((time.perf_counter() - t) * 1000)
        txt = tok.decode(list(o.outputs[0].token_ids), skip_special_tokens=False)
        idx = sorted({int(x) for x in OBJ_RE.findall(txt) if int(x) < len(ann["candidate_boxes"])})
        rows.append({"id": ann["id"], "image_name": ann["image_name"], "domain": ann.get("domain", ""),
                     "referring": ann["referring"], "idx": idx,
                     "candidate_boxes": ann["candidate_boxes"],
                     "answer_boxes": ann.get("answer_boxes", []),
                     "raw": txt})

    json.dump({"n": len(rows), "ms_mean": float(np.mean(lat)), "ms_median": float(np.median(lat)),
               "rows": rows}, open(RES, "w"), ensure_ascii=False)
    print(json.dumps({"n": len(rows), "ms_mean": round(float(np.mean(lat)), 2),
                      "ms_median": round(float(np.median(lat)), 2)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
