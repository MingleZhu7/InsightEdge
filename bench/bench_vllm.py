"""Measure the deployed Qwen2.5-3B (RexSeek LM) latency under vLLM with CUDA graphs.

RexSeek feeds ~629 prompt tokens and decodes 2-3 index tokens, so measure prefill (1 token) and
prefill+decode (3 tokens) with a prompt of the same length, batch 1.
"""
import os

os.environ.setdefault("VLLM_ENFORCE_CUDA_GRAPH", "1")
os.environ.setdefault("VLLM_USE_V1", "1")

import statistics  # noqa: E402
import time  # noqa: E402

MODEL = "/models/Qwen2.5-3B-RexSeek-LM"
TARGET = 629


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    base = "Please detect the person wearing a red jacket in this image and answer with object indexes. "
    ids = tok(base, add_special_tokens=False)["input_ids"]
    prompt = base * (TARGET // len(ids) + 1)
    n = len(tok(prompt, add_special_tokens=False)["input_ids"])
    print(f"prompt tokens: {n}", flush=True)

    t0 = time.perf_counter()
    llm = LLM(model=MODEL, max_model_len=4096, gpu_memory_utilization=0.55,
              disable_log_stats=True, trust_remote_code=True, enforce_eager=False,
              compilation_config={"mode": "none", "cudagraph_mode": "full"})
    print(f"vLLM loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    def bench(max_tokens, n_iter=20, warm=3):
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        for _ in range(warm):
            llm.generate([prompt], sp, use_tqdm=False)
        ts = []
        for _ in range(n_iter):
            t = time.perf_counter()
            llm.generate([prompt], sp, use_tqdm=False)
            ts.append((time.perf_counter() - t) * 1000)
        ts.sort()
        return statistics.mean(ts), statistics.median(ts), ts[int(0.9 * len(ts)) - 1]

    m1, med1, p1 = bench(1)
    print(f"prefill only   (629->1): mean {m1:7.2f}  median {med1:7.2f}  p90 {p1:7.2f} ms", flush=True)
    m3, med3, p3 = bench(3)
    print(f"prefill+decode (629->3): mean {m3:7.2f}  median {med3:7.2f}  p90 {p3:7.2f} ms", flush=True)
    print(f"decode per token ≈ {(m3 - m1) / 2:.2f} ms", flush=True)


if __name__ == "__main__":
    main()

