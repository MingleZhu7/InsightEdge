"""Tower latency on the board: eager vs CUDA-graph-captured eager (no Triton needed).

torch.compile is unavailable in this container (torch._inductor requires Triton), so the
optimisation lever that actually exists here is CUDA Graph capture of the eager module.
"""
import statistics
import sys
import time

import torch

sys.path.insert(0, "/cix/third_party/RexSeek")
from rexseek.builder import load_rexseek_model  # noqa: E402

torch.set_grad_enabled(False)
tokenizer, model, ip, _ = load_rexseek_model("/cix/models_orig/IDEA-Research__RexSeek-3B")
model.eval()
VT = model.vision_tower
VA = model.vision_tower_aux

x336 = torch.randn(1, 3, 336, 336, device="cuda", dtype=torch.float16)
x768 = torch.randn(1, 3, 768, 768, device="cuda", dtype=torch.float16)


def timeit(fn, n=30, warm=5):
    with torch.inference_mode(), torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        for _ in range(warm):
            fn()
    torch.cuda.synchronize()
    ts = []
    with torch.inference_mode(), torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        for _ in range(n):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(ts), statistics.median(ts), sorted(ts)[int(0.9 * len(ts)) - 1]


def report(name, fn):
    m, med, p90 = timeit(fn)
    print(f"{name:34s} mean {m:7.2f}  median {med:7.2f}  p90 {p90:7.2f}", flush=True)
    return m


def capture(mod, x):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode(), torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        for _ in range(3):
            mod(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        with torch.cuda.graph(g):
            out = mod(x)
    return g, out


print("=== eager bf16 ===", flush=True)
e_clip = report("CLIP ViT-L/14-336 eager", lambda: VT(x336))
e_cn = report("ConvNeXt-L@768 eager", lambda: VA(x768))

print("\n=== eager + CUDA graph ===", flush=True)
g_clip, o1 = capture(VT, x336)
c_clip = report("CLIP ViT-L/14-336 graph", lambda: g_clip.replay())
g_cn, o2 = capture(VA, x768)
c_cn = report("ConvNeXt-L@768 graph", lambda: g_cn.replay())

print(f"\nsum eager {e_clip + e_cn:.2f} ms | sum graph {c_clip + c_cn:.2f} ms | "
      f"speedup {(e_clip + e_cn) / (c_clip + c_cn):.2f}x", flush=True)
