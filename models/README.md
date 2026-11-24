# 模型与权重（不入库）

本目录只放**导出/编译脚本与配置**。权重体积约 15 GB，按下列清单放到板卡对应路径。

## 1. 权重清单

| 文件 / 目录 | 放到板卡 | 大小 | 来源 |
|---|---|---|---|
| `IDEA-Research__RexSeek-3B/` | `/cix/models_orig/` | 7.3 GB | HuggingFace `IDEA-Research/RexSeek-3B` |
| `Qwen2.5-3B-RexSeek-LM/` | `/models/` | 6.3 GB | RexSeek-3B 内的 LM（可用 `extract_lm.py` 抽出） |
| `Qwen2.5-1.5B-Instruct/` | `/cix/rexseek-studio/models/` | 3.1 GB | HuggingFace |
| `grounding-dino-tiny/` | `/cix/rexseek-studio/models/` | 0.7 GB | HuggingFace `IDEA-Research/grounding-dino-tiny` |
| `yoloe-26l-seg.pt` | `/cix/` | 79 MB | ultralytics YOLOE（AGPL-3.0） |
| `mobileclip2_b.ts` | `/cix/` | 254 MB | ultralytics MobileCLIP2 文本编码器 |
| `yoloe26l_person_split_i8.cix` | `/home/cix/` | 23.4 MB | **本仓库编译产物**（见 §3） |
| `yoloe-26l-person-split.onnx` | `/cix/export_person/` | 112 MB | **本仓库导出产物**（见 §2） |

> HumanRef 数据集（`bench/imgs` + `annotations.jsonl`）同样不入库，从官方仓库获取。

## 2. 导出 person 专用 ONNX

为什么必须重导：原始 116 通道导出的分类分支全 0（文本嵌入没烘进图），NPU 侧会静默返回错误类别。

```bash
# 容器 s1c 内
python3 export_person_onnx.py     # set_classes(["person"]) -> 37 通道 (4 框 + 1 分 + 32 mask)
python3 split_outputs.py          # 拆成 box / score / mask 三个独立输出（NPU 必需）
```

校验：`score` 最大值 ≈ 0.87，`>0.25` 的 anchor ≈ 144 个。

> RepRTA 文本嵌入由 `build_w.py` 重建（HF `CLIPTokenizer` + `mobileclip2_b.ts`），与官方 `W200.npy` 逐条对齐 **cos = 1.00000**。

## 3. 编译 NPU 模型（x86 WSL）

```bash
# 依赖：CixBuilder 6.1.3753（/opt/cixenv_min/bin/cixbuild）
cd ~/npu_build
cp /path/to/yoloe-26l-person-split.onnx .
cp /path/to/calib_yoloe_real.npy .        # 64 张 640×640 校准图
cixbuild yoloe26l_person_split.cfg        # 见本目录同名 cfg
# 产物：yoloe26l_person_split_i8.cix（约 23 MB）
```

### 关键配置

| 项 | 值 | 说明 |
|---|---|---|
| `weight_bits` / `activation_bits` | 8 / 8 | INT8 |
| `target` | `X2_1204MP3` | NPU 型号 |
| `ocm_addr` / `ocm_size` | `0xa0017000` / `0x3000` | 片上内存 |
| `tiling` | `fps` | 分块策略 |
| **输出** | `box_out, score_out, mask_out, output1` | **必须拆开**；37 通道混排 Concat 的 INT8 版本分数恒为 0 |

> 若一定要保留混排 Concat，只能让它留在 fp16（`trigger_float_op = disable & <[Concat]:float16_preferred!>`），精度会从 0.7468 降到 0.7273。

## 4. 部署时指定模型

```bash
LD_LIBRARY_PATH=/usr/share/cix/lib YOLOE_CIX=/home/cix/yoloe26l_person_split_i8.cix \
  setsid nohup python3 yoloe_service.py >> .work_yoloe.log 2>&1 &
```

`yoloe_service.py` 默认就指向这个模型；`/api/health` 的 `model` 字段会显示实际加载的文件名。
