# Insight · 端侧视觉值守系统（Edge Referring Agent）

> 在一块国产 SoC（CIX P1：12 核 ARM + 45 TOPS NPU + 天数 MR-V100 GPU）上，全离线跑通「自然语言指代 → 目标定位 → 事件值守 → 追问」的完整链路。
> 整机 18 W 级、检测 51.7 ms、端到端问答 0.6–2 s、开放词表指代 F1@0.5 = 0.7468。

[![board](https://img.shields.io/badge/board-CIX%20P1%20CD8180-blue)]() [![NPU](https://img.shields.io/badge/NPU-Arm%20China%20X2__1204MP3-green)]() [![GPU](https://img.shields.io/badge/GPU-Iluvatar%20MR--V100%2032GB-orange)]()

---

## 1. 这是什么

现场（工地/园区）的安全值守需求是"守住一小块地方、规则经常变"。传统闭集检测每加一条规则就要重训模型；本项目改用**指代式检索**：

```
一句话（"哪些番茄已经熟了" / "穿蓝色衣服的人进没进危险区"）
        ↓
   person/开放词表候选（NPU / GPU）
        ↓
   RexSeek-3B 从候选中选索引（GPU，vLLM）
        ↓
   框 + 事件台账 + 可追问的证据
```

**没有标注、没有微调**：换一条规则只需要换一句话。

### 能力

- 开放词表指代定位（人物/物品/属性/空间关系）
- 计数、属性判断、拒答（图上没有就不出框）
- 多轮指代记忆（"她右边是谁"）
- 区域值守 + 事件台账（进入/离开/持续/去重人数/取证图）
- 事件追问（数字/时间由事件库直接回答，不经过模型）

---

## 2. 仓库结构

```
insight-edge/
├── rexseek-studio/          # 可直接部署到板卡 /home/cix/rexseek-studio 的完整服务
│   ├── server.py            # 前端网关 + 静态资源 + /api/*
│   ├── rexseek_service.py   # Agent 后端（RexSeek-3B + Qwen2.5-1.5B，vLLM）
│   ├── agent.py             # 规划 / 工具 / 记忆 / 降级
│   ├── ppe_service.py       # 值守检测 + 事件聚合
│   ├── yoloe_service.py     # NPU YOLOE-26l person 检测服务
│   ├── engine/              # mock（纯规则）与 proxy（真实推理）两种引擎
│   ├── web/                 # 三套前端页面（值守台 / 对话分析 / 通用演示）
│   └── start_*.sh           # 一键与单服务启动脚本
├── models/                  # 检测模型导出 / 拆分输出 / NPU 编译配置（不含权重）
├── bench/                   # 精度评测与分阶段测速脚本
├── eval/                    # 评测集构建脚本与结果 CSV
└── docs/                    # 部署手册、性能数据、白皮书章节
```

> 权重（RexSeek-3B 7.3 GB、Qwen2.5-3B 6.3 GB、YOLOE .cix/.onnx）不入库，见 [`models/README.md`](models/README.md)。
> **演示素材（番茄等 6 张图 + 3 段值守视频，约 6 MB）已随仓库**，clone 后无需准备素材即可跑通两个前端。

---

## 3. 环境要求

| 项 | 要求 |
|---|---|
| 板卡 | CIX P1（CD8180）：12 核 Cortex-A720/A520 + Arm China NPU X2_1204MP3 + 天数 MR-V100 32 GB |
| 系统 | Debian 12 aarch64，内核 6.6.89-cix-build-generic |
| 容器 | `s1c`（PyTorch 2.7 + IxRT）、`qwen_opt`（vLLM 0.23） |
| NPU 侧 | host python3.11 + `NOE_Engine`（`/usr/local/lib/python3.11/dist-packages`） |
| GPU | Corex 4.4.0 驱动 + IxRT |
| Python | 容器内 3.10；host 3.11（NPU 必须） |

---

## 4. 一键部署

```bash
# 0) 代码放到板卡（目录名保持 rexseek-studio）
scp -r rexseek-studio cix@<BOARD_IP>:/home/cix/

# 1) 权重与模型（按 models/README.md 放置，约 15 GB）
#    /cix/models_orig/IDEA-Research__RexSeek-3B
#    /models/Qwen2.5-3B-RexSeek-LM
#    /cix/rexseek-studio/models/{Qwen2.5-1.5B-Instruct,grounding-dino-tiny}
#    /home/cix/yoloe26l_person_split_i8.cix

# 2) 启动后端 + 前端一（值守台 :8848），约 90 s
bash /home/cix/rexseek-studio/start_all_fixed.sh

# 3) 启动前端二（通用演示 :8849），约 90 s
bash /home/cix/rexseek-studio/start_general_demo.sh

# 4) 可选：纯对话演示（:8850，不加载模型）
bash /home/cix/rexseek-studio/start_chat_demo.sh 8850
```

访问：

| 入口 | 地址 | 用途 |
|---|---|---|
| 值守台 | `http://<BOARD_IP>:8848/` | 视频值守、框选危险区、事件台账 |
| 对话分析 | `http://<BOARD_IP>:8848/studio.html` | 就事件提问、看思考过程（trace） |
| 通用演示 | `http://<BOARD_IP>:8849/` | 番茄/求婚/海报/合影/高速 5 个示例 |

健康检查：

```bash
curl -s http://127.0.0.1:8848/api/status | python3 -m json.tool
# 期望 ok=true，gateway/agent/ppe/yoloe 全 true
```

> 详细的启动顺序、单服务重启、排障见 [`docs/DEPLOY.md`](docs/DEPLOY.md)。

---

## 5. 性能与精度（板卡实测）

### 5.1 分阶段耗时

| 段 | 设备 | 耗时 |
|---|---|---:|
| 预处理（letterbox + 768 双路） | CPU | 39.4 ms |
| **YOLOE-26l person 候选** | **NPU INT8** | **51.7 ms** |
| CLIP ViT-L/14-336 | GPU | 28.1 ms（CUDA Graph） |
| ConvNeXt-L@768 | GPU | 61.3 ms（CUDA Graph） |
| RoIAlign + 投影器 | GPU | 20–35 ms |
| **Qwen2.5-3B 选索引** | GPU（vLLM） | 78.5 ms（629→3 token） |

端到端关键路径 **207.3 ms（4.8 FPS）**：`39.4 + max(51.7 NPU, 89.4 GPU 双塔) + 78.5`。

### 5.2 精度（HumanRef 6 域分层 120 条）

| 配置 | F1@0.5 | 严格 IoU | 说明 |
|---|---:|---:|---|
| **本系统（YOLOE-NPU + L 塔）** | **0.7468** | 0.5350 | 部署配置 |
| GD graft448 + L 塔 | 0.7475 | 0.6885 | 全 GPU 参照 |
| annotation 候选 + L 塔 | 0.7681 | 0.7669 | 上界 |

分域 F1@0.5：attribute 0.8274 / position 0.7000 / interaction 0.5433 / reasoning 0.6728 / celebrity 0.9873。

> 完整表格与复现命令见 [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) 与 [`eval/`](eval/)。

---

## 6. 复现实验

```bash
# 构建 6 域分层评测集（需要 HumanRef annotations.jsonl + 图片）
python3 bench/build_strat.py

# 精度 + 分阶段耗时（部署配置）
docker exec -w /cix s1c python3 /cix/unified_bench.py --cand yoloe_npu \
  --npu-cand /cix/cand120/annot_npu_split_i8.json \
  --ann /cix/cand120/annot_strat120.json --img-dir /cix/bench/imgs --n 120

# 双塔 CUDA Graph
docker exec -w /cix s1c python3 /cix/tower_graph_only.py

# vLLM（必须带 main guard，且 mode=none + cudagraph）
docker exec qwen_opt python3 /cix/bench_vllm.py

# Agent 端到端延迟
python3 /home/cix/agent_latency.py
```

---

## 7. 已知限制

1. **严格 IoU 偏弱**：INT8 候选框在 IoU 0.75 下明显掉点（0.5350 vs GPU 0.6885），IoU 0.5 可用。
2. **本板无 Triton**：`torch.compile` 不可用，视觉塔只能靠 CUDA Graph（双塔 1.07×）。
3. **拒答能力弱**：person 检测器永远给框，rejection 子集 20 条只有 1 条正确拒答（v1 是 11 条），需下游规则兜底。
4. **量化约束**：NPU 模型必须把 `box/score/mask` 拆成独立输出，混排 Concat 的 INT8 版本分数恒为 0。
5. 评测集每域 20 条（共 120），置信区间约 ±0.1，对外发布建议扩到每域 ≥100。

---

## 8. 第三方组件与许可

| 组件 | 用途 | 许可 |
|---|---|---|
| [IDEA-Research/RexSeek](https://github.com/IDEA-Research/RexSeek) | 指代模型与 HumanRef benchmark | 见上游 LICENSE（研究用途，需自行 clone，不随本仓库分发） |
| 演示图片 / 现场视频（`web/assets`、`web/media`） | 前端演示素材 | 项目自制或已获授权；如需公开发布，请自行确认肖像与版权 |
| [ultralytics](https://github.com/ultralytics/ultralytics) | YOLOE 导出（**仅离线导出用**） | **AGPL-3.0**：`models/` 下的导出脚本依赖它；板端运行时只用导出的 `.cix/.onnx`，不 import ultralytics |
| Qwen2.5-3B / 1.5B | LLM | Apache-2.0 |
| OpenAI CLIP / open_clip | 视觉塔 | MIT |
| vLLM | 推理引擎 | Apache-2.0 |

> ⚠️ 若以公开仓库形式发布，请确认 AGPL-3.0（ultralytics）对 `models/` 目录的传染性：建议把 `models/` 单独标注或仅在私有仓库提供，板端镜像不含 ultralytics。

---

## 9. 引用

```bibtex
@inproceedings{jiang2025rexseek,
  title={Referring to Any Person},
  author={Jiang, Qing and Wu, Lin and Zeng, Zhaoyang and Ren, Tianhe and Xiong, Yuda and Chen, Yihao and Qin, Liu and Zhang, Lei},
  booktitle={ICCV},
  year={2025}
}
```

