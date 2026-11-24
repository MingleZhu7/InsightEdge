# NOTICE · 第三方组件与许可

本项目（Insight 端侧视觉值守系统）由以下第三方组件构建。仓库本身不含任何上游权重与数据集。

## 运行时依赖（板端镜像内）

| 组件 | 版本（实测） | 许可 | 用途 | 是否随仓库分发 |
|---|---|---|---|---|
| RexSeek-3B（IDEA Research） | 官方权重 | 见上游 LICENSE（研究用途） | 指代模型（视觉塔 + LM） | 否，需自行下载 |
| Qwen2.5-3B / Qwen2.5-1.5B（阿里） | 官方权重 | Apache-2.0 | LLM / Agent planner | 否 |
| CLIP / open_clip | — | MIT | CLIP-L/14-336 视觉塔 | 否 |
| vLLM | 0.23.0 | Apache-2.0 | LLM 推理（CUDA Graph + paged attention） | 否（容器镜像） |
| PyTorch / torchvision | 2.7 / 0.20 | BSD-3-Clause | 推理框架 | 否 |
| IxRT / Corex | 4.4.0 | 天数智芯 EULA | GPU 引擎与驱动 | 否 |
| NOE Engine / AIPUBuilder | 6.1.3753 | 此芯科技 EULA | NPU 编译与运行时 | 否 |
| transformers | — | Apache-2.0 | 模型加载与分词 | 否 |

## 离线工具依赖（仅导出模型时使用，不进入板端运行镜像）

| 组件 | 许可 | 说明 |
|---|---|---|
| **ultralytics（YOLOE）** | **AGPL-3.0** | `models/export_person_onnx.py` 通过它加载 `yoloe-26l-seg.pt`、灌入 `person` 词表并导出 ONNX。**AGPL-3.0 具传染性**：若以公开仓库发布，建议把 `models/` 目录单独说明，或仅在私有仓库/内部构建环境提供；板端运行时只加载导出的 `.cix` / `.onnx`，不 import ultralytics。 |
| MobileCLIP2（Apple） | Apple Sample Code License | `mobileclip2_b.ts` 文本编码器，用于生成 RepRTA 词表 |
| CixBuilder | 此芯科技 EULA | NPU 编译 |

## 数据

| 数据 | 许可/说明 |
|---|---|
| HumanRef Benchmark（6000 条指代 + 图片） | 随 RexSeek 发布，按上游 LICENSE 使用；**本仓库不分发其内容**，`bench/build_strat.py` 只提供抽样脚本 |
| 现场演示视频 / 图片 | 项目自制或已授权素材；入库前请确认无肖像/隐私问题 |

## 引用

```bibtex
@inproceedings{jiang2025rexseek,
  title={Referring to Any Person},
  author={Jiang, Qing and Wu, Lin and Zeng, Zhaoyang and Ren, Tianhe and Xiong, Yuda and Chen, Yihao and Qin, Liu and Zhang, Lei},
  booktitle={ICCV},
  year={2025}
}
```
