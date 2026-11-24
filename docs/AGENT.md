# Agent 的设计与部署（白皮书章节草稿）

> 对应架构图：`图 3-1 检索式 Agent 的数据流`（Qwen2.5-1.5B Agent → 预处理 CPU → YOLOE-26l NPU / CLIP ViT-L/14 GPU / ConvNeXt-L GPU → 目标令牌 GPU → Qwen2.5-3B GPU·vLLM → 结果）
> 数据：2026-11-23 板卡实测（CIX P1 + MR-V100 32GB + NPU X2_1204MP3）

---

## 3.1 功能设计

### 3.1.1 能力矩阵：工具，而不是提示词

端侧 Agent 最大的风险不是"回答得不好"，而是"编"。我们把能力切成一组**确定性的工具**，每个工具只做一件可验证的事，模型只负责在工具之间做选择：

| 工具 | 解决什么问题 | 依赖的算力 | 输出 |
|---|---|---|---|
| `ground` | 按描述定位目标（"穿蓝色衣服的人在哪"） | YOLOE 候选（NPU）+ 双塔 + 3B（GPU） | 框 + 空间描述 |
| `count` | 计数（"画面里有几个人"） | 同上（结果可缓存） | 数字 + 框 |
| `zone_query` | 开放描述 + 区域归属（"他进没进危险区"） | `ground` + 多边形判定 | 是/否 + 证据 |
| `ppe_check` | 装备检测（安全帽、反光背心） | 按需 open-vocab 检测 | 人/装备配对结果 |
| `attribute` | 反向属性（"这个人穿什么颜色"） | 多假设定位 + IoU 匹配 | 颜色/属性 |
| `guard_check` / `guard_events` / `guard_segment` | 实时区域状态 / 台账 / 整段统计 | 事件库（SQLite） | 结构化事实 |
| `describe` / `chat` | 开放式描述 / 闲聊 | 1.5B 语言模型 | 自然语言 |

三个设计取舍：

1. **不在值守时自动标注所有属性**。值守只跑 person 检测；安全帽、颜色这类属性只在用户提问时按需检测一次。这样"每帧成本"与"提问成本"解耦——值守是常驻的、必须便宜（NPU 51.7 ms），提问是偶发的、可以慢一点。
2. **拒答优先于猜测**。外观类问题（"他穿什么颜色的衣服"这类**反向属性**）走 `attribute` 的多假设定位：对 8 种颜色逐个定位再比 IoU，低于阈值就如实说"看不清"，不让模型自由发挥。
3. **数字不经过模型**。人数、时间、时长这类事实由事件库直接拼装；只有"有什么风险""发生了什么"这类开放式问题才交给语言模型组织。

### 3.1.2 路由：规则护栏 + planner

纯 planner 在小模型上不稳定（会拿台账统计去回答"戴没戴安全帽"），纯规则又覆盖不了开放问法。最终是**规则护栏 + planner 决策**的双层结构：

```
用户问题
  ├─ 护栏（确定性，优先级最高）
  │   台账类 → guard_events ｜ 装备类 → ppe_check
  │   事件统计 → guard_segment ｜ 区域归属 → zone_query
  └─ 其余 → Qwen2.5-1.5B planner 在工具目录里选
        └─ 失败 → 关键词规则兜底 → 再兜底 ground
```

每次回答的 `trace` 里记录 `chosen_by`（Agent 决策 / 规则护栏 / 规则兜底），既方便调试，也让"为什么这么答"可追溯。实测下来，护栏覆盖了现场最高频的三类问题，且这部分**不需要 planner 参与**——`ppe_check` 的 planner 耗时为 0，`zone_query` 也只有 2 ms。

### 3.1.3 回答分层：事实与语言分离

| 问题类型 | 处理方式 | 是否经过模型 |
|---|---|---|
| 事实型（时间、人数、时长、戴没戴） | 由事件库/检测结果直接拼装 | **否** |
| 区域归属（"他进没进危险区"） | `zone_query`：定位 + 多边形判定 | 定位经过，判定不经过 |
| 开放式（"有什么风险""发生了什么"） | `compose`：把事实交给 1.5B 组织语言 | 是（1.5B） |
| 外观类（"穿的什么颜色"） | 多假设定位，低于阈值如实说"看不清" | 定位经过，不猜 |

这一层设计让"数字和事实"永远不会被模型改写，同时保留了自然语言表达的灵活性。

### 3.1.4 事件与会话：从"帧"折叠成"事件"

值守台不是逐帧报警，而是维护一张事件表（`guard_episodes`）：`enter_ts / exit_ts / duration_s / persons_max / persons_total / mission / evidence / segment / n_frames`。

规则：一次"进入→离开"记一条；离开超过 15 秒才回填离开时间（人短暂出画不会把一次事件刷成多条）；事件期间逐帧归档（默认 960 px、上限 300 帧），支持片段回放与整段追问。会话侧用 SQLite 存多轮上下文，`session_id` 严格校验（`^s_[A-Za-z0-9_-]{6,64}$`），避免不同会话串上下文。

### 3.1.5 为速度做的三个设计

| 设计 | 效果（实测） |
|---|---|
| **图像索引缓存**（按图像 hash） | 同图第二次 `index` 从 **319 ms → 7–12 ms** |
| **答案缓存**（同图同题） | `count` 从 3868 ms → **19 ms** |
| **翻译缓存**（中英短语，最多 256 条） | 避免每次重复调用 1.5B |
| 按需检测（不做全属性自动标注） | 值守链路只需 person 检测（NPU 51.7 ms） |

---

## 3.2 部署形态

### 3.2.1 组件与设备分配

| 组件 | 设备 | 实现 | 实测 |
|---|---|---|---:|
| Qwen2.5-1.5B（planner / compose / 翻译） | GPU | transformers，fp16，进程内 | 0–1.63 s（按问题复杂度） |
| 预处理（解码 + letterbox） | CPU | PIL/numpy | ≈39 ms |
| YOLOE-26l person 候选 | **NPU** | 独立服务 `:8090`，INT8，HTTP | **51.7 ms** |
| CLIP ViT-L/14-336 | GPU | RexSeek-3B 自带视觉塔 | 33.8 ms（eager） |
| ConvNeXt-L@768 | GPU | 同上 | 62.3 ms（eager） |
| RoIAlign + 投影器 + ctx | GPU | RexSeek | 20–35 ms |
| **Qwen2.5-3B（RexSeek 选索引）** | GPU | **vLLM 0.23**，prompt embeds，CUDA Graph | **390–500 ms** |

### 3.2.2 3B 的推理引擎

三个关键配置（缺一不可）：

1. **vLLM 而不是 HF `generate`**：同一块卡、同样 629→3 token，HF 1651.8 ms，vLLM 78.5 ms（**21×**）。
2. **CUDA Graph 必须开**：本板 `torch.compile` 不可用（缺 Triton），vLLM 需显式设置 `compilation_config={"mode": "none", "cudagraph_mode": "full"}`（Agent 服务里用 `FULL_DECODE_ONLY`），否则会因为 inductor 编译失败而起不来。
3. **prompt embeds 路径**：RexSeek 的图像/目标令牌在送入 LLM 前已是 embedding，vLLM 必须 `enable_prompt_embeds=True`；这条路径比 token 路径慢（见 3.4 的讨论），但省掉了在服务里重建 token 的复杂度。

### 3.2.3 串行还是并行

当前部署是**串行**：`1.5B planner → 预处理 → YOLOE ∥ 双塔 → RoI/投影 → 3B → 1.5B compose`。其中唯一真正并行的是 **NPU 检测与 GPU 双塔**（两者在不同单元上，51.7 ms 被 96 ms 掩盖）。

这个选择不是一开始就有的，下一节是演进过程。

---

## 3.3 演进：从「CPU 并行」到「GPU 串行」

### 3.3.1 初始方案：llama.cpp 放 CPU，与 NPU/GPU 并行

最初的动机很自然：板子有 12 个 CPU 核、一块 NPU、一块 GPU，Agent 的小模型（Qwen3.5-0.8B，Q4 GGUF）完全可以放在 CPU 上，用 llama.cpp 起一个 `llama-server`，这样：

- GPU 完整留给 3B 与视觉塔；
- CPU 上的 planner 与 GPU 上的检索**理论上可以并行**；
- 整机功耗也更可控。

### 3.3.2 为什么失败：prompt 太长，prefill 成了瓶颈

实测数据（CPU 0.8B，8 线程，`-ngl 0`）：

| 项 | 数值 |
|---|---|
| llama.cpp prefill 吞吐 | **183 tok/s** |
| 真实 planner 提示词 | **813 token**（工具目录 401 + 14 条示例 232 + 模板 157 + 问题 23） |
| 冷启动 prefill | **2811 ms**（即 1–2 s 量级） |
| 命中前缀缓存 | 35 ms（但每次请求都是新问题，前缀命中不了） |
| 单轮端到端 | **约 4 s** |

问题不在 decode（0.8B 在 CPU 上 decode 很快），而在 **prefill 是 O(prompt) 的**：813 token、183 tok/s，单是"读懂任务"就要 2.8 秒。而"并行"节省的是 GPU 那 100–200 ms 的窗口，根本抵不过 prefill 多出来的 2 秒。

**结论：并行带来的收益（~150 ms）远小于 prefill 的代价（~2000 ms），不如直接串行。**

### 3.3.3 转向 GPU 串行

改法有三步：

1. **把 Agent 的 LLM 换成 GPU 上的 Qwen2.5-1.5B**（进程内，fp16），planner / 翻译 / 组织语言都走它；
2. **planner 提示词瘦身 813 → ~350 token**（工具目录压成"工具名 + 短说明"，示例 14 → 6）；
3. 翻译结果加缓存。

效果（同一批问题，热启动）：

| 问题 | 改前（CPU 0.8B） | 改后（GPU 1.5B） |
|---|---:|---:|
| 画面里的人在哪里？ | 6356 ms | **1038–1352 ms** |
| 你好 | 5625 ms | **1114 ms** |
| 穿蓝衣服的人在不在危险区？ | 1805 ms | **889 ms** |
| 现在危险区有人吗？ | 804 ms | **403 ms** |
| 这个人戴安全帽了吗？ | 588 ms | 554 ms |

分阶段：planner 3179–3598 → **593–1011 ms**；`ground` 工具 2700 → **313 ms**；compose 2423 → **498 ms**。

**串行反而更快**——因为 prefill 从"CPU 上 2.8 秒"变成"GPU 上几百毫秒"，而所谓的并行窗口本来就只有一两百毫秒。

---

## 3.4 当前耗时分解

### 3.4.1 组件级

| 组件 | 设备 | 耗时 |
|---|---|---:|
| 预处理 | CPU | ≈39 ms |
| YOLOE-26l person | NPU INT8 | 51.7 ms |
| CLIP ViT-L/14-336 | GPU | 33.8 ms |
| ConvNeXt-L@768 | GPU | 62.3 ms |
| RoIAlign + 投影器 | GPU | 20–35 ms |
| **Qwen2.5-3B（vLLM，embeds）** | GPU | **390–500 ms** |
| Qwen2.5-1.5B planner | GPU | 0–1630 ms |
| Qwen2.5-1.5B compose | GPU | 0（事实型）/ 950–1060 ms（开放式） |

冷启动（新图第一次）`index` 一次 319 ms = 预处理 + YOLOE 候选 + 双塔编码 + RoI/投影 + 写缓存；之后同图 7–12 ms。

### 3.4.2 请求级（同图稳态）

| 问题 | 路由 | planner | index | tool | compose | 总计 |
|---|---|---:|---:|---:|---:|---:|
| 画面里有几个人？（缓存） | count | — | 10 | — | 0 | **19 ms** |
| 左边的人戴安全帽了吗？ | ppe_check | 0 | 11 | 552 | 0 | **565 ms** |
| 画面里的人在哪里？ | ground | 1631 | 13 | 520 | 0 | 2158 ms |
| 画面里的人在哪里？（新图） | ground | 1038 | 319 | 685 | 0 | 1729 ms |
| 现在危险区有人吗？ | zone_query | 2 | 11 | 2423 | 0 | 2437 ms |
| 这个人穿什么颜色的衣服？ | attribute（8 次定位） | 844 | 10 | 3782 | 0 | 4631 ms |
| 描述一下这张图 | describe | 663 | 15 | 0 | 945 | 1613 ms |
| 你好 | chat | 1117 | 19 | 0 | 1062 | 2184 ms |

统计：非缓存请求中位约 **1.8–2.3 s**，事实型问题（护栏路由）可以做到 **0.4–0.6 s**。

### 3.4.3 一个明确的优化点

`ground` 的 520 ms 里，3B 解码占 400–500 ms，而同一块卡上走 **token 路径**（629→3 token）只要 **78.5 ms**。差别来自 `prompt_embeds` 路径与 `max_tokens=24`。如果把 3B 改成 token 路径（或把 max_tokens 降到 4–8），`ground` 有望压到 150 ms 量级——这是 Agent 端目前最大的单项空间。

---

## 3.5 回顾与展望：还没用完的异构算力

### 3.5.1 现状

- **NPU 只做了检测**（51.7 ms），占整轮时间不到 10%，大部分时间空闲；
- **GPU 串行承担**双塔 + 3B + 1.5B，负载峰值 48 W（均值 40 W）；
- **CPU 12 核**基本只做预处理（39 ms）和 Python 调度，解码阶段几乎闲置。

也就是说，**异构算力用了一半**：并行只发生在"检测 ∥ 双塔"这一处，LLM 的两段（prefill / decode）始终在同一块 GPU 上串行。

### 3.5.2 下一步：PD 分离（Prefill-Decode Disaggregation）

思路是把 LLM 推理的两个阶段拆到不同单元：

| 阶段 | 特征 | 适合的单元 | 放到哪 |
|---|---|---|---|
| **Prefill** | 计算密集（矩阵乘）、并行度高 | GPU | GPU（保持） |
| **Decode** | 访存密集（每 token 读一遍 KV/权重）、并行度低 | 高带宽或低成本单元 | CPU（或 NPU） |
| 传输 | KV cache 交接 | 高带宽链路 | 共享内存 / PCIe |

预期收益：

1. GPU 不再"陪着" decode 空转，可以把这段时间让给视觉塔或下一帧的 prefill；
2. CPU 的 12 核在 decode 阶段本来闲置，正好承接访存密集的小批量解码；
3. 与现有的"检测 ∥ 双塔"并行叠加，整条链路可能变成"检测(NPU) ∥ 双塔(GPU) ∥ decode(CPU)"。

要解决的问题也很清楚：

- **KV cache 传输**：3B 模型单请求的 KV 在几十 MB 量级，跨设备搬运需要共享内存或 PCIe DMA，延迟必须低于 decode 本身收益；
- **CPU decode 的速度**：3B 在 CPU 上 decode 受内存带宽限制，需要量化（GGUF Q4）与多线程调优，否则"省下的 GPU 时间"会被 CPU 的慢解码吃掉；
- **调度与一致性**：prefill/decode 分设备后，采样策略、停止条件、缓存前缀命中都要重新对齐；
- **工具链**：本板 vLLM 尚不支持跨设备 PD 分离，需要自研调度或等厂商支持。

### 3.5.3 小结

从"llama.cpp 放 CPU 并行"到"GPU 串行"，这一轮演进说明了一个反直觉的结论：**在端侧，先把 prefill 放对地方，比强行并行更重要**。而下一阶段的方向不是继续减少串行环节，而是把 decode 从 GPU 上挪走，让 GPU 专注 prefill 与视觉——这也是我们下一步要在板卡上验证 PD 分离的原因。

---

## 3.6 运行与复现

### 3.6.1 环境依赖

| 项 | 值 |
|---|---|
| 项目根 | 板卡 `/home/cix/rexseek-studio`（容器内 `/cix/rexseek-studio`） |
| 容器 | `s1c`（RexSeek + 双塔）、`qwen_opt`（vLLM） |
| 模型 | `/cix/models_orig/IDEA-Research__RexSeek-3B`、`/models/Qwen2.5-3B-RexSeek-LM`、`/cix/rexseek-studio/models/Qwen2.5-1.5B-Instruct`、`/cix/rexseek-studio/models/grounding-dino-tiny` |
| NPU 侧 | host python3.11 + `NOE_Engine`，`LD_LIBRARY_PATH=/usr/share/cix/lib` |
| 端口 | 8848 网关+前端 / 9000 Agent / 8092 值守检测 / 8090 NPU YOLOE |
| 关键环境变量 | `REXSEEK_USE_VLLM=1`、`VLLM_GPU_UTIL=0.35`、`YOLOE_URL=http://10.4.0.1:8090`、`PPE_URL=http://127.0.0.1:8092`；**不要开** `REXSEEK_COMPILE`（本板缺 Triton） |
| YOLOE 模型 | `YOLOE_CIX`（默认 `/home/cix/yoloe26l_person_split_i8.cix`，split 版 4 输出）；改模型只改这个变量，不用动代码 |

### 3.6.2 启动

**一键启动（推荐）**

```bash
bash /home/cix/rexseek-studio/start_all_fixed.sh
# 容器 qwen_opt=10.4.0.x | 宿主桥 10.4.0.1
# ✅ 全部就绪
#     值守台    http://<板卡IP>:8848/
#     对话分析  http://<板卡IP>:8848/studio.html
```

约 90 s（其中 Agent 加载 vLLM 约 76 s）。

**单服务重启**

```bash
# Agent（9000，约 76 s）
docker exec -d qwen_opt bash /cix/rexseek-studio/start_agent.sh
# 值守检测（8092）
docker exec -d qwen_opt bash /cix/rexseek-studio/start_ppe.sh
# NPU YOLOE（8090，必须带库路径）
cd /home/cix/rexseek-studio && \
  LD_LIBRARY_PATH=/usr/share/cix/lib setsid nohup python3 yoloe_service.py >> .work_yoloe.log 2>&1 &
# 换模型只改环境变量（默认就是 split 版）
#   LD_LIBRARY_PATH=/usr/share/cix/lib YOLOE_CIX=/path/to/other.cix setsid nohup python3 yoloe_service.py >> .work_yoloe.log 2>&1 &
# 启动日志会打印 [yoloe] model=<实际路径>；/api/health 的 model 字段也会变成对应文件名
# 网关 + 前端（8848）
bash /home/cix/rexseek-studio/start_ui.sh
```

> ⚠️ 原版 `start_all.sh` 里有三个硬编码会失效：容器 IP `10.4.0.35`（换容器后变）、容器内脚本路径 `/home/cix/...`（容器里是 `/cix/...`）、桥接地址 `172.17.0.1`（本板是 `nerdctl0` 的 `10.4.0.1`）。
> `start_all_fixed.sh`、`start_ui.sh`、`start_agent.sh` 已改为运行期解析，直接用即可。

### 3.6.3 健康检查

```bash
curl -s http://127.0.0.1:8848/api/status
```

期望：`"ok": true`，且 `gateway / agent / ppe / yoloe` 四项全为 `true`。

### 3.6.4 跑一次问答

**浏览器**（推荐）：值守台 → 选/上传视频 → 开始值守 → 对话分析提问。页面会自动截当前帧上传，不需要 `sample_id`。

**脚本 / API**（注意 `data/samples.json` 为空，直接传 `sample_id` 会报 400，必须传 `media.data_url`）：

```bash
# 现成脚本（内置一张示例图 + 计时）
python3 /home/cix/agent_speed_test.py '画面里有几个人？' s_demo01

# 或 curl
curl -s -X POST http://127.0.0.1:8848/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"s_test01",
       "messages":[{"role":"user","content":"画面里有几个人？"}],
       "media":{"data_url":"data:image/jpeg;base64,<BASE64>"}}'
```

返回字段：`reply`（答案）、`objects`（框 + 空间描述）、`trace`（planner 决策、工具调用、`chosen_by`、facts）。

### 3.6.5 复现 3.4 的耗时数据

```bash
# 8 条问题，按路由打印总耗时
python3 /home/cix/agent_latency.py

# 服务内三段日志
grep -E '\[index\]|\[agent\] done|\[prof\]' /home/cix/rexseek-studio/.work_service.log
```

| 日志 | 含义 | 期望 |
|---|---|---|
| `[index] cached=False ms=319 candidates=11` | 预处理 + YOLOE 候选 + 双塔 + RoI/投影（新图首次） | 冷 300–400 ms，热 7–12 ms |
| `[agent] done total_ms=.. plan=.. perc=.. tools=[(tool, ms)] compose=..` | 四段耗时 | 见表 3.4.2 |
| `[prof] cand=.. clip=.. cn=.. proj=.. vis_total=.. llm3b=..` | 视觉与 3B 细项 | `vis_total` 20–35 ms、`llm3b` 390–500 ms |

> profiling 是后加的轻量补丁（只多打一行日志），备份在 `rexseek_service.py.bak.prof`；
> 回滚：`cp rexseek_service.py.bak.prof rexseek_service.py` 后重启 Agent。

### 3.6.6 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `/api/status` 里 agent/ppe 超时、接口 502 | 容器 IP 变了，网关仍指向旧 IP | 用 `start_all_fixed.sh` 重启，或只重启 `start_ui.sh`（已动态取 IP） |
| YOLOE 起不来：`noe_init_context fail` / `libaipudrv.so not found` | 缺 NPU 库路径 | 启动命令必须带 `LD_LIBRARY_PATH=/usr/share/cix/lib` |
| Agent 起不来：`TritonMissing` / inductor 报错 | vLLM 默认走 inductor 编译 | 保持 `compilation_config={"mode":"none","cudagraph_mode":"FULL_DECODE_ONLY"}` |
| `/api/chat` 返回 500 `media.data_url or valid sample_id is required` | `data/samples.json` 为空 | 传 `media.data_url`（浏览器路径自动满足） |
| 首次提问 4–5 s | 新图首次索引 + 模型 warmup | 正常；同图后续 0.6–2 s，同题缓存 19 ms |
| YOLOE 返回 `laptop / chair / remote` 这类标签，score 0.9962 | 用了老的 `y26l.cix`（116ch，分类分支全 0），服务把 `pred[36:116]` 里的 mask 系数当成类分数 | 必须用 split 版：默认已是 `yoloe26l_person_split_i8.cix`；可用 `YOLOE_CIX` 覆盖。备份 `yoloe_service.py.bak.yolo26` |

