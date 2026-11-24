# Agent 部署手册（后端服务 + 两个前端任务）

> 板卡：CIX P1（172.16.1.35）｜MR-V100 32GB｜NPU X2_1204MP3
> 容器：`s1c`（RexSeek/双塔）、`qwen_opt`（vLLM / Agent）
> 最后验证：2026-11-24 11:06，四个后端 + 三个前端全部 `ok=true`

---

## 1. 端口与进程总览

| 端口 | 进程（在哪） | 用到的算力 | 作用 |
|---|---|---|---|
| **8090** | `yoloe_service.py`（宿主机 cix） | NPU | person 候选生成（YOLOE-26l split INT8） |
| **8092** | `ppe_service.py`（容器 qwen_opt） | GPU（GDINO-tiny） | 值守检测 + 事件聚合（`guard_episodes`） |
| **9000** | `rexseek_service.py`（容器 qwen_opt） | GPU（RexSeek-3B + Qwen2.5-1.5B，vLLM） | **Agent 后端**：规划/工具/选框 |
| **8848** | `server.py --engine proxy`（宿主机） | CPU | **前端任务一：值守台**（index.html + studio.html） |
| **8849** | `server.py --engine proxy --samples … --no-media`（宿主机） | CPU | **前端任务二：通用视觉演示**（番茄/求婚/海报/合影/高速） |
| 8850（可选） | `server.py --engine mock`（宿主机） | 无（纯规则） | 纯对话演示，不加载模型 |

两个前端**共用同一个 Agent（9000）**，所以它们同时可用；区别只是入口页与示例注册表。

---

## 2. 前置条件

| 项 | 要求 |
|---|---|
| 容器 | `s1c`、`qwen_opt` 均为 Up（`docker ps`） |
| 模型 | `/cix/models_orig/IDEA-Research__RexSeek-3B`、`/models/Qwen2.5-3B-RexSeek-LM`、`/cix/rexseek-studio/models/Qwen2.5-1.5B-Instruct`、`/cix/rexseek-studio/models/grounding-dino-tiny` |
| 检测模型 | `/home/cix/yoloe26l_person_split_i8.cix`（split 版，4 输出） |
| NPU 依赖 | host python3.11 + `NOE_Engine`；启动时必须带 `LD_LIBRARY_PATH=/usr/share/cix/lib` |
| 端口空闲 | 8090 / 8092 / 9000 / 8848 / 8849 |
| 显存 | Agent 常驻约 **28 GB / 32 GB**（RexSeek-3B + vLLM 3B + 1.5B + 懒加载 GDINO） |

---

## 3. 启动

### 3.1 一键（推荐，先跑这个）

```bash
bash /home/cix/rexseek-studio/start_all_fixed.sh
```

它会依次拉起 **yoloe(8090) → ppe(8092) → agent(9000) → 前端(8848)**，约 90 秒（Agent 加载 vLLM 约 76 s），最后打印：

```
✅ 全部就绪
    值守台    http://<板卡IP>:8848/
    对话分析  http://<板卡IP>:8848/studio.html
```

### 3.2 加上第二个前端任务（通用视觉演示，8849）

```bash
bash /home/cix/rexseek-studio/start_general_demo.sh
```

做的事：**重启 Agent 并把示例注册表切到 `data/samples_general.json`**，再在 8849 起一个独立前端（`--no-media`）。约 90 秒后打印：

```
✅ 通用演示就绪
    通用视觉演示  http://<板卡IP>:8849/
    番茄示例      http://<板卡IP>:8849/?sample=tomatoes
```

> 为什么要重启 Agent：示例注册表是进程启动时读取的全局配置。值守台走 `media.data_url`（上传帧），**不依赖**示例注册表，所以切换示例集不影响 8848。

### 3.3 可选：纯对话版（8850，不加载模型）

```bash
bash /home/cix/rexseek-studio/start_chat_demo.sh 8850
```

### 3.4 分步启动（排障或单服务重启用）

```bash
ROOT=/home/cix/rexseek-studio
IP=$(docker inspect qwen_opt --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')   # 容器 IP，换容器会变
GW=$(ip -4 addr show nerdctl0 | awk '/inet /{print $2}' | cut -d/ -f1)                              # 宿主桥，通常 10.4.0.1

# 1) NPU 检测服务（8090）
cd $ROOT && LD_LIBRARY_PATH=/usr/share/cix/lib setsid nohup python3 yoloe_service.py >> .work_yoloe.log 2>&1 &

# 2) 值守检测（8092，容器内）
docker exec -d qwen_opt bash -lc "cd /cix/rexseek-studio && GUARD_GRACE_S=15 exec python3 -u ppe_service.py > /cix/rexseek-studio/.ppe.log 2>&1"

# 3) Agent 后端（9000，容器内）
docker exec -d qwen_opt bash -lc "cd /cix/rexseek-studio && \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 REXSEEK_FROM_CONFIG=1 REXSEEK_USE_VLLM=1 VLLM_GPU_UTIL=0.35 \
  YOLOE_CLASS_AGNOSTIC=1 YOLOE_SCORE_MIN=0.6 REXSEEK_LM=/models/Qwen2.5-3B-RexSeek-LM \
  REXSEEK_REPO=/cix/rexseek-studio/vendor/repo REXSEEK_MODEL=/cix/models_orig/IDEA-Research__RexSeek-3B \
  REXSEEK_GDINO=/cix/rexseek-studio/models/grounding-dino-tiny \
  REXSEEK_CHAT_MODEL=/cix/rexseek-studio/models/Qwen2.5-1.5B-Instruct \
  REXSEEK_SAMPLES=/cix/rexseek-studio/data/samples_general.json \
  YOLOE_URL=http://$GW:8090 PPE_URL=http://127.0.0.1:8092; \
  exec python3 -u rexseek_service.py --host 0.0.0.0 --port 9000 > /cix/rexseek-studio/.work_service.log 2>&1"

# 4) 前端一：值守台（8848）
cd $ROOT && setsid nohup python3 -u server.py --engine proxy \
  --endpoint "http://$IP:9000" --ppe-endpoint "http://$IP:8092" \
  --host 0.0.0.0 --port 8848 >> .work_ui.log 2>&1 &

# 5) 前端二：通用演示（8849）
cd $ROOT && setsid nohup python3 -u server.py --engine proxy \
  --endpoint "http://$IP:9000" --ppe-endpoint "http://$IP:8092" \
  --samples "$ROOT/data/samples_general.json" --no-media \
  --host 0.0.0.0 --port 8849 >> .work_ui_general.log 2>&1 &
```

> 单服务快捷脚本：`start_agent.sh`（9000）、`start_ppe.sh`（8092）、`start_ui.sh`（8848）——均已改成运行期解析容器 IP / 桥接 IP，可直接跑。

---

## 4. 验收

### 4.1 命令行校验

```bash
# 8848：四个后端服务都应 ok
curl -s http://127.0.0.1:8848/api/status | python3 -m json.tool
# 期望: "ok": true，gateway/agent/ppe/yoloe 全 true；
#       yoloe.info.model = yoloe26l_person_split_i8.cix
#       agent.info.model = "RexSeek-3B + Qwen2.5-1.5B"（llm=vLLM）

# 三个前端的健康
for p in 8848 8849 8850; do echo -n "$p: "; curl -s http://127.0.0.1:$p/api/health | head -c 120; echo; done
# 期望: 8848/8849 name=proxy, 8850 name=mock

# 通用演示的示例注册表
curl -s http://127.0.0.1:8849/api/samples | python3 -c "import json,sys;print([s['id'] for s in json.load(sys.stdin)['samples']])"
# 期望: ['tomatoes','proposal','webinar','team','highway']

# NPU 检测自检（容器内访问宿主桥）
docker exec -e YOLOE_URL=http://10.4.0.1:8090/detect -w /cix s1c python3 /cix/yoloe_probe_svc.py
# 期望: person score≈0.75，3 个框，npu_ms≈42
```

### 4.2 浏览器验收

| 前端 | 打开 | 验收动作 |
|---|---|---|
| 值守台 8848 | `http://<板卡IP>:8848/` | 默认视频 → 框选危险区 → 开始值守 → 台账出现事件；点「追问」跳到对话分析 |
| 对话分析 | `http://<板卡IP>:8848/studio.html` | 问「画面里有几个人？」应回答 3 人（示例图 helmet_all 时为 3） |
| 通用演示 8849 | `http://<板卡IP>:8849/?sample=tomatoes` | 问「哪些番茄已经熟了？」→ 2 个；「画面里一共有几颗番茄？」→ 8 个 |
| 纯对话 8850 | `http://<板卡IP>:8850/` | 任意示例，回答在 100 ms 内（规则路由，不加载模型） |

---

## 5. 停止与重启

```bash
# 全部停止
docker exec qwen_opt pkill -f '[r]exseek_service' || true
docker exec qwen_opt pkill -f '[V]LLM::EngineCore' || true
docker exec qwen_opt pkill -f '[p]pe_service' || true
pkill -f '[y]oloe_service.py' || true
pkill -f 'server.py' || true

# 只重启 Agent（约 76 s）
docker exec -d qwen_opt bash /cix/rexseek-studio/start_agent.sh

# 只重启某个前端
pkill -f 'server.py.*--port 8849' && bash /home/cix/rexseek-studio/start_general_demo.sh
```

> 停 Agent 后务必确认 `VLLM::EngineCore` 也被清掉，否则显存残留会导致下次启动失败（`start_agent.sh` 已内置这段清理）。

---

## 6. 换检测模型 / 调阈值

| 目的 | 做法 |
|---|---|
| 换 YOLOE 模型 | `LD_LIBRARY_PATH=/usr/share/cix/lib YOLOE_CIX=/path/to/x.cix setsid nohup python3 yoloe_service.py …`（默认 `yoloe26l_person_split_i8.cix`） |
| 调检测置信度 | 环境变量 `YOLOE_SCORE_MIN`（Agent 侧过滤，默认 0.6）；服务侧 `CONF=0.25` |
| 切换示例集 | Agent 的 `REXSEEK_SAMPLES`（默认 `data/samples_general.json`）；前端 `--samples` 参数 |

---

## 7. 常见故障

| 现象 | 原因 | 处理 |
|---|---|---|
| 8848 里 agent/ppe 超时、接口 502 | 容器 IP 变了，前端 endpoint 仍指旧 IP | 用 `start_all_fixed.sh`，或重启 `start_ui.sh`（已动态取 IP） |
| YOLOE 起不来：`libaipudrv.so not found` / `noe_init_context fail` | 缺 NPU 库路径 | 启动命令加 `LD_LIBRARY_PATH=/usr/share/cix/lib` |
| Agent 起不来：`TritonMissing` / inductor 报错 | vLLM 默认走 inductor 编译 | 保持 `compilation_config={"mode":"none","cudagraph_mode":"FULL_DECODE_ONLY"}` |
| 检测返回 `laptop/chair` 标签、score 0.9962 | 用了老 `y26l.cix`（分类分支全 0） | 用 split 版模型（`YOLOE_CIX` 默认已是） |
| 问番茄答「没有定位到匹配目标」 | 开放词表提示走了 person-only 的 YOLOE | 已按提示词路由：纯 person 走 NPU，含其它类名走 GDINO（懒加载） |
| `/api/chat` 报 `media.data_url or valid sample_id is required` | 直接调 API 但没给图 | 传 `media.data_url`，或用 8849 的 `sample_id` |
| 首次提问 4–5 s | 新图首次索引 + 模型 warmup | 正常；同图后续 0.6–2 s |

---

## 8. 日志与文件

| 文件 | 内容 |
|---|---|
| `/home/cix/rexseek-studio/.work_service.log` | Agent（含 `[index]`、`[agent] done`、`[prof]` 计时行） |
| `.ppe.log` | 值守检测/事件 |
| `.work_yoloe.log` | NPU 检测服务 |
| `.work_ui.log` / `.work_ui_general.log` / `.work_ui_chat.log` | 三个前端的访问日志 |
| `data/samples_general.json` | 通用示例（番茄等 5 个） |
| `data/media.json` | 值守台视频素材 |
| `data/samples.json` | 值守台示例目录（空，不影响） |

---

## 9. 重启板卡后的恢复顺序

```bash
# 约 3 分钟恢复全部
bash /home/cix/rexseek-studio/start_all_fixed.sh      # 后端 + 8848
bash /home/cix/rexseek-studio/start_general_demo.sh   # 8849（会重启 Agent 切示例集）
bash /home/cix/rexseek-studio/start_chat_demo.sh 8850 # 可选
```
