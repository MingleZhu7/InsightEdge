#!/bin/bash
# ============================================================
#  Insight 端侧视觉值守系统 · 一键启动（修正版）
#  修正点：
#   1) 容器 IP 不再写死 10.4.0.35，改为运行时从 docker 读取（换容器后会变）
#   2) NPU YOLOE 服务需要 LD_LIBRARY_PATH=/usr/share/cix/lib，否则 libaipudrv.so 找不到
#   3) 容器访问宿主的桥接地址改为 nerdctl0 的实际 IP（原来写 172.17.0.1 不可达）
# ============================================================
set -u
ROOT=/home/cix/rexseek-studio
GW_BRIDGE=$(ip -4 addr show nerdctl0 | awk '/inet /{print $2}' | cut -d/ -f1)
IP=$(docker inspect qwen_opt --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
say() { echo "[$(date +%H:%M:%S)] $*"; }
say "容器 qwen_opt=$IP | 宿主桥 $GW_BRIDGE"

say "1/4 NPU YOLOE 检测服务 (8090)"
pkill -f '[y]oloe_service.py' 2>/dev/null || true
sleep 1
(cd "$ROOT" && LD_LIBRARY_PATH=/usr/share/cix/lib setsid nohup python3 yoloe_service.py >> "$ROOT/.work_yoloe.log" 2>&1 &)

say "2/4 值守检测服务 (8092, 容器)"
docker exec qwen_opt pkill -f '[p]pe_service.py' 2>/dev/null || true
sleep 1
docker exec -d qwen_opt bash -lc "cd /cix/rexseek-studio && GUARD_GRACE_S=15 exec python3 -u ppe_service.py > /cix/rexseek-studio/.ppe.log 2>&1"

say "3/4 Agent 推理服务 (9000, 容器)"
docker exec qwen_opt pkill -f '[r]exseek_service' 2>/dev/null || true
docker exec qwen_opt pkill -f '[V]LLM::EngineCore' 2>/dev/null || true
sleep 2
docker exec -d qwen_opt bash -lc "cd /cix/rexseek-studio && \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 REXSEEK_FROM_CONFIG=1 REXSEEK_USE_VLLM=1 VLLM_GPU_UTIL=0.35 \
  YOLOE_CLASS_AGNOSTIC=1 YOLOE_SCORE_MIN=0.6 REXSEEK_LM=/models/Qwen2.5-3B-RexSeek-LM \
  REXSEEK_REPO=/cix/rexseek-studio/vendor/repo REXSEEK_MODEL=/cix/models_orig/IDEA-Research__RexSeek-3B \
  REXSEEK_GDINO=/cix/rexseek-studio/models/grounding-dino-tiny \
  REXSEEK_CHAT_MODEL=/cix/rexseek-studio/models/Qwen2.5-1.5B-Instruct \
  YOLOE_URL=http://$GW_BRIDGE:8090 PPE_URL=http://127.0.0.1:8092; \
  exec python3 -u rexseek_service.py --host 0.0.0.0 --port 9000 > /cix/rexseek-studio/.work_service.log 2>&1"

say "4/4 前端 + API 网关 (8848)"
pkill -f '^python3 -u (.*/)?server\.py' 2>/dev/null || true
sleep 1
(cd "$ROOT" && setsid nohup python3 -u server.py --engine proxy \
  --endpoint "http://$IP:9000" --ppe-endpoint "http://$IP:8092" \
  --host 0.0.0.0 --port 8848 >> "$ROOT/.work_ui.log" 2>&1 &)

say "等待全部服务就绪（Agent 首次加载约 80 秒）…"
for _ in $(seq 1 60); do
  ok=$(curl -s -m 3 http://127.0.0.1:8848/api/status 2>/dev/null \
       | python3 -c "import json,sys;print(1 if json.load(sys.stdin).get('ok') else 0)" 2>/dev/null)
  if [ "${ok:-0}" = "1" ]; then
    IPHOST=$(hostname -I | awk '{print $1}')
    say "✅ 全部就绪"
    echo "    值守台    http://${IPHOST}:8848/"
    echo "    对话分析  http://${IPHOST}:8848/studio.html"
    exit 0
  fi
  sleep 3
done
say "❌ 启动超时：检查 $ROOT/.work_service.log / .ppe.log / .work_ui.log"
exit 1
