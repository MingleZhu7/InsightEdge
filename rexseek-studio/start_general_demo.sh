#!/bin/bash
# 启动"通用视觉演示"（番茄/求婚/海报/合影/高速）——独立端口 8849，不影响 8848 值守台
set -u
ROOT=/home/cix/rexseek-studio
IP=$(docker inspect qwen_opt --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
GW=$(ip -4 addr show nerdctl0 | awk '/inet /{print $2}' | cut -d/ -f1)
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "1/2 Agent 重启，示例集换成通用集（samples_general.json）"
docker exec qwen_opt pkill -f '[r]exseek_service' 2>/dev/null || true
docker exec qwen_opt pkill -f '[V]LLM::EngineCore' 2>/dev/null || true
sleep 3
docker exec -d qwen_opt bash -lc "cd /cix/rexseek-studio && \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 REXSEEK_FROM_CONFIG=1 REXSEEK_USE_VLLM=1 VLLM_GPU_UTIL=0.35 \
  YOLOE_CLASS_AGNOSTIC=1 YOLOE_SCORE_MIN=0.6 REXSEEK_LM=/models/Qwen2.5-3B-RexSeek-LM \
  REXSEEK_REPO=/cix/rexseek-studio/vendor/repo REXSEEK_MODEL=/cix/models_orig/IDEA-Research__RexSeek-3B \
  REXSEEK_GDINO=/cix/rexseek-studio/models/grounding-dino-tiny \
  REXSEEK_CHAT_MODEL=/cix/rexseek-studio/models/Qwen2.5-1.5B-Instruct \
  REXSEEK_SAMPLES=/cix/rexseek-studio/data/samples_general.json \
  YOLOE_URL=http://$GW:8090 PPE_URL=http://127.0.0.1:8092; \
  exec python3 -u rexseek_service.py --host 0.0.0.0 --port 9000 > /cix/rexseek-studio/.work_service.log 2>&1"

say "2/2 通用演示前端（8849）"
pkill -f 'server.py --samples' 2>/dev/null || true
sleep 1
cd "$ROOT"
setsid nohup python3 -u server.py --engine proxy \
  --endpoint "http://$IP:9000" --ppe-endpoint "http://$IP:8092" \
  --samples "$ROOT/data/samples_general.json" --no-media \
  --host 0.0.0.0 --port 8849 >> "$ROOT/.work_ui_general.log" 2>&1 &

say "等待 Agent 就绪（约 80 秒）…"
for _ in $(seq 1 50); do
  ok=$(curl -s -m 3 http://127.0.0.1:8849/api/health 2>/dev/null | python3 -c 'import json,sys;print(1 if json.load(sys.stdin) else 0)' 2>/dev/null)
  ag=$(curl -s -m 3 "http://$IP:9000/api/health" 2>/dev/null | python3 -c 'import json,sys;print(1 if json.load(sys.stdin).get("ok") else 0)' 2>/dev/null)
  if [ "${ok:-0}" = "1" ] && [ "${ag:-0}" = "1" ]; then
    IPHOST=$(hostname -I | awk '{print $1}')
    say "✅ 通用演示就绪"
    echo "    通用视觉演示  http://${IPHOST}:8849/"
    echo "    番茄示例      http://${IPHOST}:8849/?sample=tomatoes"
    echo "    （值守台仍在 http://${IPHOST}:8848/）"
    exit 0
  fi
  sleep 3
done
say "❌ 超时：检查 $ROOT/.work_ui_general.log 与 .work_service.log"
exit 1
