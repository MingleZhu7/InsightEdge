#!/bin/bash
# 重启 RexSeek Agent 服务（容器 qwen_opt 内，:9000）
# 约 30-60 秒不可用
pkill -f '^python3 -u rexseek_service' 2>/dev/null
# 旧进程被 kill 后 EngineCore 子进程可能残留并占着显存，必须一起清掉
pkill -f '[V]LLM::EngineCore' 2>/dev/null
for i in $(seq 1 15); do
  pgrep -f '[V]LLM::EngineCore' >/dev/null || break
  sleep 1
done
sleep 2
cd /cix/rexseek-studio || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export REXSEEK_FROM_CONFIG=1 REXSEEK_USE_VLLM=1 VLLM_GPU_UTIL=0.35
export YOLOE_CLASS_AGNOSTIC=1 YOLOE_SCORE_MIN=0.6
export REXSEEK_LM=/models/Qwen2.5-3B-RexSeek-LM
export REXSEEK_REPO=/cix/rexseek-studio/vendor/repo
export REXSEEK_MODEL=/cix/models_orig/IDEA-Research__RexSeek-3B
export REXSEEK_GDINO=/cix/rexseek-studio/models/grounding-dino-tiny
export REXSEEK_CHAT_MODEL=/cix/rexseek-studio/models/Qwen2.5-1.5B-Instruct
# 不设 QWEN_URL：planner/翻译/组织语言走进程内的 Qwen2.5-1.5B（GPU）
# 如需退回 CPU 0.8B，把 QWEN_URL=http://172.17.0.1:8081 加回来
export YOLOE_URL="${YOLOE_URL:-http://10.4.0.1:8090}" PPE_URL="${PPE_URL:-http://127.0.0.1:8092}"
exec python3 -u rexseek_service.py --host 0.0.0.0 --port 9000 > /cix/rexseek-studio/.work_service.log 2>&1
