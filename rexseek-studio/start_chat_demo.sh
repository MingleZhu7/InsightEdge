#!/bin/bash
# 纯对话版（mock 引擎，不加载任何模型）：通用示例集 + 独立端口 8850
set -u
ROOT=/home/cix/rexseek-studio
PORT="${1:-8850}"
say() { echo "[$(date +%H:%M:%S)] $*"; }

pkill -f "server.py.*--port ${PORT}" 2>/dev/null || true
sleep 1
cd "$ROOT"
setsid nohup python3 -u server.py \
  --engine mock \
  --samples "$ROOT/data/samples_general.json" --no-media \
  --host 0.0.0.0 --port "$PORT" >> "$ROOT/.work_ui_chat.log" 2>&1 &
say "started mock chat demo on :$PORT"
for _ in $(seq 1 10); do
  if curl -s -m 2 "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then
    IPHOST=$(hostname -I | awk '{print $1}')
    say "✅ 纯对话版就绪"
    echo "    纯对话演示     http://${IPHOST}:${PORT}/"
    echo "    番茄示例       http://${IPHOST}:${PORT}/?sample=tomatoes"
    exit 0
  fi
  sleep 1
done
say "❌ 启动失败，见 $ROOT/.work_ui_chat.log"
exit 1
