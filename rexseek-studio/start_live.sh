#!/usr/bin/env bash
# 启动真实 RexSeek-3B 推理服务（9000）+ UI（8848）
set -euo pipefail
cd "$(dirname "$0")"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

REXSEEK_PY="${REXSEEK_PY:-python3}"
REXSEEK_PORT="${REXSEEK_PORT:-9000}"
UI_PORT="${UI_PORT:-8848}"

"$REXSEEK_PY" rexseek_service.py --port "$REXSEEK_PORT" &
SERVICE_PID=$!
trap 'kill "$SERVICE_PID" 2>/dev/null || true' EXIT

echo "等待 RexSeek-3B 加载（首次约 6 秒）..."
for _ in $(seq 1 90); do
  if python3 - "$REXSEEK_PORT" <<'PY' 2>/dev/null
import sys, urllib.request
urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=1)
PY
  then
    break
  fi
  sleep 1
done

python3 server.py --engine proxy --endpoint "http://127.0.0.1:$REXSEEK_PORT" --port "$UI_PORT" "$@"

