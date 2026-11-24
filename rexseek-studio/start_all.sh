#!/bin/bash
# ============================================================
#  Insight 端侧视觉值守系统 · 一键启动
#  板卡 192.168.10.35  →  http://192.168.10.35:8848/
# ============================================================
set -u
ROOT=/home/cix/rexseek-studio
LOG_UI=$ROOT/.work_ui.log

say() { echo "[$(date +%H:%M:%S)] $*"; }

say "1/4 NPU YOLOE 检测服务 (8090)"
if ! pgrep -f "[y]oloe_service.py" >/dev/null 2>&1; then
  ( cd "$ROOT" && setsid nohup python3 yoloe_service.py >> "$ROOT/.work_yoloe.log" 2>&1 & )
  sleep 2
fi

say "2/4 塔吊值守检测服务 (8092，容器 qwen_opt)"
docker exec -d qwen_opt bash "$ROOT/start_ppe.sh"
sleep 4

say "3/4 Agent 推理服务 (9000，容器 qwen_opt)"
docker exec -d qwen_opt bash "$ROOT/start_agent.sh"

say "4/4 前端 + API 网关 (8848)"
setsid nohup "$ROOT/start_ui.sh" </dev/null >/dev/null 2>&1 &

say "等待全部服务就绪（Agent 首次加载约 75 秒）…"
for i in $(seq 1 90); do
  ok=$(curl -s -m 2 http://127.0.0.1:8848/api/status 2>/dev/null \
       | python3 -c "import json,sys;print(1 if json.load(sys.stdin).get('ok') else 0)" 2>/dev/null)
  if [ "${ok:-0}" = "1" ]; then
    say "✅ 全部就绪"
    echo "    值守台   http://192.168.10.35:8848/"
    echo "    对话分析 http://192.168.10.35:8848/studio.html"
    exit 0
  fi
  sleep 2
done
say "❌ 启动超时，检查日志：$ROOT/.work_service.log / .ppe.log / .work_ui.log"
exit 1
