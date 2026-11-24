#!/bin/bash
# 重启塔吊值守检测服务（容器 qwen_opt 内运行）
# 一次进入 = 一条事件；离开判定迟滞由 GUARD_GRACE_S 控制（默认 15 秒）
pkill -f '^python3 -u ppe_service' 2>/dev/null
sleep 2
cd /cix/rexseek-studio || exit 1
export GUARD_GRACE_S="${GUARD_GRACE_S:-15}"
exec python3 -u ppe_service.py > /cix/rexseek-studio/.ppe.log 2>&1
