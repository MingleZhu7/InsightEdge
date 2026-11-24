#!/usr/bin/env bash
# 从本仓库部署到板卡并启动（后端 + 前端一）
# 用法：BOARD=172.16.1.35 bash deploy.sh [also-general]
set -euo pipefail

BOARD="${BOARD:-172.16.1.35}"
USER_="${USER_:-cix}"
ROOT=/home/cix/rexseek-studio
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "→ 同步代码到 ${USER_}@${BOARD}:${ROOT}"
ssh "${USER_}@${BOARD}" "mkdir -p ${ROOT}"
rsync -az --delete \
  --exclude 'data/*.db*' --exclude 'web/media/' --exclude 'web/data/' \
  --exclude '__pycache__/' --exclude '*.log' \
  "${HERE}/rexseek-studio/" "${USER_}@${BOARD}:${ROOT}/"

echo "→ 启动后端 + 值守台（:8848）"
ssh "${USER_}@${BOARD}" "bash ${ROOT}/start_all_fixed.sh"

if [ "${1:-}" = "also-general" ]; then
  echo "→ 启动通用演示（:8849，会重启 Agent 切换示例集）"
  ssh "${USER_}@${BOARD}" "bash ${ROOT}/start_general_demo.sh"
fi

echo "✅ 完成：http://${BOARD}:8848/  (通用演示加参数 also-general 后为 :8849)"
