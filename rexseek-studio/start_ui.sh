#!/bin/bash
# 重启前端 + API 网关（8848）
pkill -f '^python3 -u (.*/)?server\.py' 2>/dev/null
sleep 1
cd /home/cix/rexseek-studio || exit 1
# 容器 IP 运行期读取（换容器后会变，不能写死）
IP=$(docker inspect qwen_opt --format "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}")
exec python3 -u server.py \
  --engine proxy \
  --endpoint "http://${IP}:9000" \
  --ppe-endpoint "http://${IP}:8092" \
  --host 0.0.0.0 --port 8848 \
  >> /home/cix/rexseek-studio/.work_ui.log 2>&1
