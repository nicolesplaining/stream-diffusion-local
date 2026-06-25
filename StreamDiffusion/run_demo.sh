#!/usr/bin/env bash
# DGX Spark cyborg demo — fixed config: Absolute Reality, strength 21,40, tuned for tracking.
set -e
cd "$(dirname "$0")"
source ../.venv/bin/activate
pkill -9 -f compare_server.py 2>/dev/null || true
sleep 2
export ONLY=hyperreal SD_T_INDEX=20,40 SD_GUIDANCE=1.4 NO_LIVE2DIFF=1
echo "Starting DGX Spark demo (Absolute Reality, strength 21,40)…"
echo "Open http://localhost:8000 on the Spark."
exec python web/compare_server.py
