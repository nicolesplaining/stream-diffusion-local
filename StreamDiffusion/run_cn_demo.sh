#!/usr/bin/env bash
# DGX Spark — ControlNet (depth) demo: full transformation that follows your body.
# Type any prompt (cat, zombie, cyborg, alien...) and it transforms you while tracking motion.
set -e
cd "$(dirname "$0")"
source ../.venv/bin/activate
pkill -9 -f controlnet_server.py 2>/dev/null || true
pkill -9 -f compare_server.py 2>/dev/null || true
sleep 2
export CN_RES=384 CN_STEPS=2 CN_SCALE=0.3 CN_COMPILE=1   # full VAE + compile = sharp + fast
echo "Starting ControlNet demo (depth-locked, transform + track)..."
echo "Open http://localhost:8000 on the Spark."
exec python web/controlnet_server.py
