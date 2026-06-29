#!/usr/bin/env bash
# One command to run the live diffusion demo (StreamDiffusion web UI).
# Mutually exclusive with the RL booth on the single GPU: starting this service
# auto-stops the booth (systemd Conflicts=), and starting the booth stops this.
set -u
SVC=diffusion-demo.service

echo "[start_diffusion] starting diffusion demo (this stops the RL booth automatically)..."
systemctl --user start "$SVC"

printf "[start_diffusion] bringing up the web server "
ok=0
for i in $(seq 1 40); do
  if ss -tln 2>/dev/null | grep -q ":8000\b" && curl -fsS -m2 http://localhost:8000/ >/dev/null 2>&1; then ok=1; break; fi
  printf "."; sleep 2
done
printf "\n"

LAN=$(hostname -I 2>/dev/null | awk "{print \$1}")
if [ "$ok" = 1 ]; then echo "[start_diffusion] web UI is UP."; else
  echo "[start_diffusion] not up yet — check: journalctl --user -u diffusion-demo -f"; fi
cat <<URLS

  open in a browser ON THE SPARK (getUserMedia needs localhost or HTTPS):
    http://localhost:8000          (LAN, view only: http://$LAN:8000)

  A USB webcam must be plugged into the Spark. In the page: pick the style,
  click Start, allow the camera. First frame warms the TensorRT pipeline (~10-20s),
  then it runs live (~22 FPS, 1-step Realistic Vision).

  manage:
    logs:              journalctl --user -u diffusion-demo -f
    stop:              systemctl --user stop diffusion-demo
    back to RL booth:  systemctl --user start booth-demo     (auto-stops diffusion)
URLS
