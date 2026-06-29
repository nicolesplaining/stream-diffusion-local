# Live Diffusion Demo — StreamDiffusion on DGX Spark

Real-time webcam → stylized video. A browser shows the webcam running through
StreamDiffusion (img2img) at ~22 FPS using **1-step denoising + TensorRT** on the
default model **Realistic Vision V5.1**. Everything runs on the Spark.

> Shares the single GB10 GPU with the RL booth — **only one demo runs at a time.**
> Starting this stops the booth automatically (and vice-versa).

---

## Quick start

```bash
~/stream-diffusion-local/start_diffusion.sh
```

Then, **in a browser running on the Spark**, open **http://localhost:8000**, pick the
style, click **Start**, and allow the camera. (The first frame warms the TensorRT
pipeline for ~10–20 s, then it's live.)

To switch back to the RL booth:

```bash
systemctl --user start booth-demo      # auto-stops the diffusion demo
```

---

## Requirements / caveats

- **A USB webcam must be plugged into the Spark.** The browser's `getUserMedia` only
  grants the camera on `localhost` (or HTTPS), so the browser must run **on the Spark**
  — which is where the camera must be. A remote `http://<spark-ip>:8000` page loads but
  can't access a remote camera. (There is no camera attached by default — plug one in.)
- **One demo at a time** (single GPU). Mutual exclusion is enforced by systemd
  (`Conflicts=` between `diffusion-demo.service` and `booth-demo.service`).
- **Not autostarted on boot** — the RL booth is the boot demo; this is on-demand.

---

## Managing the service

```bash
systemctl --user status  diffusion-demo
journalctl --user -u diffusion-demo -f
systemctl --user stop    diffusion-demo
systemctl --user start   diffusion-demo     # stops the RL booth
```

---

## Tuning

Config is via environment variables in `~/.config/systemd/user/diffusion-demo.service`
(`Environment=…`). After editing: `systemctl --user daemon-reload && systemctl --user restart diffusion-demo`.

| Var              | Current  | Meaning                                                      |
|------------------|----------|-------------------------------------------------------------|
| `SD_ACCELERATION`| tensorrt | `tensorrt` (fast) or `none` (no engines needed, ~9 FPS)     |
| `SD_T_INDEX`     | 40       | denoising steps. **`40` = 1 step (fastest).** More values   |
|                  |          | (e.g. `22,32,45`) = more steps = more stylized but slower.  |
| `SD_WIDTH/HEIGHT`| 512      | frame size.                                                 |

- **Model / styles** live in `StreamDiffusion/web/server.py` (`STYLES` dict). It's
  currently a single model — **Realistic Vision V5.1** — to keep one GPU pipeline and
  avoid on-the-fly engine builds. The previous multi-style set is in git history.
- **Prompt:** type in the page; it's appended to the style's prompt suffix.

### TensorRT engines (important re: disk)
- `SD_ACCELERATION=tensorrt` needs a prebuilt engine **for the exact model + step-count
  (batch) + size**. Realistic Vision @ 1-step (batch-1) @ 512 is **already built** in
  `StreamDiffusion/engines/SG161222/…`.
- Changing the **model** or the **number of steps** (`SD_T_INDEX` count) triggers a
  one-time engine build (~1–2 min, a few GB). **Disk runs tight** — check `df -h /`
  first. The `.onnx` build intermediates in `engines/` can be deleted after a build
  (only the `.engine` files are needed at runtime):
  `find StreamDiffusion/engines -name '*.onnx' -delete`.

---

## How it runs

- `start_diffusion.sh` → `systemctl --user start diffusion-demo` → runs
  `web/server.py` (uvicorn, FastAPI, port **8000**, binds `0.0.0.0`).
- The browser captures the webcam, streams frames over a websocket; the server runs
  each frame through StreamDiffusion img2img and streams the stylized frame back.
- Full environment/port-of-TensorRT-to-CUDA-13 notes are in `RUNNING.md`.

---

## Troubleshooting

- **Page loads but no output:** make sure the browser is on the Spark (`localhost:8000`)
  and you allowed the camera; check `journalctl --user -u diffusion-demo -f`.
- **Slow / ~9 FPS:** it fell back to `SD_ACCELERATION=none` (engine missing for the
  current model/steps). Rebuild the engine or revert `SD_T_INDEX`/model to the prebuilt combo.
- **Booth won't start:** diffusion is still running — `systemctl --user start booth-demo`
  stops it via `Conflicts=`; give it a few seconds.
