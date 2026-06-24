"""
Real-time webcam app for Live2Diff (temporally-coherent video-to-video).

Browser captures the webcam, this server runs the streaming AnimateDiff-Depth
pipeline (TensorRT, 384x384, 2 steps) and streams frames back. Live prompt box.

Run (Live2Diff repo root, venv active):
    source ../.venv-live2diff/bin/activate
    python web/server.py
    # open http://localhost:8001 on the Spark, or tunnel it:
    #   ssh -L 8001:localhost:8001 sp9@<spark>   then open localhost:8001 on your laptop

Notes:
- Live2Diff is STATEFUL (kv-cache): one streaming session at a time. Each browser
  connection re-warms the pipeline with the first 8 frames before output starts.
- TensorRT engines load from engines_trt384/ (built earlier); first ever run builds
  them (~3 min). Startup loads the model+engines, takes ~30-60s.
"""

import os
import sys
import base64
import asyncio
import threading
from io import BytesIO

import numpy as np
import torch
from PIL import Image
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from live2diff.utils.wrapper import StreamAnimateDiffusionDepthWrapper

W = H = 384
WARMUP_N = 8
DEFAULT_PROMPT = "a photorealistic portrait of a person, natural light, sharp focus"

HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("Loading Live2Diff pipeline (TensorRT, 384, 2-step) — this takes ~30-60s ...", flush=True)
stream = StreamAnimateDiffusionDepthWrapper(
    few_step_model_type="lcm",
    config_path=os.path.join(REPO_ROOT, "configs", "spike.yaml"),
    cfg_type="none",
    strength=None,
    num_inference_steps=50,
    t_index_list=[31, 43],
    frame_buffer_size=1,
    width=W, height=H,
    acceleration="tensorrt",
    do_add_noise=True,
    output_type="pt",
    use_denoising_batch=True,
    use_tiny_vae=True,
    engine_dir=os.path.join(REPO_ROOT, "engines_trt384"),
    seed=42,
)
gpu_lock = threading.Lock()
print("Live2Diff ready.", flush=True)

app = FastAPI()


@app.get("/")
async def index():
    with open(HTML_PATH) as f:
        return HTMLResponse(f.read())


def _decode(data_url: str) -> torch.Tensor:
    raw = base64.b64decode(data_url.split(",", 1)[-1])
    img = Image.open(BytesIO(raw)).convert("RGB").resize((W, H))
    arr = torch.from_numpy(np.array(img)).float() / 255.0   # [H,W,3]
    return arr.permute(2, 0, 1).contiguous()                # [3,H,W]


def _encode(t: torch.Tensor) -> str:
    arr = (t.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _prepare(frames, prompt):
    with gpu_lock:
        stream.prepare(warmup_frames=torch.stack(frames), prompt=prompt, guidance_scale=1)


def _infer(frame, prompt):
    with gpu_lock:
        return stream(image=frame, prompt=prompt)


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    warmup = []
    prepared = False
    last_prompt = None
    try:
        while True:
            data = await websocket.receive_json()
            prompt = (data.get("prompt") or "").strip() or DEFAULT_PROMPT
            image = data.get("image")
            if not image:
                continue
            frame = _decode(image)

            if not prepared:
                warmup.append(frame)
                await websocket.send_json(
                    {"type": "status", "message": f"Warming up… {len(warmup)}/{WARMUP_N}"})
                if len(warmup) >= WARMUP_N:
                    await asyncio.to_thread(_prepare, warmup, prompt)
                    prepared = True
                    last_prompt = prompt
                    await websocket.send_json({"type": "status", "message": "Streaming"})
                continue

            send_prompt = prompt if prompt != last_prompt else None
            last_prompt = prompt
            out = await asyncio.to_thread(_infer, frame, send_prompt)
            await websocket.send_json({"type": "frame", "image": _encode(out)})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws] error: {e}", flush=True)


if __name__ == "__main__":
    print("Open http://localhost:8001 on the Spark (or tunnel the port to your laptop).")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
