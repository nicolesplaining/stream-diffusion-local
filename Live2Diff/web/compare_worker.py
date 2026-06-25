"""
Headless Live2Diff inference worker for the comparison grid.

Runs in the .venv-live2diff venv (incompatible diffusers, stateful kv-cache), so
it lives in its own process. The main compare_server.py spawns it and talks to it
over localhost HTTP.

  GET  /health  -> {"ready": true}   (model + TRT engines loaded)
  POST /infer   {"image": dataurl, "prompt": str}
       -> {"warming": n, "need": N}  while filling the warmup window
       -> {"status": "ready"}        once warmed
       -> {"image": dataurl}         per streamed frame
"""
import os, sys, base64, threading
from io import BytesIO

import numpy as np
import torch
from PIL import Image
import uvicorn
from fastapi import FastAPI, Request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Live2Diff/
sys.path.insert(0, REPO_ROOT)
from live2diff.utils.wrapper import StreamAnimateDiffusionDepthWrapper

W = H = 384
WARMUP_N = 8
PORT = int(os.environ.get("L2D_PORT", 8002))
DEFAULT_PROMPT = "a photorealistic portrait of a person, natural light, sharp focus"

print("[worker] loading Live2Diff (TensorRT 384, 2-step) ...", flush=True)
stream = StreamAnimateDiffusionDepthWrapper(
    few_step_model_type="lcm",
    config_path=os.path.join(REPO_ROOT, "configs", "spike.yaml"),
    cfg_type="none", strength=None, num_inference_steps=50,
    t_index_list=[31, 43], frame_buffer_size=1, width=W, height=H,
    acceleration="tensorrt", do_add_noise=True, output_type="pt",
    use_denoising_batch=True, use_tiny_vae=True,
    engine_dir=os.path.join(REPO_ROOT, "engines_trt384"), seed=42)
gpu_lock = threading.Lock()
state = {"warmup": [], "prepared": False, "last_prompt": None}
print("[worker] Live2Diff ready.", flush=True)


def _decode(data_url):
    raw = base64.b64decode(data_url.split(",", 1)[-1])
    img = Image.open(BytesIO(raw)).convert("RGB").resize((W, H))
    return torch.from_numpy(np.array(img)).float().div(255).permute(2, 0, 1).contiguous()


def _encode(t):
    arr = (t.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


app = FastAPI()


@app.get("/health")
def health():
    return {"ready": True}


@app.post("/infer")
async def infer(req: Request):
    data = await req.json()
    image = data.get("image")
    prompt = (data.get("prompt") or "").strip() or DEFAULT_PROMPT
    if not image:
        return {"error": "no image"}
    frame = _decode(image)
    if not state["prepared"]:
        state["warmup"].append(frame)
        n = len(state["warmup"])
        if n >= WARMUP_N:
            with gpu_lock:
                stream.prepare(warmup_frames=torch.stack(state["warmup"]),
                               prompt=prompt, guidance_scale=1)
            state["prepared"] = True
            state["last_prompt"] = prompt
            return {"status": "ready"}
        return {"warming": n, "need": WARMUP_N}
    # Live2Diff update_prompt is broken upstream (_encode_prompt kwarg mismatch),
    # so we never push prompt changes mid-stream; it keeps its warm-up prompt.
    sp = None
    state["last_prompt"] = prompt
    with gpu_lock:
        out = stream(image=frame, prompt=sp)
    return {"image": _encode(out)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
