"""
ControlNet (depth) real-time webcam demo — full transformation that follows your body.

Prompt drives the transform (cat, zombie, cyborg...); per-frame DEPTH conditioning
tracks your pose. Sharp + fast: full VAE, channels_last, torch.compile, few-step LCM.

Run:  python web/controlnet_server.py   (open http://localhost:8000)
Env:  CN_RES(512) CN_STEPS(2) CN_GUIDANCE(1.5) CN_SCALE(0.4) CN_BASE CN_PROMPT CN_COMPILE(1)
"""
import os, base64, asyncio, threading, time
from io import BytesIO
import numpy as np, torch
from PIL import Image
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, LCMScheduler
from transformers import pipeline as hf_pipeline

torch.set_float32_matmul_precision("high")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = int(os.environ.get("CN_RES", 512))
STEPS = int(os.environ.get("CN_STEPS", 2))
GUID = float(os.environ.get("CN_GUIDANCE", 1.5))
SCALE = float(os.environ.get("CN_SCALE", 0.4))
BASE = os.environ.get("CN_BASE", "Lykon/absolute-reality-1.81")
COMPILE = os.environ.get("CN_COMPILE", "1") == "1"
DEFAULT_PROMPT = os.environ.get("CN_PROMPT", "a fluffy cat, feline face, whiskers, fur, cute, detailed")

print(f"Loading ControlNet demo (res={RES}, steps={STEPS}, compile={COMPILE})...", flush=True)
depth_est = hf_pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=0)
cn = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE, controlnet=cn, torch_dtype=torch.float16, safety_checker=None).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5"); pipe.fuse_lora()
pipe.unet.to(memory_format=torch.channels_last)
pipe.controlnet.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)
pipe.set_progress_bar_config(disable=True)
if COMPILE:
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)
    pipe.controlnet = torch.compile(pipe.controlnet, mode="reduce-overhead", fullgraph=False)
    pipe.vae.decode = torch.compile(pipe.vae.decode, mode="reduce-overhead", fullgraph=False)
gpu_lock = threading.Lock()
print("warming up (compile may take ~60-90s)...", flush=True)
_blank = Image.new("RGB", (RES, RES), (128, 128, 128))
for _ in range(6):
    pipe("warmup", image=_blank, height=RES, width=RES, num_inference_steps=STEPS,
         guidance_scale=GUID, controlnet_conditioning_scale=SCALE)
print("ControlNet ready.", flush=True)


def _to_depth(im):
    d = np.array(depth_est(im)["depth"]).astype("float32")
    d = (d - d.min()) / (d.max() - d.min() + 1e-8) * 255.0
    return Image.fromarray(d.astype("uint8")).convert("RGB").resize((RES, RES))


def _decode(data_url):
    raw = base64.b64decode(data_url.split(",", 1)[-1])
    return Image.open(BytesIO(raw)).convert("RGB").resize((RES, RES))


def _encode(img):
    buf = BytesIO(); img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def infer(pil, prompt):
    p = prompt.strip() or DEFAULT_PROMPT
    with gpu_lock:
        ctrl = _to_depth(pil)
        out = pipe(p, image=ctrl, height=RES, width=RES, num_inference_steps=STEPS,
                   guidance_scale=GUID, controlnet_conditioning_scale=SCALE,
                   generator=torch.Generator("cuda").manual_seed(2)).images[0]
    return out


app = FastAPI()


@app.get("/")
async def index():
    with open(os.path.join(HERE, "compare_index.html")) as f:
        return HTMLResponse(f.read())


@app.get("/tiles")
async def tiles():
    return [{"key": "input", "label": "Webcam", "sub": "input"},
            {"key": "cnout", "label": "Output", "sub": "ControlNet · depth-locked"}]


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            image = data.get("image")
            if not image:
                continue
            pil = _decode(image)
            t0 = time.time()
            out = await asyncio.to_thread(infer, pil, data.get("prompt", ""))
            ms = round((time.time() - t0) * 1000, 1)
            await websocket.send_json({"type": "frames", "tiles": {"cnout": {"image": _encode(out), "ms": ms}}})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws] error: {e}", flush=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"=== ControlNet demo ready. Open http://localhost:{port} ===", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
