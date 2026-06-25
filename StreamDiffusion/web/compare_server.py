"""
Live comparison grid: one webcam stream -> every pipeline/model strategy we've
built, all running side by side so the outputs can be compared in real time.

Tiles (all fed the SAME webcam frame):
  - Realistic Vision  (SD1.5 + LCM, TensorRT, 2-step)   photorealistic
  - Absolute Reality  (SD1.5 + LCM, TensorRT, 2-step)   hyperreal
  - Dreamshaper 8     (SD1.5 + LCM, TensorRT, 2-step)   fantasy
  - Kohaku v2.1       (SD1.5 + LCM, TensorRT, 2-step)   anime
  - SD-Turbo          (SD2.1-Turbo, TensorRT, 2-step)   fast + clean few-step i2i
  - SDXS              (distilled 1-step, manual epsilon img2img)  experimental / blooms
  - Live2Diff         (temporally-coherent v2v, TensorRT 384) -- separate venv worker

Run (StreamDiffusion repo root, .venv active):
    python web/compare_server.py
    # open http://localhost:8000 on the Spark (webcam needs localhost/HTTPS)

Engines are built once and cached under engines/; models are pre-downloaded.
Disable Live2Diff with NO_LIVE2DIFF=1.
"""
import os, sys, time, json, base64, asyncio, threading, subprocess, atexit, urllib.request
from io import BytesIO

import numpy as np
from PIL import Image
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

HERE = os.path.dirname(os.path.abspath(__file__))
SD_ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(SD_ROOT)
sys.path.append(SD_ROOT)
from utils.wrapper import StreamDiffusionWrapper
from sdxs_i2i import SDXSImg2Img

WIDTH = int(os.environ.get("SD_WIDTH", 512))
HEIGHT = int(os.environ.get("SD_HEIGHT", 512))
ENGINE_DIR = os.path.join(SD_ROOT, "engines")
T_INDEX = [int(x) for x in os.environ.get("SD_T_INDEX", "40").split(",")]
SDXS_T = int(os.environ.get("SDXS_T", 700))

L2D_DIR = os.path.join(REPO_ROOT, "Live2Diff")
L2D_PY = os.path.join(REPO_ROOT, ".venv-live2diff", "bin", "python")
L2D_PORT = int(os.environ.get("L2D_PORT", 8002))
ENABLE_L2D = os.environ.get("NO_LIVE2DIFF", "") == "" and os.path.exists(L2D_PY)

# Each tile == one StreamDiffusionWrapper. Optional per-spec overrides:
#   lcm (use_lcm_lora, default True), cfg (cfg_type, default "self"),
#   guidance (default 1.2), t_index (default T_INDEX).
STYLES = [
    dict(key="photorealistic", label="Realistic Vision", sub="SD1.5+LCM · TRT 1-step",
         model="SG161222/Realistic_Vision_V5.1_noVAE",
         suffix="RAW photo, photorealistic, 35mm photograph, natural skin texture, soft natural lighting, highly detailed, sharp focus",
         negative="anime, illustration, cartoon, drawing, painting, cgi, 3d render, low quality, blurry, deformed"),
    dict(key="hyperreal", label="Absolute Reality", sub="SD1.5+LCM · TRT 1-step",
         model="Lykon/absolute-reality-1.81",
         suffix="hyperrealistic photo, ultra detailed, cinematic lighting, sharp focus, 8k",
         negative="anime, cartoon, painting, drawing, low quality, blurry, deformed"),
    dict(key="fantasy", label="Dreamshaper 8", sub="SD1.5+LCM · TRT 1-step",
         model="Lykon/dreamshaper-8",
         suffix="fantasy art, dramatic cinematic lighting, highly detailed, vivid colors, concept art, artstation",
         negative="low quality, bad quality, blurry, deformed, watermark, text"),
    dict(key="anime", label="Kohaku v2.1", sub="SD1.5+LCM · TRT 1-step",
         model="KBlueLeaf/kohaku-v2.1",
         suffix="anime, illustration, vibrant colors, clean lines, masterpiece, best quality",
         negative="photo, realistic, 3d render, low quality, bad quality, blurry, deformed"),
    dict(key="turbo", label="SD-Turbo", sub="SD2.1-Turbo · TRT 1-step",
         model="stabilityai/sd-turbo", lcm=False, cfg="none", guidance=1.0,
         suffix="a photorealistic portrait, natural light, sharp focus, highly detailed",
         negative="low quality, blurry, deformed"),
]


def _decode(data_url):
    raw = base64.b64decode(data_url.split(",", 1)[-1])
    return Image.open(BytesIO(raw)).convert("RGB").resize((WIDTH, HEIGHT))


def _encode(img):
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class SDTile:
    def __init__(self, spec):
        self.spec = spec
        self.last_prompt = None
        t_index = spec.get("t_index", T_INDEX)
        nstep = len(t_index)
        cfg = spec.get("cfg", "self" if nstep > 1 else "none")
        guid = spec.get("guidance", 1.2 if nstep > 1 else 1.0)
        guid = float(os.environ.get("SD_GUIDANCE", guid))
        print(f"[build] {spec['label']} ({spec['model']}) - TensorRT ...", flush=True)
        self.w = StreamDiffusionWrapper(
            model_id_or_path=spec["model"], lora_dict=None, t_index_list=t_index,
            frame_buffer_size=1, width=WIDTH, height=HEIGHT, warmup=10,
            acceleration="tensorrt", mode="img2img", use_denoising_batch=True,
            use_lcm_lora=spec.get("lcm", True), cfg_type=cfg,
            output_type="pil", engine_dir=ENGINE_DIR, seed=2)
        self.w.prepare(prompt=spec["suffix"], negative_prompt=spec["negative"],
                       num_inference_steps=50, guidance_scale=guid, delta=0.5)
        gray = Image.new("RGB", (WIDTH, HEIGHT), (128, 128, 128))
        for _ in range(self.w.batch_size - 1):
            self.w(image=gray)
        print(f"[build] {spec['label']} ready.", flush=True)

    def infer(self, pil, user_prompt):
        s = self.spec
        full = (user_prompt.strip() + ", " + s["suffix"]).strip(", ") if user_prompt.strip() else s["suffix"]
        sp = full if self.last_prompt != full else None
        self.last_prompt = full
        return self.w(image=pil, prompt=sp)


class SDXSTile:
    def __init__(self, t_int=700):
        self.t = t_int
        self._default = "a photorealistic portrait of a person, sharp focus, natural light"
        print("[build] SDXS (distilled 1-step, manual i2i) ...", flush=True)
        self.m = SDXSImg2Img(width=WIDTH, height=HEIGHT)
        self.m.set_prompt(self._default)
        self.last_prompt = self._default
        print("[build] SDXS ready.", flush=True)

    def infer(self, pil, user_prompt):
        up = user_prompt.strip()
        want = up if up else self._default
        if want != self.last_prompt:
            self.m.set_prompt(want)
            self.last_prompt = want
        return self.m(pil, self.t)


_only = [k.strip() for k in os.environ.get("ONLY", "").split(",") if k.strip()]
if _only:
    STYLES = [s for s in STYLES if s["key"] in _only]
INCLUDE_SDXS = os.environ.get("NO_SDXS", "") == "" and (not _only or "sdxs" in _only)

print("=== Building StreamDiffusion-family pipelines (TensorRT) ===", flush=True)
SD_TILES = [SDTile(spec) for spec in STYLES]
SDXS_TILE = SDXSTile(SDXS_T) if INCLUDE_SDXS else None
print("=== StreamDiffusion-family ready ===", flush=True)

TILE_META = [{"key": "input", "label": "Webcam", "sub": "input"}]
for s in STYLES:
    _ns = len(s.get("t_index", T_INDEX))
    _sub = s["sub"].split(" · TRT")[0] + f" · TRT {_ns}-step"
    TILE_META.append({"key": s["key"], "label": s["label"], "sub": _sub})
if INCLUDE_SDXS:
    TILE_META.append({"key": "sdxs", "label": "SDXS", "sub": "1-step manual i2i · blooms"})
if ENABLE_L2D:
    TILE_META.append({"key": "live2diff", "label": "Live2Diff", "sub": "coherent v2v · TRT 384"})


def run_all_sd(pil, prompt):
    out = {}
    for tile in SD_TILES:
        t0 = time.time()
        img = tile.infer(pil, prompt)
        out[tile.spec["key"]] = {"image": _encode(img), "ms": round((time.time() - t0) * 1000, 1)}
    if SDXS_TILE is not None:
        t0 = time.time()
        img = SDXS_TILE.infer(pil, prompt)
        out["sdxs"] = {"image": _encode(img), "ms": round((time.time() - t0) * 1000, 1)}
    return out


# ---- Live2Diff cross-venv worker ------------------------------------------
latest_input = {"image": None, "prompt": ""}
l2d_state = {"image": None, "ms": 0.0, "status": "starting"}
_worker_proc = None


def _l2d_post(path, payload, timeout=60):
    req = urllib.request.Request(f"http://127.0.0.1:{L2D_PORT}{path}",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _l2d_health():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{L2D_PORT}/health", timeout=2) as r:
            return json.loads(r.read().decode()).get("ready", False)
    except Exception:
        return False


def start_worker():
    global _worker_proc
    worker = os.path.join(L2D_DIR, "web", "compare_worker.py")
    print(f"[l2d] spawning worker ({L2D_PORT}) ...", flush=True)
    _worker_proc = subprocess.Popen([L2D_PY, worker], cwd=L2D_DIR,
        env={**os.environ, "L2D_PORT": str(L2D_PORT)})
    atexit.register(lambda: _worker_proc and _worker_proc.terminate())


async def l2d_loop():
    l2d_state["status"] = "loading model..."
    for _ in range(240):  # up to ~8 min for first-time load
        if _l2d_health():
            break
        await asyncio.sleep(2)
    else:
        l2d_state["status"] = "worker failed to start"
        return
    l2d_state["status"] = "warming up..."
    while True:
        du = latest_input["image"]
        if du is None:
            await asyncio.sleep(0.05)
            continue
        try:
            t0 = time.time()
            resp = await asyncio.to_thread(_l2d_post, "/infer",
                {"image": du, "prompt": latest_input["prompt"]})
            dt = (time.time() - t0) * 1000
            if resp.get("image"):
                l2d_state.update(image=resp["image"], ms=round(dt, 1), status="streaming")
            elif resp.get("status") == "ready":
                l2d_state["status"] = "streaming"
            elif "warming" in resp:
                l2d_state["status"] = f"warming {resp['warming']}/{resp['need']}"
        except Exception:
            l2d_state["status"] = "worker error"
            await asyncio.sleep(0.2)


app = FastAPI()
HTML_PATH = os.path.join(HERE, "compare_index.html")


@app.get("/")
async def index():
    with open(HTML_PATH) as f:
        return HTMLResponse(f.read())


@app.get("/tiles")
async def tiles():
    return TILE_META


@app.on_event("startup")
async def _startup():
    if ENABLE_L2D:
        asyncio.create_task(l2d_loop())


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            image = data.get("image")
            prompt = data.get("prompt", "")
            if not image:
                continue
            if ENABLE_L2D:
                latest_input["image"] = image
                latest_input["prompt"] = prompt
            pil = _decode(image)
            tiles_out = await asyncio.to_thread(run_all_sd, pil, prompt)
            if ENABLE_L2D:
                tiles_out["live2diff"] = {"image": l2d_state["image"],
                    "ms": l2d_state["ms"], "status": l2d_state["status"]}
            await websocket.send_json({"type": "frames", "tiles": tiles_out})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws] error: {e}", flush=True)


if __name__ == "__main__":
    if ENABLE_L2D:
        start_worker()
    port = int(os.environ.get("PORT", 8000))
    print(f"=== Compare grid ready. Open http://localhost:{port} on the Spark ===", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
