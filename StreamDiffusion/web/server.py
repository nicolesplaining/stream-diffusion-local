"""
Real-time webcam img2img web app for StreamDiffusion.

Serves a web page where you can:
  - see your webcam transformed live (input | output)
  - type your own prompt
  - pick a STYLE from a dropdown -- each style is a different model

Run (from the StreamDiffusion repo root, venv active):

    python web/server.py
    # then open http://localhost:8000 IN A BROWSER ON THE SPARK
    # (getUserMedia/webcam only works on localhost or HTTPS)

Models are loaded on demand the first time a style is selected and then cached,
so the first switch to a new style takes a while (download + warmup); after that
switching is instant.
"""

import os
import sys
import base64
import asyncio
import threading
from io import BytesIO

import numpy as np
from PIL import Image
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.wrapper import StreamDiffusionWrapper


# ---------------------------------------------------------------------------
# Styles. Each style == one model. Add more by appending entries here.
# `suffix` is appended to the user's prompt; `negative` is the negative prompt.
# ---------------------------------------------------------------------------
STYLES = {
    "photorealistic": {
        "label": "Photorealistic",
        "model": "SG161222/Realistic_Vision_V5.1_noVAE",
        "suffix": "RAW photo, photorealistic, 35mm photograph, natural skin texture, "
                  "soft natural lighting, highly detailed, sharp focus",
        "negative": "anime, illustration, cartoon, drawing, painting, cgi, 3d render, "
                    "low quality, blurry, deformed",
    },
    "hyperreal": {
        "label": "Hyperreal",
        "model": "Lykon/absolute-reality-1.81",
        "suffix": "hyperrealistic photo, ultra detailed, cinematic lighting, sharp focus, 8k",
        "negative": "anime, cartoon, painting, drawing, low quality, blurry, deformed",
    },
    "fantasy": {
        "label": "Dreamy / Fantasy",
        "model": "Lykon/dreamshaper-8",
        "suffix": "fantasy art, dramatic cinematic lighting, highly detailed, vivid colors, "
                  "concept art, artstation",
        "negative": "low quality, bad quality, blurry, deformed, watermark, text",
    },
    "anime": {
        "label": "Anime",
        "model": "KBlueLeaf/kohaku-v2.1",
        "suffix": "anime, illustration, vibrant colors, clean lines, masterpiece, best quality",
        "negative": "photo, realistic, 3d render, low quality, bad quality, blurry, deformed",
    },
}

WIDTH = int(os.environ.get("SD_WIDTH", 512))
HEIGHT = int(os.environ.get("SD_HEIGHT", 512))
T_INDEX = [int(x) for x in os.environ.get("SD_T_INDEX", "22,32,45").split(",")]
ACCELERATION = os.environ.get("SD_ACCELERATION", "none")


class StyleManager:
    """Loads/caches one StreamDiffusion pipeline per style and runs inference."""

    def __init__(self):
        self.cache = {}            # style_key -> wrapper
        self.last_prompt = {}      # style_key -> last full prompt sent
        self.lock = threading.Lock()

    def is_loaded(self, style_key):
        return style_key in self.cache

    def ensure(self, style_key):
        """Build + warm up the pipeline for a style if not already cached."""
        with self.lock:
            if style_key in self.cache:
                return
            style = STYLES[style_key]
            print(f"[load] building pipeline for '{style_key}' ({style['model']}) ...", flush=True)
            wrapper = StreamDiffusionWrapper(
                model_id_or_path=style["model"],
                lora_dict=None,
                t_index_list=T_INDEX,
                frame_buffer_size=1,
                width=WIDTH,
                height=HEIGHT,
                warmup=10,
                acceleration=ACCELERATION,
                mode="img2img",
                use_denoising_batch=True,
                cfg_type="self",
                output_type="pil",
                seed=2,
            )
            wrapper.prepare(
                prompt=style["suffix"],
                negative_prompt=style["negative"],
                num_inference_steps=50,
                guidance_scale=1.2,
                delta=0.5,
            )
            # prime the denoising batch buffer with a neutral gray frame
            gray = Image.new("RGB", (WIDTH, HEIGHT), (128, 128, 128))
            for _ in range(wrapper.batch_size - 1):
                wrapper(image=gray)
            self.cache[style_key] = wrapper
            self.last_prompt[style_key] = None
            print(f"[load] '{style_key}' ready.", flush=True)

    def infer(self, style_key, pil_img, user_prompt):
        style = STYLES[style_key]
        full_prompt = (user_prompt.strip() + ", " + style["suffix"]).strip(", ") \
            if user_prompt.strip() else style["suffix"]
        with self.lock:
            wrapper = self.cache[style_key]
            send_prompt = full_prompt if self.last_prompt.get(style_key) != full_prompt else None
            self.last_prompt[style_key] = full_prompt
            return wrapper(image=pil_img, prompt=send_prompt)


manager = StyleManager()
app = FastAPI()

HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")


@app.get("/")
async def index():
    with open(HTML_PATH, "r") as f:
        return HTMLResponse(f.read())


@app.get("/styles")
async def styles():
    return {k: v["label"] for k, v in STYLES.items()}


def _decode(data_url: str) -> Image.Image:
    b64 = data_url.split(",", 1)[-1]
    raw = base64.b64decode(b64)
    return Image.open(BytesIO(raw)).convert("RGB").resize((WIDTH, HEIGHT))


def _encode(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    current_style = None
    try:
        while True:
            data = await websocket.receive_json()
            style_key = data.get("style", "photorealistic")
            prompt = data.get("prompt", "")
            image = data.get("image")
            if style_key not in STYLES or not image:
                continue

            if not manager.is_loaded(style_key):
                await websocket.send_json(
                    {"type": "status", "message": f"Loading “{STYLES[style_key]['label']}” "
                                                  f"model (first time, please wait…)"})
                await asyncio.to_thread(manager.ensure, style_key)
                await websocket.send_json(
                    {"type": "status", "message": f"{STYLES[style_key]['label']} ready"})
            current_style = style_key

            pil = _decode(image)
            out = await asyncio.to_thread(manager.infer, style_key, pil, prompt)
            await websocket.send_json({"type": "frame", "image": _encode(out)})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws] error: {e}", flush=True)


if __name__ == "__main__":
    print("Open http://localhost:8000 in a browser ON THIS MACHINE (webcam needs localhost/HTTPS).")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
