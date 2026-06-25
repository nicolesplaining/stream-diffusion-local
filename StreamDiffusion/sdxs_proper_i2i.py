"""Does SDXS img2img work via PROPER multi-step SDEdit (its own scheduler)?
Compares the single-jump approach vs diffusers img2img at a few steps/strengths."""
import os, sys, time
import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image

MODEL = "IDKiro/sdxs-512-dreamshaper"
W = H = 512
src = Image.open("images/inputs/input.png").convert("RGB").resize((W, H))
prompt = "a photorealistic portrait of a person, natural light, sharp focus, detailed skin"

pipe = AutoPipelineForImage2Image.from_pretrained(MODEL, torch_dtype=torch.float16, safety_checker=None).to("cuda")
print("scheduler:", type(pipe.scheduler).__name__, flush=True)
g = torch.Generator("cuda").manual_seed(0)

configs = [
    (2, 0.5), (2, 0.6), (2, 0.7),
    (3, 0.6), (4, 0.6),
]
for steps, strength in configs:
    out = pipe(prompt=prompt, image=src, num_inference_steps=steps, strength=strength,
               guidance_scale=1.0, generator=g).images[0]
    # contrast proxy
    import numpy as np
    std = float(np.array(out).std())
    fn = f"images/outputs/sdxs_proper_s{steps}_str{strength}.png"
    out.save(fn)
    print(f"steps={steps} strength={strength}  contrast(std)={std:5.1f}  -> {fn}", flush=True)

# speed for the best-looking quick config (2 steps)
steps, strength = 2, 0.6
for _ in range(5):
    pipe(prompt=prompt, image=src, num_inference_steps=steps, strength=strength, guidance_scale=1.0, generator=g)
t0 = time.time(); N = 30
for _ in range(N):
    pipe(prompt=prompt, image=src, num_inference_steps=steps, strength=strength, guidance_scale=1.0, generator=g)
print(f"SDXS proper img2img steps={steps} strength={strength}: {N/(time.time()-t0):.2f} FPS @512", flush=True)
