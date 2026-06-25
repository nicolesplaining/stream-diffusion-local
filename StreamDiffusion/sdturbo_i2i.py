"""SD-Turbo img2img test: clean + fast + structure-preserving few-step i2i?"""
import os, sys, time
import numpy as np, torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image

MODEL = "stabilityai/sd-turbo"
W = H = 512
src = Image.open("images/inputs/input.png").convert("RGB").resize((W, H))
prompt = "a photorealistic portrait of a person, natural light, sharp focus, detailed skin"

pipe = AutoPipelineForImage2Image.from_pretrained(MODEL, torch_dtype=torch.float16, safety_checker=None).to("cuda")
print("scheduler:", type(pipe.scheduler).__name__, flush=True)
g = torch.Generator("cuda").manual_seed(0)

# keep num_inference_steps*strength >= 1 (>=1 real denoising step)
for steps, strength in [(2, 0.5), (2, 0.6), (2, 0.7), (3, 0.5), (4, 0.5)]:
    out = pipe(prompt=prompt, image=src, num_inference_steps=steps, strength=strength,
               guidance_scale=0.0, generator=g).images[0]
    std = float(np.array(out).std())
    fn = f"images/outputs/sdturbo_s{steps}_str{strength}.png"
    out.save(fn)
    print(f"steps={steps} strength={strength}  contrast(std)={std:5.1f}  -> {fn}", flush=True)

steps, strength = 2, 0.5
for _ in range(5):
    pipe(prompt=prompt, image=src, num_inference_steps=steps, strength=strength, guidance_scale=0.0, generator=g)
t0 = time.time(); N = 30
for _ in range(N):
    pipe(prompt=prompt, image=src, num_inference_steps=steps, strength=strength, guidance_scale=0.0, generator=g)
print(f"SD-Turbo img2img steps={steps} strength={strength}: {N/(time.time()-t0):.2f} FPS @512", flush=True)
