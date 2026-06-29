import time, torch, numpy as np
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, LCMScheduler
from transformers import pipeline as hf_pipeline

DEV, DT = "cuda", torch.float16
img = Image.open("images/inputs/input.png").convert("RGB").resize((512, 512))

print("loading depth estimator (DPT/MiDaS)...", flush=True)
depth_est = hf_pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=0)
def to_depth(im):
    d = np.array(depth_est(im)["depth"]).astype("float32")
    d = (d - d.min()) / (d.max() - d.min() + 1e-8) * 255.0
    return Image.fromarray(d.astype("uint8")).convert("RGB")
control = to_depth(img)
control.save("images/outputs/cn_depth.png")
print("depth ok -> cn_depth.png", flush=True)

print("loading ControlNet depth + Absolute Reality + LCM...", flush=True)
cn = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=DT)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "Lykon/absolute-reality-1.81", controlnet=cn, torch_dtype=DT, safety_checker=None).to(DEV)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5"); pipe.fuse_lora()
print("pipeline ready", flush=True)

PROMPT = "a fluffy cat, feline face, whiskers, fur, cute, detailed photo"
g = torch.Generator(DEV).manual_seed(0)
kw = dict(image=control, num_inference_steps=6, guidance_scale=1.5,
          controlnet_conditioning_scale=1.0, generator=g)
out = pipe(PROMPT, **kw).images[0]; out.save("images/outputs/cn_cat.png")
print("generated -> cn_cat.png", flush=True)

for _ in range(3): pipe(PROMPT, **kw)
t0 = time.time(); N = 10
for _ in range(N): pipe(PROMPT, **kw)
gen_fps = N / (time.time() - t0)
t0 = time.time()
for _ in range(N): to_depth(img)
depth_fps = N / (time.time() - t0)
print(f"RESULT gen={gen_fps:.2f} fps, depth={depth_fps:.2f} fps, combined~{1/(1/gen_fps+1/depth_fps):.2f} fps", flush=True)
