import time, torch, numpy as np
from PIL import Image
from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                       LCMScheduler, AutoencoderTiny)
from transformers import pipeline as hf_pipeline

DEV, DT = "cuda", torch.float16
img = Image.open("images/inputs/input.png").convert("RGB")
depth_est = hf_pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=0)

cn = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=DT)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "Lykon/absolute-reality-1.81", controlnet=cn, torch_dtype=DT, safety_checker=None).to(DEV)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=DT).to(DEV)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5"); pipe.fuse_lora()
pipe.set_progress_bar_config(disable=True)
PROMPT = "a fluffy cat, feline face, whiskers, fur, cute"

def depth_at(res):
    d = np.array(depth_est(img.resize((res, res)))["depth"]).astype("float32")
    d = (d-d.min())/(d.max()-d.min()+1e-8)*255
    return Image.fromarray(d.astype("uint8")).convert("RGB")

for res, steps in [(512,4),(384,3),(384,2),(320,3),(256,3)]:
    ctrl = depth_at(res)
    kw = dict(image=ctrl, height=res, width=res, num_inference_steps=steps,
              guidance_scale=1.5, controlnet_conditioning_scale=0.4)
    out = pipe(PROMPT, generator=torch.Generator(DEV).manual_seed(2), **kw).images[0]
    out.save(f"images/outputs/cnsp_{res}_{steps}.png")
    for _ in range(3): pipe(PROMPT, generator=torch.Generator(DEV).manual_seed(2), **kw)
    t0=time.time(); N=12
    for _ in range(N):
        d0=time.time(); c=depth_at(res); 
        pipe(PROMPT, image=c, height=res, width=res, num_inference_steps=steps, guidance_scale=1.5, controlnet_conditioning_scale=0.4, generator=torch.Generator(DEV).manual_seed(2))
    fps=N/(time.time()-t0)
    print(f"res={res} steps={steps} -> {fps:.2f} fps (incl depth) -> cnsp_{res}_{steps}.png", flush=True)
