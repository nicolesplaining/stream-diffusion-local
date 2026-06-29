import time, torch, numpy as np
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, LCMScheduler
from transformers import pipeline as hf_pipeline
torch.set_float32_matmul_precision("high")
DEV, DT = "cuda", torch.float16
src = Image.open("images/inputs/input.png").convert("RGB")
depth_est = hf_pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=0)
cn = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=DT)
pipe = StableDiffusionControlNetPipeline.from_pretrained("Lykon/absolute-reality-1.81", controlnet=cn, torch_dtype=DT, safety_checker=None).to(DEV)
# FULL VAE (sharp), channels_last, compiled
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5"); pipe.fuse_lora()
pipe.unet.to(memory_format=torch.channels_last); pipe.controlnet.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)
pipe.set_progress_bar_config(disable=True)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)
pipe.controlnet = torch.compile(pipe.controlnet, mode="reduce-overhead", fullgraph=False)
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="reduce-overhead", fullgraph=False)
def depth(res):
    d=np.array(depth_est(src.resize((res,res)))["depth"]).astype("float32"); d=(d-d.min())/(d.max()-d.min()+1e-8)*255
    return Image.fromarray(d.astype("uint8")).convert("RGB")
PROMPT="a fluffy cat, feline face, whiskers, fur, cute, detailed"
for res, steps in [(512,2),(512,3),(384,2)]:
    c=depth(res)
    kw=dict(image=c,height=res,width=res,num_inference_steps=steps,guidance_scale=1.5,controlnet_conditioning_scale=0.4)
    for _ in range(5): pipe(PROMPT, generator=torch.Generator(DEV).manual_seed(2), **kw)  # compile warmup
    out=pipe(PROMPT, generator=torch.Generator(DEV).manual_seed(2), **kw).images[0]; out.save(f"images/outputs/cnsharp_{res}_{steps}.png")
    torch.cuda.synchronize(); t=time.time(); N=12
    for _ in range(N):
        c2=depth(res); pipe(PROMPT, image=c2, height=res, width=res, num_inference_steps=steps, guidance_scale=1.5, controlnet_conditioning_scale=0.4, generator=torch.Generator(DEV).manual_seed(2))
    torch.cuda.synchronize(); print(f"res={res} steps={steps} FULLVAE+compiled -> {N/(time.time()-t):.2f} fps (incl depth)", flush=True)
