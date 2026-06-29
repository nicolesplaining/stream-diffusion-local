import time, torch, numpy as np
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, LCMScheduler
from transformers import pipeline as hf_pipeline
torch.set_float32_matmul_precision("high")
DEV, DT = "cuda", torch.float16
RES = 512
src = Image.open("images/inputs/input.png").convert("RGB")
depth_est = hf_pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=0)
def depth(res_in):
    d=np.array(depth_est(src.resize((res_in,res_in)))["depth"]).astype("float32"); d=(d-d.min())/(d.max()-d.min()+1e-8)*255
    return Image.fromarray(d.astype("uint8")).convert("RGB").resize((RES,RES))
cn = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=DT)
pipe = StableDiffusionControlNetPipeline.from_pretrained("Lykon/absolute-reality-1.81", controlnet=cn, torch_dtype=DT, safety_checker=None).to(DEV)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5"); pipe.fuse_lora()
for m in (pipe.unet, pipe.controlnet, pipe.vae): m.to(memory_format=torch.channels_last)
pipe.set_progress_bar_config(disable=True)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
pipe.controlnet = torch.compile(pipe.controlnet, mode="reduce-overhead")
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="reduce-overhead")
P="a fluffy cat, feline face, whiskers, fur, cute, detailed"
# depth cost at two input sizes
c512=depth(512)
for r in (512,256):
    for _ in range(3): depth(r)
    t=time.time(); [depth(r) for _ in range(15)]; print(f"depth(in={r}): {1000*(time.time()-t)/15:.0f} ms", flush=True)
def run(tag, **extra):
    kw=dict(image=c512,height=RES,width=RES,guidance_scale=1.5,controlnet_conditioning_scale=0.3, **extra)
    for _ in range(5): pipe(P, generator=torch.Generator(DEV).manual_seed(2), **kw)
    pipe(P, generator=torch.Generator(DEV).manual_seed(2), **kw).images[0].save(f"images/outputs/cnf_{tag}.png")
    torch.cuda.synchronize(); t=time.time(); N=15
    for _ in range(N): pipe(P, generator=torch.Generator(DEV).manual_seed(2), **kw)
    torch.cuda.synchronize(); print(f"{tag}: gen {1000*(time.time()-t)/N:.0f} ms ({N/(time.time()-t):.1f} fps gen-only)", flush=True)
run("2step", num_inference_steps=2)
run("2step_cge05", num_inference_steps=2, control_guidance_end=0.5)
run("1step", num_inference_steps=1)
run("1step_cge0", num_inference_steps=1, control_guidance_end=0.6)
