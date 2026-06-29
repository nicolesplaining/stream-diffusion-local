import time, torch, numpy as np
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, LCMScheduler, AutoencoderTiny
from transformers import pipeline as hf_pipeline
torch.set_float32_matmul_precision("high")
DEV, DT = "cuda", torch.float16
img = Image.open("images/inputs/input.png").convert("RGB").resize((384,384))
depth_est = hf_pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=0)
d=np.array(depth_est(img)["depth"]).astype("float32"); d=(d-d.min())/(d.max()-d.min()+1e-8)*255
c=Image.fromarray(d.astype("uint8")).convert("RGB")
cn = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=DT)
pipe = StableDiffusionControlNetPipeline.from_pretrained("Lykon/absolute-reality-1.81", controlnet=cn, torch_dtype=DT, safety_checker=None).to(DEV)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=DT).to(DEV)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5"); pipe.fuse_lora()
pipe.set_progress_bar_config(disable=True)
pipe.unet.to(memory_format=torch.channels_last)
pipe.controlnet.to(memory_format=torch.channels_last)
gk=dict(image=c,height=384,width=384,num_inference_steps=2,guidance_scale=1.5,controlnet_conditioning_scale=0.4)
def bench(tag,n=15):
    for _ in range(4): pipe("a cat", generator=torch.Generator(DEV).manual_seed(2), **gk)
    torch.cuda.synchronize(); t=time.time()
    for _ in range(n): pipe("a cat", generator=torch.Generator(DEV).manual_seed(2), **gk)
    torch.cuda.synchronize(); print(f"{tag}: {1000*(time.time()-t)/n:.0f} ms/frame ({n/(time.time()-t):.2f} fps gen)", flush=True)
bench("eager")
print("compiling (this takes ~30-90s warmup)...", flush=True)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)
pipe.controlnet = torch.compile(pipe.controlnet, mode="reduce-overhead", fullgraph=False)
out = pipe("a fluffy cat, feline face, whiskers, fur", generator=torch.Generator(DEV).manual_seed(2), **gk).images[0]
out.save("images/outputs/cn_compiled_cat.png")
bench("compiled")
