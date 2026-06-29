import time, torch, numpy as np
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, LCMScheduler, AutoencoderTiny
from transformers import pipeline as hf_pipeline
DEV, DT = "cuda", torch.float16
img = Image.open("images/inputs/input.png").convert("RGB").resize((384,384))
depth_est = hf_pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=0)
cn = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=DT)
pipe = StableDiffusionControlNetPipeline.from_pretrained("Lykon/absolute-reality-1.81", controlnet=cn, torch_dtype=DT, safety_checker=None).to(DEV)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=DT).to(DEV)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5"); pipe.fuse_lora()
pipe.set_progress_bar_config(disable=True)
def depth():
    d=np.array(depth_est(img)["depth"]).astype("float32"); d=(d-d.min())/(d.max()-d.min()+1e-8)*255
    return Image.fromarray(d.astype("uint8")).convert("RGB")
c=depth()
gk=dict(image=c,height=384,width=384,num_inference_steps=2,guidance_scale=1.5,controlnet_conditioning_scale=0.4)
for _ in range(3): pipe("a cat", generator=torch.Generator(DEV).manual_seed(2), **gk)
N=15
t=time.time()
for _ in range(N): depth()
print(f"depth only: {1000*(time.time()-t)/N:.0f} ms/frame", flush=True)
t=time.time()
for _ in range(N): pipe("a cat", generator=torch.Generator(DEV).manual_seed(2), **gk)
print(f"gen only (384,2step): {1000*(time.time()-t)/N:.0f} ms/frame", flush=True)
