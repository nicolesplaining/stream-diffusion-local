import time, torch, numpy as np
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, LCMScheduler
from transformers import pipeline as hf_pipeline

DEV, DT = "cuda", torch.float16
img = Image.open("images/inputs/input.png").convert("RGB").resize((512, 512))
depth_est = hf_pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=0)
d = np.array(depth_est(img)["depth"]).astype("float32"); d = (d-d.min())/(d.max()-d.min()+1e-8)*255
control = Image.fromarray(d.astype("uint8")).convert("RGB")

cn = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=DT)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "Lykon/absolute-reality-1.81", controlnet=cn, torch_dtype=DT, safety_checker=None).to(DEV)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5"); pipe.fuse_lora()
pipe.set_progress_bar_config(disable=True)

PROMPT = "a fluffy cat, feline face, whiskers, fur, cute, detailed photo"
for steps in [2, 3, 4]:
    g = torch.Generator(DEV).manual_seed(0)
    kw = dict(image=control, num_inference_steps=steps, guidance_scale=1.5,
              controlnet_conditioning_scale=1.0, generator=g)
    out = pipe(PROMPT, **kw).images[0]; out.save(f"images/outputs/cn_s{steps}.png")
    for _ in range(3): pipe(PROMPT, **kw)
    t0 = time.time(); N = 12
    for _ in range(N): pipe(PROMPT, **kw)
    print(f"steps={steps}  gen={N/(time.time()-t0):.2f} fps  -> cn_s{steps}.png", flush=True)
