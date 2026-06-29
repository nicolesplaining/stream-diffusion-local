import torch, numpy as np
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

PROMPT = "a fluffy cat, feline face, whiskers, fur, cute"
for steps in [3, 4]:
    for scale in [0.4, 0.55, 0.7, 0.85]:
        g = torch.Generator(DEV).manual_seed(2)
        out = pipe(PROMPT, image=control, num_inference_steps=steps, guidance_scale=1.5,
                   controlnet_conditioning_scale=scale, generator=g).images[0]
        out.save(f"images/outputs/cnsc_s{steps}_{scale}.png")
        print(f"steps={steps} scale={scale} saved", flush=True)
