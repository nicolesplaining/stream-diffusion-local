"""Manual one-step img2img around SDXS's native pipeline (epsilon prediction)."""
import os, sys, time, fire
import torch
from PIL import Image

MODEL = "IDKiro/sdxs-512-dreamshaper"


class SDXSImg2Img:
    def __init__(self, width=512, height=512, device="cuda", dtype=torch.float16):
        from diffusers import StableDiffusionPipeline
        self.w, self.h, self.device, self.dtype = width, height, device, dtype
        pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=dtype).to(device)
        self.unet, self.vae, self.proc = pipe.unet, pipe.vae, pipe.image_processor
        self.sf = self.vae.config.scaling_factor
        self.acp = pipe.scheduler.alphas_cumprod.to(device)
        self._encode_prompt = pipe.encode_prompt
        self.embeds = None

    def set_prompt(self, prompt):
        pe = self._encode_prompt(prompt, self.device, 1, do_classifier_free_guidance=False)[0]
        self.embeds = pe

    @torch.no_grad()
    def __call__(self, pil, t_int=500):
        x = self.proc.preprocess(pil, height=self.h, width=self.w).to(self.device, self.dtype)
        lat = self.vae.encode(x).latents * self.sf
        t = torch.tensor(int(t_int), device=self.device)
        a = self.acp[int(t_int)].to(self.dtype)
        sa, sb = a.sqrt(), (1 - a).sqrt()
        noise = torch.randn(lat.shape, device=self.device, dtype=self.dtype)
        noisy = sa * lat + sb * noise
        eps = self.unet(noisy, t, encoder_hidden_states=self.embeds).sample
        x0 = (noisy - sb * eps) / sa
        img = self.vae.decode(x0 / self.sf).sample
        return self.proc.postprocess(img, output_type="pil")[0]


def main(t="300,500,700,900", n=40):
    m = SDXSImg2Img()
    m.set_prompt("a photorealistic portrait of a person, sharp focus, natural light")
    src = Image.open("images/inputs/input.png").convert("RGB").resize((512, 512))
    tvals = [int(x) for x in t] if isinstance(t, (tuple, list)) else [int(x) for x in str(t).split(",")]
    for ti in tvals:
        m(src, ti)  # warm
        out = m(src, ti); out.save(f"images/outputs/sdxs_i2i_t{ti}.png")
        print(f"t={ti} saved")
    # timing at t=500
    for _ in range(5): m(src, 500)
    t0 = time.time()
    for _ in range(n): m(src, 500)
    print(f"SDXS manual img2img: {n/(time.time()-t0):.2f} FPS @512")


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    fire.Fire(main)
