import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.wrapper import StreamDiffusionWrapper
from PIL import Image

ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engines")
PROMPT = ("a futuristic cyborg, half-machine face, chrome metal plating, glowing blue "
          "cybernetic eyes, exposed circuitry and wires, steel panels, LED implants, sci-fi, intricate detail")
NEG = "low quality, blurry, deformed, text, watermark"
TIDX = [22, 40]  # 2-step (batch-2 engines already built for all)
src = Image.open("images/inputs/input.png").convert("RGB").resize((512, 512))

MODELS = [
    ("realistic",  "SG161222/Realistic_Vision_V5.1_noVAE", True),
    ("absolute",   "Lykon/absolute-reality-1.81",          True),
    ("dreamshaper","Lykon/dreamshaper-8",                  True),
    ("kohaku",     "KBlueLeaf/kohaku-v2.1",                True),
    ("sdturbo",    "stabilityai/sd-turbo",                 False),
]
for key, model, lcm in MODELS:
    print(f"[{key}] building...", flush=True)
    w = StreamDiffusionWrapper(
        model_id_or_path=model, lora_dict=None, t_index_list=TIDX,
        frame_buffer_size=1, width=512, height=512, warmup=10,
        acceleration="tensorrt", mode="img2img", use_denoising_batch=True,
        use_lcm_lora=lcm, cfg_type="none" if not lcm else "self",
        output_type="pil", engine_dir=ENGINE_DIR, seed=2)
    w.prepare(prompt=PROMPT, negative_prompt=NEG, num_inference_steps=50,
              guidance_scale=1.4, delta=0.5)
    for _ in range(w.batch_size - 1): w(image=src)
    for _ in range(6): out = w(image=src)
    out.save(f"images/outputs/cybcmp_{key}.png")
    print(f"[{key}] saved", flush=True)
    del w
    import torch, gc; gc.collect(); torch.cuda.empty_cache()
