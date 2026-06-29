import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.wrapper import StreamDiffusionWrapper
from PIL import Image
import torch, gc

ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engines")
PROMPT = "a fluffy cat, feline face, cat ears, whiskers, fur, cute, detailed"
NEG = "human, person, low quality, blurry, deformed"
src = Image.open("images/inputs/input.png").convert("RGB").resize((512, 512))

# (label, model, use_lcm, t_index, guidance)
CONFIGS = [
    ("ar_20_40_g14", "Lykon/absolute-reality-1.81", True,  [20, 40], 1.4),
    ("ar_16_38_g15", "Lykon/absolute-reality-1.81", True,  [16, 38], 1.5),
    ("ar_12_36_g15", "Lykon/absolute-reality-1.81", True,  [12, 36], 1.5),
    ("turbo_24_41",  "stabilityai/sd-turbo",        False, [24, 41], 1.0),
    ("turbo_20_40",  "stabilityai/sd-turbo",        False, [20, 40], 1.0),
]
for label, model, lcm, tidx, guid in CONFIGS:
    print(f"[{label}] building...", flush=True)
    w = StreamDiffusionWrapper(
        model_id_or_path=model, lora_dict=None, t_index_list=tidx,
        frame_buffer_size=1, width=512, height=512, warmup=8,
        acceleration="tensorrt", mode="img2img", use_denoising_batch=True,
        use_lcm_lora=lcm, cfg_type="none" if not lcm else "self",
        output_type="pil", engine_dir=ENGINE_DIR, seed=2)
    w.prepare(prompt=PROMPT, negative_prompt=NEG, num_inference_steps=50,
              guidance_scale=guid, delta=0.5)
    for _ in range(w.batch_size - 1): w(image=src)
    for _ in range(6): out = w(image=src)
    out.save(f"images/outputs/cat_{label}.png")
    print(f"[{label}] saved", flush=True)
    del w; gc.collect(); torch.cuda.empty_cache()
