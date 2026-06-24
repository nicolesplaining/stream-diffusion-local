import os, sys, time, fire
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.wrapper import StreamDiffusionWrapper
from PIL import Image

ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engines")

def run(t_index="40", accel="tensorrt", width=512, height=512, n=40, label=""):
    t_list = [int(x) for x in str(t_index).split(",")]
    stream = StreamDiffusionWrapper(
        model_id_or_path="SG161222/Realistic_Vision_V5.1_noVAE",
        t_index_list=t_list, frame_buffer_size=1, width=width, height=height,
        warmup=10, acceleration=accel, mode="img2img", use_denoising_batch=True,
        cfg_type="self" if len(t_list) > 1 else "none", output_type="pil",
        engine_dir=ENGINE_DIR, seed=2,
    )
    stream.prepare(prompt="a photorealistic portrait, sharp focus, natural light",
                   negative_prompt="anime, cartoon, low quality",
                   num_inference_steps=50, guidance_scale=1.0 if len(t_list)==1 else 1.2, delta=0.5)
    src = Image.open("images/inputs/input.png").convert("RGB").resize((width,height))
    for _ in range(stream.batch_size - 1): stream(image=src)
    # warm
    for _ in range(10): out = stream(image=src)
    t0 = time.time()
    for _ in range(n): out = stream(image=src)
    dt = time.time() - t0
    out.save(f"images/outputs/sd_bench_{label or t_index}_{accel}.png")
    print(f"RESULT  steps={len(t_list)} t_index={t_list} accel={accel} {width}x{height} -> {n/dt:.2f} FPS", flush=True)

if __name__ == "__main__":
    fire.Fire(run)
