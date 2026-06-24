import os, sys, time, fire
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.wrapper import StreamDiffusionWrapper
from PIL import Image

def run(model="IDKiro/sdxs-512-dreamshaper", t_index="40", accel="none",
        use_lcm_lora=False, use_tiny_vae=True, width=512, height=512, n=40, label="sdxs"):
    t_list = [int(x) for x in str(t_index).split(",")]
    stream = StreamDiffusionWrapper(
        model_id_or_path=model,
        t_index_list=t_list, frame_buffer_size=1, width=width, height=height,
        warmup=10, acceleration=accel, mode="img2img", use_denoising_batch=True,
        use_lcm_lora=use_lcm_lora, use_tiny_vae=use_tiny_vae, cfg_type="none", output_type="pil",
        engine_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "engines"), seed=2,
    )
    stream.prepare(prompt="a photorealistic portrait, sharp focus, natural light",
                   negative_prompt="low quality, blurry",
                   num_inference_steps=50, guidance_scale=1.0, delta=0.5)
    src = Image.open("images/inputs/input.png").convert("RGB").resize((width, height))
    for _ in range(stream.batch_size - 1): stream(image=src)
    for _ in range(10): out = stream(image=src)
    t0 = time.time()
    for _ in range(n): out = stream(image=src)
    dt = time.time() - t0
    out.save(f"images/outputs/{label}_{t_index}_{accel}.png")
    print(f"RESULT model={model} lcm_lora={use_lcm_lora} steps={len(t_list)} t={t_list} accel={accel} {width}x{height} -> {n/dt:.2f} FPS", flush=True)

if __name__ == "__main__":
    fire.Fire(run)
