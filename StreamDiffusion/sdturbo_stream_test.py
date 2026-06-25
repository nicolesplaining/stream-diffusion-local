import os, sys, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.wrapper import StreamDiffusionWrapper
from PIL import Image

ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engines")
accel = sys.argv[1] if len(sys.argv) > 1 else "none"
print(f"=== StreamDiffusion + sd-turbo, accel={accel} ===", flush=True)
t0 = time.time()
stream = StreamDiffusionWrapper(
    model_id_or_path="stabilityai/sd-turbo",
    t_index_list=[16, 32], frame_buffer_size=1, width=512, height=512,
    warmup=10, acceleration=accel, mode="img2img", use_denoising_batch=True,
    use_lcm_lora=False, cfg_type="none", output_type="pil",
    engine_dir=ENGINE_DIR, seed=2)
print(f"built in {time.time()-t0:.1f}s", flush=True)
stream.prepare(prompt="a photorealistic portrait of a person, natural light, sharp focus",
               num_inference_steps=50, guidance_scale=1.0, delta=0.5)
src = Image.open("images/inputs/input.png").convert("RGB").resize((512, 512))
for _ in range(stream.batch_size - 1): stream(image=src)
for _ in range(10): out = stream(image=src)
t0 = time.time(); N = 40
for _ in range(N): out = stream(image=src)
out.save(f"images/outputs/sdturbo_stream_{accel}.png")
print(f"RESULT accel={accel} -> {N/(time.time()-t0):.2f} FPS", flush=True)
