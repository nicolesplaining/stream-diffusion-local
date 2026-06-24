import os, sys, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.wrapper import StreamDiffusionWrapper
from PIL import Image

ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engines")
print("=== building TensorRT engines (img2img, Realistic Vision, t_index=[32,45]) ===", flush=True)
t0 = time.time()
stream = StreamDiffusionWrapper(
    model_id_or_path="SG161222/Realistic_Vision_V5.1_noVAE",
    t_index_list=[32, 45],
    frame_buffer_size=1,
    width=512, height=512,
    warmup=10,
    acceleration="tensorrt",
    mode="img2img",
    use_denoising_batch=True,
    cfg_type="self",
    output_type="pil",
    engine_dir=ENGINE_DIR,
    seed=2,
)
print(f"=== engines ready / pipeline built in {time.time()-t0:.1f}s ===", flush=True)
stream.prepare(prompt="a photorealistic portrait, natural light",
               negative_prompt="anime, cartoon, low quality",
               num_inference_steps=50, guidance_scale=1.2, delta=0.5)
img = Image.new("RGB", (512, 512), (120, 130, 140))
for _ in range(stream.batch_size - 1):
    stream(image=img)
# benchmark
N = 30
t0 = time.time()
for _ in range(N):
    out = stream(image=img)
dt = time.time() - t0
out.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "outputs", "trt_test.png"))
print(f"=== TensorRT inference: {N} frames in {dt:.2f}s -> {N/dt:.2f} FPS ===", flush=True)
