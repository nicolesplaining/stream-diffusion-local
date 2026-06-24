import os, sys, time
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live2diff.utils.wrapper import StreamAnimateDiffusionDepthWrapper

W = H = 512
img = Image.open("../StreamDiffusion/images/outputs/trt_real.png").convert("RGB").resize((W, H))
arr = torch.from_numpy(np.array(img)).float() / 255.0
clip = arr[None].repeat(16, 1, 1, 1)

print("=== building Live2Diff TensorRT engines (acceleration=tensorrt) ===", flush=True)
t0 = time.time()
stream = StreamAnimateDiffusionDepthWrapper(
    few_step_model_type="lcm",
    config_path="configs/spike.yaml",
    cfg_type="none",
    strength=None,
    num_inference_steps=50,
    t_index_list=[31, 43],
    frame_buffer_size=1,
    width=W, height=H,
    acceleration="tensorrt",
    do_add_noise=True,
    output_type="pt",
    use_denoising_batch=True,
    use_tiny_vae=True,
    engine_dir="engines_trt",
    seed=42,
)
print(f"=== engines ready / pipeline built in {time.time()-t0:.1f}s ===", flush=True)

stream.prepare(warmup_frames=clip[:8].permute(0, 3, 1, 2),
               prompt="a photorealistic portrait of a person, natural light", guidance_scale=1)
for _ in range(4):
    out = stream(clip[0].permute(2, 0, 1))
N = 20
t0 = time.time()
for i in range(N):
    out = stream(clip[i % 16].permute(2, 0, 1))
dt = time.time() - t0
last = out.detach().cpu().float().clamp(0, 1)
Image.fromarray((last.permute(1, 2, 0).numpy() * 255).astype("uint8")).save("out_trt.png")
print(f"=== TensorRT: {N} frames in {dt:.2f}s -> {N/dt:.2f} FPS ===", flush=True)
