import os, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live2diff.utils.wrapper import StreamAnimateDiffusionDepthWrapper

CFG = "configs/spike.yaml"
H = W = 512

# real structured input: reuse the portrait from the StreamDiffusion work, as a static clip
src_path = "../StreamDiffusion/images/outputs/trt_real.png"
if not os.path.exists(src_path):
    src_path = "../StreamDiffusion/images/outputs/webcam_selftest.png"
img = Image.open(src_path).convert("RGB").resize((W, H))
arr = torch.from_numpy(np.array(img)).float() / 255.0   # [H,W,3]
frame_hw3 = arr                                          # one frame
clip = frame_hw3[None].repeat(16, 1, 1, 1)               # [16,H,W,3]

print("=== building Live2Diff pipeline (acceleration=none) ===", flush=True)
t0 = time.time()
stream = StreamAnimateDiffusionDepthWrapper(
    few_step_model_type="lcm",
    config_path=CFG,
    cfg_type="none",
    strength=None,
    num_inference_steps=50,
    t_index_list=[25, 31, 37, 43],
    frame_buffer_size=1,
    width=W, height=H,
    acceleration="none",
    do_add_noise=True,
    output_type="pt",
    use_denoising_batch=True,
    use_tiny_vae=True,
    seed=42,
)
print(f"=== wrapper built in {time.time()-t0:.1f}s ===", flush=True)

warmup = clip[:8].permute(0, 3, 1, 2)  # [8,3,H,W]
res = stream.prepare(warmup_frames=warmup, prompt="a photorealistic portrait of a person, natural light, sharp focus", guidance_scale=1)
print("prepare() OK; warmup result shape:", tuple(res.shape), flush=True)

# run a few streaming frames + time them
skip = stream.batch_size - 1
N = 12
outs = []
t0 = time.time()
for i in range(N):
    out = stream(clip[i % clip.shape[0]].permute(2, 0, 1))  # [3,H,W]
    outs.append(out)
dt = time.time() - t0
last = outs[-1].detach().cpu().float().clamp(0, 1)
Image.fromarray((last.permute(1, 2, 0).numpy() * 255).astype("uint8")).save("outputs_spike.png")
print(f"=== ran {N} frames in {dt:.2f}s -> {N/dt:.2f} FPS (acceleration=none) ===", flush=True)
print("saved outputs_spike.png; output std (texture if >5):", round(float(last.std()*255), 1), flush=True)
