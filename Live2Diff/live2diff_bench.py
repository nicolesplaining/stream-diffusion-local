import os, sys, time, fire
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live2diff.utils.wrapper import StreamAnimateDiffusionDepthWrapper

def run(t_index="31,43", width=512, height=512, n=16):
    if isinstance(t_index, (tuple, list)):
        t_list = [int(x) for x in t_index]
    else:
        t_list = [int(x) for x in str(t_index).split(",")]
    img = Image.open("../StreamDiffusion/images/outputs/trt_real.png").convert("RGB").resize((width,height))
    arr = torch.from_numpy(np.array(img)).float()/255.0
    clip = arr[None].repeat(16,1,1,1)
    stream = StreamAnimateDiffusionDepthWrapper(
        few_step_model_type="lcm", config_path="configs/spike.yaml", cfg_type="none",
        strength=None, num_inference_steps=50, t_index_list=t_list, frame_buffer_size=1,
        width=width, height=height, acceleration="none", do_add_noise=True,
        output_type="pt", use_denoising_batch=True, use_tiny_vae=True, seed=42,
    )
    stream.prepare(warmup_frames=clip[:8].permute(0,3,1,2),
                   prompt="a photorealistic portrait of a person, natural light", guidance_scale=1)
    for _ in range(4): stream(clip[0].permute(2,0,1))  # warm
    t0=time.time()
    for i in range(n): out = stream(clip[i%16].permute(2,0,1))
    dt=time.time()-t0
    last=out.detach().cpu().float().clamp(0,1)
    Image.fromarray((last.permute(1,2,0).numpy()*255).astype("uint8")).save(f"out_bench_{len(t_list)}step_{width}.png")
    print(f"RESULT steps={len(t_list)} t_index={t_list} {width}x{height} -> {n/dt:.2f} FPS", flush=True)

if __name__ == "__main__":
    fire.Fire(run)
