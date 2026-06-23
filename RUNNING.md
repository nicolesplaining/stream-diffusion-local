# StreamDiffusion on DGX Spark (GB10 / Blackwell, aarch64, CUDA 13)

Working setup as of 2026-06-23. The upstream repo pins ancient deps
(`torch==2.1.0`, `diffusers==0.24.0`, CUDA 11.8/12.1) that do **not** support
Blackwell (sm_121). The notes below are the adapted, working install.

## Environment

- GPU: NVIDIA GB10, compute capability sm_121 (Blackwell)
- CUDA toolkit: 13.0, driver 580.142
- Python: 3.12 (uv venv at `./.venv`)

## Install (from scratch)

```bash
# 1. venv
uv venv --python 3.12 .venv
source .venv/bin/activate

# 2. PyTorch — aarch64 + CUDA 13 wheels (sm_120 kernels are binary-compatible with sm_121)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# -> torch 2.12.1+cu130, torchvision 0.27.1+cu130

# 3. Runtime deps (MODERN versions — do NOT use the repo's pinned diffusers==0.24.0,
#    which breaks against current huggingface_hub/transformers)
uv pip install diffusers transformers accelerate fire omegaconf peft
# peft is required by modern diffusers to load the LCM LoRA

# 4. StreamDiffusion itself, WITHOUT deps (its setup.py would drag in old pins + xformers)
uv pip install --no-deps -e ./StreamDiffusion
```

## Run

```bash
source .venv/bin/activate
cd StreamDiffusion

# txt2img single image (acceleration MUST be "none" — see below)
python examples/txt2img/single.py --acceleration none \
  --prompt "a serene mountain landscape at sunset, oil painting"
# output -> StreamDiffusion/images/outputs/output.png

# throughput benchmark
python examples/benchmark/single.py --acceleration none
# baseline measured: ~8.8 FPS @ 512x512, 4 steps
```

## Real-time webcam img2img (visual demo)

`StreamDiffusion/webcam_img2img.py` — opens the webcam, runs every frame through
img2img, shows **input | output side by side** in a window, and lets you change
the prompt live by typing in the terminal.

```bash
source .venv/bin/activate
cd StreamDiffusion

# headless smoke test (no window): grabs frames, reports FPS, saves a sample
python webcam_img2img.py --selftest

# the live window (run in YOUR terminal so prompt typing works)
python webcam_img2img.py --prompt "a watercolor painting, vibrant colors"
```

Controls: type a new prompt + Enter in the terminal to change it live; `q`/ESC
in the window to quit; `s` to save the current output frame.

Useful flags: `--t-index 32,45` (fewer steps = faster, stays closer to the
webcam image), `--t-index 0,16,32,45` (more steps = more stylized), `--camera N`
(/dev/videoN), `--guidance-scale`, `--width/--height`, `--no-mirror`.

Measured ~8.6 FPS @ 512x512 with the default 3-step config (acceleration=none).

## Notes / gotchas

- **`--acceleration none`** is required for now. `xformers` has no aarch64/cu130
  wheel (would need a source build, marginal gain over torch SDPA on Blackwell).
  `tensorrt` needs the repo's TRT path ported to current diffusers (the
  `diffusers.models.unet_2d_condition` import paths in
  `src/streamdiffusion/acceleration/tensorrt/` have moved) + a CUDA-13 TensorRT.
  That's a separate, larger effort.
- Harmless warnings on every run: `CLIPFeatureExtractor deprecated`,
  `fuse_unet deprecated`, "no file named diffusion_pytorch_model.safetensors"
  (falls back to .bin), and a LoRA text-encoder prefix warning. All safe.
- Models download to `~/.cache/huggingface`. Root fs was ~93% full — watch space.
- Default model: `KBlueLeaf/kohaku-v2.1` (SD1.5) + `latent-consistency/lcm-lora-sdv1-5`.
