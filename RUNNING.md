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

## Web app (browser, prompt + style picker)

`StreamDiffusion/web/server.py` — a FastAPI + WebSocket app. Browser captures the
webcam, the server runs img2img and streams frames back. The page has a **prompt
textbox** and a **style dropdown where each style is a different model**.

```bash
source .venv/bin/activate
cd StreamDiffusion
python web/server.py
# open http://localhost:8000 in a browser ON THE SPARK, click Start, allow camera
```

- Styles (each = one model), defined in `STYLES` at the top of `web/server.py`:
  Photorealistic (Realistic Vision 5.1), Hyperreal (Absolute Reality 1.81),
  Dreamy/Fantasy (Dreamshaper 8), Anime (kohaku-v2.1). Add more by appending entries.
- Models load on demand on first selection (download + warmup, slow once), then
  cached in memory — switching back is instant.
- The user prompt is combined with the style's built-in suffix; the style also
  sets the negative prompt.
- ~8.6 FPS over the WebSocket (JPEG+base64 round trip included).
- **Webcam needs localhost or HTTPS** — open it on the Spark's own browser at
  `http://localhost:8000`. From another machine, getUserMedia is blocked on plain
  http; use an SSH tunnel (`ssh -L 8000:localhost:8000 spark`) and open localhost.
- Override defaults via env: `SD_WIDTH`, `SD_HEIGHT`, `SD_T_INDEX` (e.g. "32,45"),
  `SD_ACCELERATION`.

## TensorRT acceleration (working on Blackwell/CUDA 13)

The repo's TensorRT path was 2023-era (TRT 9 API) and needed porting to run on
this box. It now works. Install the TRT stack and use `acceleration="tensorrt"`:

```bash
uv pip install tensorrt onnx onnxruntime polygraphy onnx-graphsurgeon cuda-python onnxscript
```

Then any entry point accepts TRT, e.g.:

```bash
python webcam_img2img.py --acceleration tensorrt          # desktop app
SD_ACCELERATION=tensorrt python web/server.py             # web app
python trt_build_test.py                                  # standalone build + benchmark
```

First use builds engines into `StreamDiffusion/engines/<model>/<config>/` (ONNX
export + TRT build, ~1-2 min the first time per model/batch/mode); afterwards the
cached engines load in seconds. Engines are gitignored (large, machine-specific).

Measured (img2img, Realistic Vision, 512x512, t_index=[32,45]):
**PyTorch ~11.5 FPS -> TensorRT ~14.7 FPS** (~1.3x). The gain grows with more
denoising steps (where the UNet dominates); at only 2 steps the VAE + overhead is
a large fraction. On this memory-bandwidth-bound box (~270 GB/s) don't expect the
2-4x that discrete GPUs see.

### What had to be patched (all in `src/streamdiffusion/acceleration/tensorrt/`)
TensorRT 11.1 (the only TRT for CUDA 13/aarch64) changed a lot vs the TRT-9 code:
- `engine.py`: 3 moved diffusers Output import paths (`models.unets.*`, `models.autoencoders.*`).
- `utilities.py`:
  - `from cuda import cudart` -> `from cuda.bindings import runtime as cudart` (cuda-python 13).
  - `Engine.build`/`load` rewritten to use **raw TensorRT** — polygraphy 0.50.3 is
    incompatible with TRT 11.1 (its `CreateConfig` errors on the FP16 flag).
  - TRT 11 has **no `BuilderFlag.FP16`** — precision now comes from a
    **strongly-typed network** built from the fp16 ONNX (`NetworkDefinitionCreationFlag.STRONGLY_TYPED`).
  - `Engine.allocate_buffers` rewritten for the TRT 10+ named-tensor API
    (the binding-index API — `get_bindings_per_profile`, `engine[idx]`,
    `get_binding_shape`, `binding_is_input`, `set_binding_shape` — was removed).
  - `export_onnx` pinned to the legacy exporter (`dynamo=False`); the torch 2.12
    default dynamo exporter path is not used.

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
