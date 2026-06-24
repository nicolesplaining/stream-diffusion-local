# Live2Diff setup on DGX Spark (GB10/Blackwell, sm_121, aarch64, CUDA 13)

Temporally-coherent real-time webcam video-to-video. This repo is **vendored** into
the parent project (its code — including the TensorRT-11 patches and
`configs/spike.yaml` — and the MiDaS submodule are already plain files here), so
reconstituting the environment only means: **recreate the venv, install deps, and
download weights.** No code changes are needed.

> Same-machine note: if you're just SSHing back into the Spark, none of this is
> needed — the venv (`../.venv-live2diff`), weights (`models/`), and TRT engines
> (`engines_trt*/`) already exist. This doc is for a fresh machine / rebuild.

All commands are run from this `Live2Diff/` directory.

## 1. venv + PyTorch (Blackwell wheels)

```bash
uv venv --python 3.10 ../.venv-live2diff
uv pip install --python ../.venv-live2diff torch torchvision \
  --index-url https://download.pytorch.org/whl/cu130
```

## 2. Dependencies

The trick: keep Live2Diff's **pinned `diffusers==0.25.0`** (its model code couples
to 0.25 internals — do NOT bump it) but pair it with the modern Blackwell torch.
Install a compatible constellation, then the package with `--no-deps` (its setup.py
lists `decord`/`lightning`/`av`, which have no aarch64 wheels and are never imported).

```bash
uv pip install --python ../.venv-live2diff \
  "diffusers==0.25.0" "transformers==4.40.2" "huggingface_hub==0.25.2" \
  "tokenizers<0.20" "accelerate==0.30.1" "peft==0.11.1" \
  einops omegaconf fire imageio "timm==0.6.7" safetensors numpy
uv pip install --python ../.venv-live2diff --no-deps -e .
```

## 3. TensorRT stack (optional — for acceleration)

```bash
uv pip install --python ../.venv-live2diff \
  tensorrt onnx onnxruntime polygraphy onnx-graphsurgeon cuda-python colored onnxscript
```

The vendored `live2diff/acceleration/tensorrt/utilities.py` is already patched for
TensorRT 11.1 / CUDA 13 (raw-TRT build, strongly-typed network, `cuda.bindings.runtime`,
`get_tensor_name`, `dynamo=False`). If you ever re-clone from upstream instead of using
the vendored copy, those patches must be re-applied — see the parent `RUNNING.md`.

## 4. Model weights (~12 GB) -> `models/`

```bash
source ../.venv-live2diff/bin/activate
python - <<'PY'
import os
from huggingface_hub import hf_hub_download, snapshot_download
hf_hub_download("Leoxing/Live2Diff", "live2diff.ckpt", local_dir="models")
snapshot_download("stable-diffusion-v1-5/stable-diffusion-v1-5",
    local_dir="models/Model/stable-diffusion-v1-5",
    allow_patterns=["*.json","*.txt","tokenizer/*","scheduler/*","feature_extractor/*",
                    "text_encoder/*.safetensors","vae/*.safetensors","unet/*.safetensors"])
PY
# MiDaS depth model (not on HF):
curl -L https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt \
  -o models/dpt_hybrid_384.pt
```

Expected layout: `models/live2diff.ckpt`, `models/dpt_hybrid_384.pt`,
`models/Model/stable-diffusion-v1-5/{unet,vae,text_encoder,tokenizer,scheduler}`.

## 5. Run

```bash
source ../.venv-live2diff/bin/activate

# headless build + benchmark (no accel)
python spike_test.py

# benchmark a config (fewer steps / smaller res = faster)
python live2diff_bench.py --t_index 31,43 --width 384 --height 384

# build TensorRT engines + benchmark (first build ~3 min; engines cached in engines_trt*/)
python live2diff_trt.py
```

TensorRT engines are **machine-specific and gitignored** — they always rebuild on
first run on a new machine. `configs/spike.yaml` is a minimal vanilla-SD1.5 config
(note `third_party_dict.dreambooth: null`, required by the wrapper).

## Measured FPS (this Spark, 512x512 unless noted)

| Config | FPS |
|---|---|
| 4-step @512, no accel (original) | 1.74 |
| 2-step @512, no accel | 3.15 |
| 2-step @384, no accel | 4.96 |
| 2-step @512, TensorRT | 5.62 |
| **2-step @384, TensorRT** | **9.38** |

1-step does **not** work — the streaming kv-cache window needs >= 2 denoising steps.
