"""
Real-time webcam img2img with StreamDiffusion.

Opens your webcam, runs every frame through StreamDiffusion (img2img), and shows
the live input next to the generated output in a single window. You can change
the prompt on the fly by typing a new prompt in the terminal and pressing Enter.

Usage (from the StreamDiffusion repo root, with the venv active):

    python webcam_img2img.py --prompt "a watercolor painting, vibrant"

Controls:
    - Type a new prompt in the TERMINAL + Enter   -> updates live
    - Press 'q' or ESC in the window              -> quit
    - Press 's' in the window                     -> save current output frame

Headless smoke test (no window, no display needed):

    python webcam_img2img.py --selftest
"""

import os
import sys
import time
import threading
import argparse

import cv2
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.wrapper import StreamDiffusionWrapper


def parse_args():
    p = argparse.ArgumentParser(description="Real-time webcam img2img with StreamDiffusion")
    p.add_argument("--model", default="SG161222/Realistic_Vision_V5.1_noVAE",
                   help="HF model id or local path (default: photorealistic SD1.5)")
    p.add_argument("--prompt",
                   default="RAW photo, photorealistic portrait, natural skin texture, "
                           "35mm photograph, soft natural lighting, highly detailed, sharp focus")
    p.add_argument("--negative-prompt",
                   default="anime, illustration, cartoon, drawing, painting, cgi, 3d render, "
                           "low quality, bad quality, blurry, deformed, disfigured")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--camera", type=int, default=0, help="webcam device index (/dev/videoN)")
    p.add_argument("--t-index", default="22,32,45",
                   help="comma-separated denoising step indices. Fewer = faster & closer to input")
    p.add_argument("--guidance-scale", type=float, default=1.2)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--acceleration", default="none", choices=["none", "xformers", "tensorrt"])
    p.add_argument("--display-scale", type=float, default=1.0, help="scale the output window")
    p.add_argument("--no-mirror", action="store_true", help="do not horizontally flip the webcam")
    p.add_argument("--selftest", action="store_true",
                   help="headless: grab a few frames, report FPS, save one output, exit")
    return p.parse_args()


def center_square_crop(frame):
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame[y0:y0 + side, x0:x0 + side]


def bgr_frame_to_pil(frame, width, height, mirror):
    if mirror:
        frame = cv2.flip(frame, 1)
    frame = center_square_crop(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).resize((width, height))
    return img


def pil_to_bgr(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def build_stream(args, t_index_list):
    cfg_type = "self" if args.guidance_scale > 1.0 else "none"
    stream = StreamDiffusionWrapper(
        model_id_or_path=args.model,
        lora_dict=None,
        t_index_list=t_index_list,
        frame_buffer_size=1,
        width=args.width,
        height=args.height,
        warmup=10,
        acceleration=args.acceleration,
        mode="img2img",
        use_denoising_batch=True,
        cfg_type=cfg_type,
        output_type="pil",
        seed=args.seed,
    )
    stream.prepare(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=50,
        guidance_scale=args.guidance_scale,
        delta=args.delta,
    )
    return stream


def selftest(args, stream):
    print("[selftest] opening camera", args.camera)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[selftest] WARNING: could not open webcam, using a synthetic gradient frame")
        cap = None
        synth = np.tile(np.linspace(0, 255, 640, dtype=np.uint8), (480, 1))
        frame = cv2.cvtColor(synth, cv2.COLOR_GRAY2BGR)

    def grab():
        if cap is None:
            return frame
        ok, f = cap.read()
        return f if ok else frame

    img = bgr_frame_to_pil(grab(), args.width, args.height, mirror=False)
    print("[selftest] priming buffer (batch_size=%d)..." % stream.batch_size)
    for _ in range(stream.batch_size - 1):
        stream(image=img)

    n = 20
    t0 = time.time()
    out = None
    for _ in range(n):
        img = bgr_frame_to_pil(grab(), args.width, args.height, mirror=False)
        out = stream(image=img)
    dt = time.time() - t0
    if cap is not None:
        cap.release()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "images", "outputs", "webcam_selftest.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.save(out_path)
    print(f"[selftest] {n} frames in {dt:.2f}s -> {n/dt:.2f} FPS")
    print(f"[selftest] saved sample output to {out_path}")


def run_live(args, stream):
    # shared prompt state, updated from a background stdin reader thread
    state = {"prompt": args.prompt, "dirty": False}
    lock = threading.Lock()

    def stdin_reader():
        while True:
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if not line:
                break
            line = line.strip()
            if line:
                with lock:
                    state["prompt"] = line
                    state["dirty"] = True
                print(f">> prompt updated: {line}")

    threading.Thread(target=stdin_reader, daemon=True).start()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: could not open webcam index {args.camera}")
        sys.exit(1)

    print("\n=== Real-time webcam img2img ===")
    print("Type a new prompt here + Enter to change it live.")
    print("In the window: 'q'/ESC to quit, 's' to save a frame.\n")

    # prime the denoising buffer
    ok, frame = cap.read()
    if not ok:
        print("ERROR: failed to read from webcam")
        sys.exit(1)
    img = bgr_frame_to_pil(frame, args.width, args.height, not args.no_mirror)
    for _ in range(stream.batch_size - 1):
        stream(image=img)

    win = "StreamDiffusion webcam img2img  (input | output)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    last_prompt = args.prompt
    fps = 0.0
    save_idx = 0

    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            print("WARNING: dropped frame")
            continue

        img = bgr_frame_to_pil(frame, args.width, args.height, not args.no_mirror)

        with lock:
            if state["dirty"]:
                last_prompt = state["prompt"]
                state["dirty"] = False
                send_prompt = last_prompt
            else:
                send_prompt = None

        out = stream(image=img, prompt=send_prompt)
        out_bgr = pil_to_bgr(out)
        in_bgr = pil_to_bgr(img)

        # side-by-side panel
        panel = np.hstack([in_bgr, out_bgr])
        if args.display_scale != 1.0:
            panel = cv2.resize(panel, None, fx=args.display_scale, fy=args.display_scale)

        # overlays
        dt = time.time() - t0
        fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)
        cv2.putText(panel, f"{fps:4.1f} FPS", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(panel, last_prompt[:60], (10, panel.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(win, panel)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "images", "outputs", f"webcam_{save_idx:04d}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            out.save(path)
            print(f"saved {path}")
            save_idx += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    t_index_list = [int(x) for x in args.t_index.split(",") if x.strip() != ""]
    print(f"Loading StreamDiffusion (model={args.model}, t_index={t_index_list}, "
          f"acceleration={args.acceleration})...")
    stream = build_stream(args, t_index_list)
    if args.selftest:
        selftest(args, stream)
    else:
        run_live(args, stream)


if __name__ == "__main__":
    main()
