"""
Retinal fundus image(s) -> disease prediction + explanation, using a PRETRAINED
model (no training needed).

Model: Mjolnirslams/retinal-disease-ensemble (Hugging Face)
       EfficientNet-B4 + Inception-ResNet-v2 ensemble, trained on ODIR-5K,
       8 labels: N, D, G, C, A, H, M, O.

Usage (PowerShell):
  python main.py --left left_eye.jpg --right right_eye.jpg
  python main.py --image eye.jpg                # single image (used for both eyes)
  python main.py --image eye.jpg --no-heatmap   # skip the slower explanation map

First run downloads the model (~300 MB) from Hugging Face, so it needs internet.
"""
import argparse
import importlib.util
import sys

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from disease_info import INFO, build_report

REPO = "Mjolnirslams/retinal-disease-ensemble"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    # The model ships its own loader (modeling_retinal.py). This downloads and
    # runs third-party code from the repo, so skim that file once before trusting it.
    code = hf_hub_download(REPO, "modeling_retinal.py")
    spec = importlib.util.spec_from_file_location("modeling_retinal", code)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_ensemble_from_hub(REPO, device=DEVICE)


def occlusion_map(model, left, right, code, which, grid=6):
    """
    Model-agnostic explanation: grey out one patch at a time and measure how much
    the probability of `code` drops. Big drop = that region mattered.
    Needs only model.predict(), so it works without access to model internals.
    """
    img = left if which == "left" else right
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    base, _ = model.predict(left, right, tta=False)
    base = base[code]
    fill = arr.reshape(-1, 3).mean(axis=0)
    ys = np.linspace(0, h, grid + 1, dtype=int)
    xs = np.linspace(0, w, grid + 1, dtype=int)
    heat = np.zeros((grid, grid), np.float32)
    for i in range(grid):
        for j in range(grid):
            occ = arr.copy()
            occ[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = fill
            occ_img = Image.fromarray(occ)
            args = (occ_img, right) if which == "left" else (left, occ_img)
            p, _ = model.predict(*args, tta=False)
            heat[i, j] = max(0.0, base - p[code])
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC)
    heat = np.clip(heat, 0, None)
    return heat / (heat.max() + 1e-8), arr


def overlay(arr, heat, alpha=0.45, height=448):
    color = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    mix = np.uint8((1 - alpha) * arr + alpha * color)
    s = height / arr.shape[0]
    size = (int(arr.shape[1] * s), height)
    return cv2.resize(arr, size), cv2.resize(mix, size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="single fundus image (used for both eyes)")
    ap.add_argument("--left", help="left-eye fundus image")
    ap.add_argument("--right", help="right-eye fundus image")
    ap.add_argument("--out", default="result.png")
    ap.add_argument("--grid", type=int, default=6, help="occlusion grid size")
    ap.add_argument("--no-heatmap", action="store_true")
    ap.add_argument("--no-tta", action="store_true", help="faster, slightly less accurate")
    args = ap.parse_args()

    single = False
    if args.image and not (args.left or args.right):
        left = right = Image.open(args.image).convert("RGB")
        single = True
    elif args.left and args.right:
        left = Image.open(args.left).convert("RGB")
        right = Image.open(args.right).convert("RGB")
    else:
        sys.exit("Give either --image eye.jpg  OR  --left L.jpg --right R.jpg")

    print(f"Loading model on {DEVICE} (first run downloads weights)...")
    model = load_model()

    print("Predicting...")
    probs, preds = model.predict(left, right, tta=not args.no_tta)
    probs = {k: float(v) for k, v in probs.items()}
    preds = {k: bool(v) for k, v in preds.items()}
    print(build_report(probs, preds, single_eye=single))

    flagged = [c for c in sorted(probs, key=probs.get, reverse=True)
               if preds.get(c) and c != "N"]
    if flagged and not args.no_heatmap:
        code = flagged[0]
        print(f"\nBuilding explanation heatmap for: {INFO[code]['name']} "
              f"(this runs the model ~{args.grid ** 2} times per eye)...")
        panels = []
        eyes = [("left", left)] if single else [("left", left), ("right", right)]
        for which, _ in eyes:
            heat, arr = occlusion_map(model, left, right, code, which, args.grid)
            orig, mix = overlay(arr, heat)
            panels += [orig, mix]
        canvas = np.hstack(panels)
        cv2.putText(canvas, f"{code}: {probs[code]:.2f}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(args.out, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"Saved: {args.out}  (original | regions that drove the prediction)")


if __name__ == "__main__":
    main()