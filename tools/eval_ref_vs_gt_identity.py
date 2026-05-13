#!/usr/bin/env python3
"""Evaluate saved SR images against GT and FFHQ-Ref references with identity loss.

This script is independent from training. It checks whether outputs are closer
to GT or reference identities. Lower id_loss is better; ids_like = 1 - id_loss.

Usage:
  python tools/eval_ref_vs_gt_identity.py \
    --csv /data/.../test_references_small10.csv \
    --sr_dir exp_vosr_test_C/.../sr/00000601 \
    --hq_dir /data/.../images1024x1024 \
    --identity_model_path /data/.../insightface_webface_r50.onnx \
    --out_csv eval_ref_vs_gt.csv
"""

from __future__ import annotations
import argparse, ast, csv
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


def parse_refs(x):
    if isinstance(x, list):
        refs = x
    elif isinstance(x, str):
        try: refs = ast.literal_eval(x)
        except Exception: refs = [x]
    else:
        refs = []
    return [str(r) for r in refs if str(r).strip()]


def load_m11(path, size=512, device="cuda"):
    to_tensor = transforms.ToTensor()
    with Image.open(path) as im:
        x = to_tensor(im.convert("RGB")).unsqueeze(0).to(device)
    if size is not None:
        x = torch.nn.functional.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0,1)
    return x * 2 - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--sr_dir", required=True)
    ap.add_argument("--hq_dir", required=True)
    ap.add_argument("--identity_model_path", required=True)
    ap.add_argument("--max_refs", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_csv", default="eval_ref_vs_gt_identity.csv")
    args = ap.parse_args()

    try:
        from models.refldm_vosr.identity_loss import IdentityLoss
    except Exception:
        from identity_loss import IdentityLoss
    model = IdentityLoss(model_path=args.identity_model_path).to(args.device).eval()

    df = pd.read_csv(args.csv)
    rows = []
    for _, row in df.iterrows():
        gt_name = row["gt_image"]
        stem = Path(gt_name).stem
        sr_path = Path(args.sr_dir) / f"{stem}.png"
        gt_path = Path(args.hq_dir) / gt_name
        if not sr_path.exists() or not gt_path.exists():
            continue
        sr = load_m11(sr_path, device=args.device)
        gt = load_m11(gt_path, device=args.device)
        with torch.no_grad():
            id_gt = model(sr, gt)
            if isinstance(id_gt, tuple): id_gt = id_gt[0]
            id_gt = float(id_gt.mean().detach().cpu())
        ref_losses = []
        ref_names = parse_refs(row.get("ref_image", None))[:args.max_refs]
        for rn in ref_names:
            rp = Path(args.hq_dir) / rn
            if not rp.exists():
                continue
            ref = load_m11(rp, device=args.device)
            with torch.no_grad():
                val = model(sr, ref)
                if isinstance(val, tuple): val = val[0]
                ref_losses.append(float(val.mean().detach().cpu()))
        id_ref = sum(ref_losses) / len(ref_losses) if ref_losses else None
        rows.append({
            "name": stem,
            "id_loss_gt": id_gt,
            "ids_gt_like": 1.0 - id_gt,
            "id_loss_ref": id_ref,
            "ids_ref_like": None if id_ref is None else 1.0 - id_ref,
            "n_refs": len(ref_losses),
        })
    out = Path(args.out_csv)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name","id_loss_gt","ids_gt_like","id_loss_ref","ids_ref_like","n_refs"])
        w.writeheader(); w.writerows(rows)
    if rows:
        avg_gt = sum(r["id_loss_gt"] for r in rows) / len(rows)
        ref_vals = [r["id_loss_ref"] for r in rows if r["id_loss_ref"] is not None]
        avg_ref = sum(ref_vals) / len(ref_vals) if ref_vals else None
        print(f"samples={len(rows)} avg_id_loss_gt={avg_gt:.6f} avg_ids_gt_like={1-avg_gt:.6f}")
        if avg_ref is not None:
            print(f"samples_ref={len(ref_vals)} avg_id_loss_ref={avg_ref:.6f} avg_ids_ref_like={1-avg_ref:.6f}")
    print(f"saved: {out}")

if __name__ == "__main__":
    main()
