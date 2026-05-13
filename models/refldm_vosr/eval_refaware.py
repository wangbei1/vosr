"""Small helpers for FFHQ-Ref ref-aware evaluation."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List

import torch
from PIL import Image


def parse_ref_names(ref_value) -> List[str]:
    if ref_value is None:
        return []
    if isinstance(ref_value, list):
        refs = ref_value
    elif isinstance(ref_value, str):
        try:
            refs = ast.literal_eval(ref_value)
        except Exception:
            refs = [ref_value]
    else:
        refs = []
    if not isinstance(refs, list):
        refs = []
    return [str(x) for x in refs if str(x).strip() != ""]


@torch.no_grad()
def load_ref_tensor_from_row(row, hq_dir, to_tensor, device, args, resize_to_512_fn=None):
    ref_names = parse_ref_names(row.get("ref_image", None))
    k = int(getattr(args, "ffhq_ref_num_refs", 1))
    ref_names = ref_names[:k]
    refs = []
    for name in ref_names:
        path = Path(hq_dir) / name
        if not path.exists():
            continue
        with Image.open(path) as img:
            ref_01 = to_tensor(img.convert("RGB")).unsqueeze(0).to(device)
        if resize_to_512_fn is not None:
            ref_01 = resize_to_512_fn(ref_01)
        refs.append(ref_01.squeeze(0))
    if len(refs) == 0:
        return None, []
    ref_eval = torch.stack(refs, dim=0).unsqueeze(0)  # [1,K,3,H,W]
    ref_eval = ref_eval * 2.0 - 1.0
    return ref_eval, ref_names


@torch.no_grad()
def build_eval_condition(refldm_addon, lq_eval_m11, row, hq_dir, to_tensor, device, args, venc, resize_to_512_fn=None):
    z_lq = refldm_addon.feature_extractor.extract(lq_eval_m11, venc)
    if not (getattr(args, "use_refldm", False) and (getattr(args, "use_ref_feature", False) or getattr(args, "use_ref_attention", False))):
        return z_lq, None, []

    ref_eval, ref_names = load_ref_tensor_from_row(row, hq_dir, to_tensor, device, args, resize_to_512_fn)
    if ref_eval is None:
        return z_lq, None, []
    ref_z = refldm_addon.feature_extractor.extract_ref(ref_eval, venc)

    if getattr(args, "use_ref_attention", False):
        z = {"lq": z_lq, "ref": ref_z}
    elif getattr(args, "use_ref_feature", False):
        z = refldm_addon.feature_extractor.fuse_lq_ref(z_lq, ref_z)
    else:
        z = z_lq
    return z, ref_z, ref_names
