#!/usr/bin/env python3
"""Apply Stage-E patches to train_vosr_refldm_modular.py.

Stage-E adds:
  1) Ref-target identity loss via RefLDMVOSRStageEAddon.
  2) True DiT ref cross-attention via patch_lightningdit_ref_attention.
  3) Ref-aware eval condition and sampler.

Usage:
  cd /data/zhangdanning/Projects/VOSR
  python tools/apply_stageE_patch.py --train train_vosr_refldm_modular.py
"""

from __future__ import annotations
import argparse
from pathlib import Path

ROOT = Path.cwd()

def backup(path: Path):
    bak = path.with_suffix(path.suffix + ".stageE.bak")
    if not bak.exists():
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return bak

def ensure_import(text: str, line: str, after_anchor: str) -> str:
    if line in text:
        print(f"[OK] import exists: {line}")
        return text
    if after_anchor in text:
        print(f"[OK] add import: {line}")
        return text.replace(after_anchor, after_anchor + "\n" + line, 1)
    print(f"[WARN] import anchor not found: {after_anchor}")
    return text

def replace(text: str, old: str, new: str, desc: str) -> str:
    if new in text:
        print(f"[OK] {desc}: already patched")
        return text
    if old not in text:
        print(f"[WARN] {desc}: pattern not found")
        return text
    print(f"[OK] {desc}")
    return text.replace(old, new, 1)

def replace_all(text: str, old: str, new: str, desc: str) -> str:
    if old not in text:
        print(f"[WARN] {desc}: pattern not found")
        return text
    n = text.count(old)
    print(f"[OK] {desc}: replace {n} occurrence(s)")
    return text.replace(old, new)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train_vosr_refldm_modular.py")
    args = ap.parse_args()
    p = ROOT / args.train
    if not p.exists():
        raise FileNotFoundError(p)
    backup(p)
    text = p.read_text(encoding="utf-8")

    # Imports.
    anchor = "from models.refldm_vosr import RefLDMVOSRAddon, patch_vosr_loss_return_extra"
    text = ensure_import(text, "from models.refldm_vosr.addon_stageE import RefLDMVOSRStageEAddon", anchor)
    text = ensure_import(text, "from models.refldm_vosr.ref_attention_patch import patch_lightningdit_ref_attention", anchor)
    text = ensure_import(text, "from models.refldm_vosr.vosr_stageE_patch import patch_vosr_stageD", anchor)
    text = ensure_import(text, "from models.refldm_vosr.eval_refaware import build_eval_condition", anchor)

    # Patch VOSR class.
    text = replace(text, "patch_vosr_loss_return_extra(VOSR)", "patch_vosr_loss_return_extra(VOSR)\npatch_vosr_stageD(VOSR)", "patch VOSR Stage-E methods")

    # Patch model after construction and before parameter count.
    old = "\n    total_params = sum(p.numel() for p in model.parameters())"
    new = (
        "\n    if getattr(args, \"use_ref_attention\", False):\n"
        "        patch_lightningdit_ref_attention(model, args)\n"
        "        if accelerator.is_main_process:\n"
        "            logger.info(\"[Stage-E] Enabled Ref cross-attention in LightningDiT.\")\n"
        "\n    total_params = sum(p.numel() for p in model.parameters())"
    )
    text = replace(text, old, new, "enable Ref cross-attention before optimizer")

    # Use Stage-E addon. Handles either original or Stage-D names.
    text = replace_all(text, "refldm_addon = RefLDMVOSRAddon(", "refldm_addon = RefLDMVOSRStageEAddon(", "use Stage-E addon from original")
    text = replace_all(text, "refldm_addon = RefLDMVOSRStageDAddon(", "refldm_addon = RefLDMVOSRStageEAddon(", "use Stage-E addon from Stage-D")

    # Pass global_step to compute_loss.
    old = (
        "loss, loss_backward, loss_log_dict = refldm_addon.compute_loss(\n"
        "                        vosr=vosr,\n"
        "                        model=model,\n"
        "                        model_ae=unwrapped_model_ae,\n"
        "                        ref_pack=ref_pack,\n"
        "                    )"
    )
    new = (
        "loss, loss_backward, loss_log_dict = refldm_addon.compute_loss(\n"
        "                        vosr=vosr,\n"
        "                        model=model,\n"
        "                        model_ae=unwrapped_model_ae,\n"
        "                        ref_pack=ref_pack,\n"
        "                        global_step=global_step,\n"
        "                    )"
    )
    text = replace(text, old, new, "pass global_step to identity scheduler")

    # Add loss_id_ref/gt to logging if only loss_id is logged.
    old = (
        "if \"loss_id\" in loss_log_dict:\n"
        "                    logs[\"id_loss\"] = loss_log_dict[\"loss_id\"].item()"
    )
    new = (
        "if \"loss_id\" in loss_log_dict:\n"
        "                    logs[\"id_loss\"] = loss_log_dict[\"loss_id\"].item()\n"
        "                if \"loss_id_ref\" in loss_log_dict:\n"
        "                    logs[\"id_loss_ref\"] = loss_log_dict[\"loss_id_ref\"].item()\n"
        "                if \"loss_id_gt\" in loss_log_dict:\n"
        "                    logs[\"id_loss_gt\"] = loss_log_dict[\"loss_id_gt\"].item()"
    )
    text = replace_all(text, old, new, "log ref/gt identity losses")

    # Ref-aware CSV eval condition: override z after LQ-only extraction.
    old = (
        "z = [z[i] for i in args.layer_dinov2b_list]\n\n"
        "                                    if args.ae_type == \"qwen\":"
    )
    new = (
        "z = [z[i] for i in args.layer_dinov2b_list]\n\n"
        "                                    # Stage-E: override eval condition with LQ+Ref if enabled.\n"
        "                                    ref_z_eval = None\n"
        "                                    ref_names_eval = []\n"
        "                                    if getattr(args, \"use_refldm\", False) and (getattr(args, \"use_ref_feature\", False) or getattr(args, \"use_ref_attention\", False)):\n"
        "                                        z, ref_z_eval, ref_names_eval = build_eval_condition(\n"
        "                                            refldm_addon=refldm_addon,\n"
        "                                            lq_eval_m11=lq_eval,\n"
        "                                            row=row,\n"
        "                                            hq_dir=hq_dir,\n"
        "                                            to_tensor=to_tensor,\n"
        "                                            device=accelerator.device,\n"
        "                                            args=args,\n"
        "                                            venc=unwrapped_venc,\n"
        "                                            resize_to_512_fn=resize_to_512,\n"
        "                                        )\n"
        "                                        if accelerator.is_main_process and n_img < 3:\n"
        "                                            logger.info(f\"[Eval Ref Debug] {name} | use_ref_attention={getattr(args, 'use_ref_attention', False)} | refs={ref_names_eval}\")\n\n"
        "                                    if args.ae_type == \"qwen\":"
    )
    text = replace(text, old, new, "make CSV eval condition ref-aware")

    # Ref-aware sampler: replace two sample_multistep_fm calls in eval blocks conservatively.
    old = "sr_m11 = vosr.sample_multistep_fm(ema, lq_latent, n_steps=args.infer_steps, venc_fea=z)"
    new = (
        "sr_m11 = (\n"
        "                                            vosr.sample_multistep_fm_ref(ema, lq_latent, n_steps=args.infer_steps, venc_fea=z.get('lq'), ref_fea=z.get('ref'))\n"
        "                                            if isinstance(z, dict) and hasattr(vosr, 'sample_multistep_fm_ref')\n"
        "                                            else vosr.sample_multistep_fm(ema, lq_latent, n_steps=args.infer_steps, venc_fea=z)\n"
        "                                        )"
    )
    text = replace_all(text, old, new, "use ref-aware sampler when z is dict")

    p.write_text(text, encoding="utf-8")
    print(f"[DONE] Patched {p}")
    print("[NEXT] Check imports and run a 10-step smoke test before full training.")

if __name__ == "__main__":
    main()
