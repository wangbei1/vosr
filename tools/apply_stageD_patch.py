#!/usr/bin/env python3
"""Apply Stage-D Ref-VOSR patches to an existing VOSR project.

Usage:
  cd /data/zhangdanning/Projects/VOSR
  python tools/apply_stageD_patch.py --train train_vosr_refldm_modular.py

The script is conservative: it creates .stageD.bak backups and only performs
simple text edits. Inspect the printed [OK]/[WARN] messages after running.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path.cwd()


def backup(path: Path):
    bak = path.with_suffix(path.suffix + ".stageD.bak")
    if not bak.exists():
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return bak


def replace_once(text: str, old: str, new: str, desc: str) -> str:
    if new in text:
        print(f"[OK] {desc}: already patched")
        return text
    if old not in text:
        print(f"[WARN] {desc}: pattern not found")
        return text
    print(f"[OK] {desc}")
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train_vosr_refldm_modular.py")
    args = parser.parse_args()

    train_path = ROOT / args.train
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    backup(train_path)
    text = train_path.read_text(encoding="utf-8")

    # 1) Imports.
    old = "from models.refldm_vosr import RefLDMVOSRAddon, patch_vosr_loss_return_extra"
    new = (
        "from models.refldm_vosr import RefLDMVOSRAddon, patch_vosr_loss_return_extra\n"
        "from models.refldm_vosr.addon_stageD import RefLDMVOSRStageDAddon\n"
        "from models.refldm_vosr.ref_attention_patch import patch_lightningdit_ref_attention\n"
        "from models.refldm_vosr.vosr_stageD_patch import patch_vosr_stageD\n"
        "from models.refldm_vosr.eval_refaware import build_eval_condition"
    )
    text = replace_once(text, old, new, "add Stage-D imports")

    # 2) Patch VOSR class.
    old = "patch_vosr_loss_return_extra(VOSR)"
    new = "patch_vosr_loss_return_extra(VOSR)\npatch_vosr_stageD(VOSR)"
    text = replace_once(text, old, new, "patch VOSR Stage-D methods")

    # 3) Patch model right after construction and before param counting.
    old = "\n    total_params = sum(p.numel() for p in model.parameters())"
    new = (
        "\n    if getattr(args, \"use_ref_attention\", False):\n"
        "        patch_lightningdit_ref_attention(model, args)\n"
        "        if accelerator.is_main_process:\n"
        "            logger.info(\"[Stage-D] Enabled Ref cross-attention in LightningDiT.\")\n"
        "\n    total_params = sum(p.numel() for p in model.parameters())"
    )
    text = replace_once(text, old, new, "enable ref cross-attention before optimizer")

    # 4) Use Stage-D addon.
    old = "refldm_addon = RefLDMVOSRAddon("
    new = "refldm_addon = RefLDMVOSRStageDAddon("
    text = replace_once(text, old, new, "use Stage-D addon")

    # 5) Pass global_step into compute_loss.
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
    text = replace_once(text, old, new, "pass global_step to identity-loss scheduler")

    # 6) Ref-aware CSV eval: after the existing LQ-only z extraction, override z if ref is enabled.
    old = (
        "z = [z[i] for i in args.layer_dinov2b_list]\n\n"
        "                                    if args.ae_type == \"qwen\":"
    )
    new = (
        "z = [z[i] for i in args.layer_dinov2b_list]\n\n"
        "                                    # Stage-D: override eval condition with LQ+Ref if enabled.\n"
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
        "                                            logger.info(f\"[Eval Ref Debug] {name} | use_ref_attention={getattr(args, 'use_ref_attention', False)} | use_ref_feature={getattr(args, 'use_ref_feature', False)} | refs={ref_names_eval}\")\n"
        "\n"
        "                                    if args.ae_type == \"qwen\":"
    )
    text = replace_once(text, old, new, "make CSV eval ref-aware")

    # 7) Use ref-aware sampler when z is a dict. Replace both qwen/sd2 sample calls.
    old = "sr_m11 = vosr.sample_multistep_fm(ema, lq_latent, n_steps=args.infer_steps, venc_fea=z)"
    new = (
        "sr_m11 = (\n"
        "                                            vosr.sample_multistep_fm_ref(ema, lq_latent, n_steps=args.infer_steps, venc_fea=z.get('lq'), ref_fea=z.get('ref'))\n"
        "                                            if isinstance(z, dict) and getattr(args, 'use_ref_attention', False)\n"
        "                                            else vosr.sample_multistep_fm(ema, lq_latent, n_steps=args.infer_steps, venc_fea=z)\n"
        "                                        )"
    )
    if old in text:
        text = text.replace(old, new)
        print("[OK] use ref-aware sampler in eval")
    else:
        print("[WARN] eval sampler pattern not found")

    train_path.write_text(text, encoding="utf-8")

    # 8) __init__.py exports.
    init_path = ROOT / "models/refldm_vosr/__init__.py"
    if init_path.exists():
        backup(init_path)
        init = init_path.read_text(encoding="utf-8")
        additions = "\nfrom .ref_attention_patch import patch_lightningdit_ref_attention\nfrom .vosr_stageD_patch import patch_vosr_stageD\nfrom .addon_stageD import RefLDMVOSRStageDAddon\n"
        if "patch_lightningdit_ref_attention" not in init:
            init += additions
            init_path.write_text(init, encoding="utf-8")
            print("[OK] update models/refldm_vosr/__init__.py")

    print("\nDone. Inspect your train script and run a short smoke test.")


if __name__ == "__main__":
    main()
