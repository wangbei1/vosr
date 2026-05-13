import ast
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


def _parse_refs(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            refs = ast.literal_eval(value)
            return refs if isinstance(refs, list) else []
        except Exception:
            return []
    return []


@torch.no_grad()
def run_ffhq_ref_eval(
    args,
    accelerator,
    logger,
    vosr,
    eval_model,
    model_ae,
    venc,
    addon,
    global_step,
    iqa_lpips=None,
    iqa_musiq=None,
    latents_mean=None,
    latents_std=None,
    to_pil=None,
    resize_to_512=None,
):
    """Reference-aware CSV evaluation.

    It uses Ref features when args.use_ref_feature=true; otherwise it behaves as
    baseline VOSR while still saving images and metrics.
    """
    if not getattr(args, "ffhq_ref_test_use_csv_for_eval", False):
        logger.warning("run_ffhq_ref_eval called but ffhq_ref_test_use_csv_for_eval=false")
        return None

    to_tensor = transforms.ToTensor()
    to_pil = to_pil or transforms.ToPILImage()
    sr_dir = Path(args.output_dir, "sr", f"{global_step:08d}")
    sr_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.ffhq_ref_test_csv)
    hq_dir = Path(args.ffhq_ref_eval_hq_dir)
    lq_dir = Path(args.ffhq_ref_eval_lq_dir)
    ref_dir = Path(getattr(args, "ffhq_ref_hq_dir", args.ffhq_ref_eval_hq_dir))

    lpips_sum, musiq_sum, n_img = 0.0, 0.0, 0
    max_eval = int(getattr(args, "ffhq_ref_eval_max_images", 0))

    for _, row in df.iterrows():
        if max_eval > 0 and n_img >= max_eval:
            break
        gt_name, lq_name = row["gt_image"], row["lq_image"]
        gt_path, lq_path = hq_dir / gt_name, lq_dir / lq_name
        if not gt_path.exists() or not lq_path.exists():
            continue

        with Image.open(lq_path) as img:
            lq_img = img.convert("RGB")
        if getattr(args, "test_upscale", 1) > 1:
            w, h = lq_img.size
            lq_img = lq_img.resize((w * args.test_upscale, h * args.test_upscale), Image.Resampling.LANCZOS)
        lq_eval = to_tensor(lq_img).unsqueeze(0).to(accelerator.device) * 2.0 - 1.0

        with Image.open(gt_path) as img:
            gt_01 = to_tensor(img.convert("RGB")).unsqueeze(0).to(accelerator.device)

        z_lq = addon.feature_extractor.extract(lq_eval, venc)
        z = z_lq

        ref_names = _parse_refs(row.get("ref_image", "[]"))
        if getattr(args, "use_ref_feature", False) and len(ref_names) > 0:
            K = int(getattr(args, "ffhq_ref_num_refs", 1))
            ref_tensors = []
            for ref_name in ref_names[:K]:
                ref_path = ref_dir / str(ref_name)
                if ref_path.exists():
                    with Image.open(ref_path) as img:
                        ref_tensors.append(to_tensor(img.convert("RGB")))
            if len(ref_tensors) > 0:
                while len(ref_tensors) < K:
                    ref_tensors.append(ref_tensors[-1].clone())
                ref = torch.stack(ref_tensors, dim=0).unsqueeze(0).to(accelerator.device)
                ref = ref * 2.0 - 1.0
                ref_z = addon.feature_extractor.extract_ref(ref, venc)
                z = addon.feature_extractor.fuse_lq_ref(z_lq, ref_z)

        if args.ae_type == "sd2":
            lq_latent = model_ae.encode(lq_eval.to(model_ae.dtype)).latent_dist.sample()
            lq_latent = lq_latent * model_ae.config.scaling_factor
            sr_latent = vosr.sample_multistep_fm(eval_model, lq_latent, n_steps=args.infer_steps, venc_fea=z)
            sr_m11 = model_ae.decode((sr_latent / model_ae.config.scaling_factor).to(model_ae.dtype), return_dict=False)[0].clamp(-1, 1)
        else:
            raise NotImplementedError("Reference-aware eval currently supports sd2 ae_type")

        sr_01 = ((sr_m11 + 1.0) / 2.0).clamp(0, 1)
        if resize_to_512 is not None:
            sr_01 = resize_to_512(sr_01)
            gt_01 = resize_to_512(gt_01)
        to_pil(sr_01.squeeze(0).cpu()).save(sr_dir / f"{Path(gt_name).stem}.png")

        sr_metric = (sr_01 * 255).clamp(0, 255).byte().float() / 255.0
        if iqa_lpips is not None:
            lpips_sum += iqa_lpips(gt_01, sr_metric.clip(0, 1)).float().item()
        if iqa_musiq is not None:
            musiq_sum += iqa_musiq(sr_metric.clip(0, 1)).float().item()
        n_img += 1

    metrics = {}
    if n_img > 0:
        if iqa_lpips is not None:
            metrics["eval/lpips"] = lpips_sum / n_img
        if iqa_musiq is not None:
            metrics["eval/musiq"] = musiq_sum / n_img
        logger.info(f"[Ref Eval @ step {global_step}] n={n_img}, metrics={metrics}")
    else:
        logger.warning(f"[Ref Eval @ step {global_step}] No valid eval images found")
    return metrics
