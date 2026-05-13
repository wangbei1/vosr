"""Stage-E identity losses for VOSR + FFHQ-Ref.

Adds Ref-target identity supervision:
  - identity_target: gt | ref | both
  - ref target supports [B,K,3,H,W] and averages identity loss over K refs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleIdentityLoss(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        self.enabled = True
        model_path = getattr(args, "identity_model_path", None)

        try:
            if model_path is None:
                raise ValueError("identity_model_path not set")

            from .identity_loss import IdentityLoss

            self.loss = IdentityLoss(model_path=model_path).to(device).eval()
            for p in self.loss.parameters():
                p.requires_grad_(False)

            print("[IdentityLoss]  Loaded successfully")

        except Exception as e:
            print(f"[IdentityLoss]  Disabled due to: {e}")
            self.enabled = False
            self.loss = None

    def forward(self, sr_m11, target_m11):
        if not self.enabled:
            return torch.tensor(0.0, device=sr_m11.device)

        out = self.loss(sr_m11.float(), target_m11.float())

        if isinstance(out, tuple):
            out = out[0]

        return out.mean() if out.ndim > 0 else out
    
class TimestepScaledIdentityLoss(nn.Module):
    def __init__(self, args, device, latent_codec):
        super().__init__()
        self.args = args
        self.latent_codec = latent_codec
        self.identity = SimpleIdentityLoss(args, device)

    def _scale(self, t):
        gamma = float(getattr(self.args, "identity_timestep_gamma", 0.5))
        return torch.clamp(1.0 - t.float(), min=0.0).pow(gamma).mean()

    def _loss_to_ref_images(self, sr_m11, ref_img_m11):
        """sr: [B,3,H,W], ref: [B,K,3,H,W] or [B,3,H,W]."""
        if ref_img_m11 is None:
            raise ValueError("identity_target='ref' requires ref_img_m11")
        if ref_img_m11.ndim == 4:
            return self.identity(sr_m11, ref_img_m11)
        if ref_img_m11.ndim != 5:
            raise ValueError(f"ref_img_m11 must be [B,K,3,H,W] or [B,3,H,W], got {tuple(ref_img_m11.shape)}")
        b, k, c, h, w = ref_img_m11.shape
        # sr_rep = sr_m11[:, None].expand(b, k,  c, h, w).reshape(b * k, c, h, w)
        sr_rep = sr_m11[:, None].expand(b, k, c, sr_m11.shape[-2], sr_m11.shape[-1])
        sr_rep = sr_rep.reshape(b * k, c, sr_m11.shape[-2], sr_m11.shape[-1])
        ref_flat = ref_img_m11.reshape(b * k, c, h, w)
        return self.identity(sr_rep, ref_flat)

    def forward(self, model_ae, hq_pred_latent, hq_gt_latent, t, ref_img_m11=None):
        """
        hq_pred_latent must keep gradient.
        GT/ref targets do not need gradients.
        """
        sr_m11 = self.latent_codec.decode(model_ae, hq_pred_latent)
        target = str(getattr(self.args, "identity_target", "gt")).lower()
        scale = self._scale(t)

        loss_dict = {}
        if target in {"gt", "both"}:
            with torch.no_grad():
                gt_m11 = self.latent_codec.decode(model_ae, hq_gt_latent.detach())
            # gt_m11 = self.latent_codec.decode(model_ae, hq_gt_latent)
            loss_dict["loss_id_gt"] = self.identity(sr_m11, gt_m11)
        if target in {"ref", "both"}:
            loss_dict["loss_id_ref"] = self._loss_to_ref_images(sr_m11, ref_img_m11)
        if not loss_dict:
            raise ValueError(f"Unsupported identity_target={target!r}; use gt/ref/both")

        if target == "both":
            ref_w = float(getattr(self.args, "identity_ref_weight", 1.0))
            gt_w = float(getattr(self.args, "identity_gt_weight", 1.0))
            denom = max(ref_w + gt_w, 1e-8)
            id_loss = (gt_w * loss_dict["loss_id_gt"] + ref_w * loss_dict["loss_id_ref"]) / denom
        elif target == "ref":
            id_loss = loss_dict["loss_id_ref"]
        else:
            id_loss = loss_dict["loss_id_gt"]

        loss_dict["loss_id_raw"] = id_loss
        loss_dict["identity_scale"] = scale.detach()


        loss_dict["loss_id"] = id_loss * scale
        return loss_dict
