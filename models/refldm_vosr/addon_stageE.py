"""Stage-E addon: Ref cross-attention + Ref-target identity loss."""

from __future__ import annotations

import torch

from .features import RefFeatureExtractor
from .latents import RefLatentEncoder
from .losses_stageE import TimestepScaledIdentityLoss


class RefLDMVOSRStageEAddon:
    def __init__(self, args, accelerator, preprocess_raw_image=None):
        self.args = args
        self.accelerator = accelerator
        self.use_refldm = bool(getattr(args, "use_refldm", False))
        self.use_ref_feature = bool(getattr(args, "use_ref_feature", False))
        self.use_ref_attention = bool(getattr(args, "use_ref_attention", False))
        self.use_ref_latent = bool(getattr(args, "use_ref_latent", False))
        self.use_identity_loss = bool(getattr(args, "use_identity_loss", False))

        self.feature_extractor = RefFeatureExtractor(args, preprocess_raw_image=preprocess_raw_image)
        self.latent_encoder = RefLatentEncoder(args)
        self.id_loss = None
        if self.use_identity_loss:
            self.id_loss = TimestepScaledIdentityLoss(args, accelerator.device, self.latent_encoder)

    def _get_ref_m11(self, batch):
        ref = batch.get("ref", None)
        if ref is None:
            return None
        ref = ref.to(self.accelerator.device, non_blocking=True)
        if ref.ndim == 4:
            ref = ref.unsqueeze(1)

        if ref.ndim != 5:
            raise ValueError(f"batch['ref'] must be [B,K,3,H,W] or [B,3,H,W], got {tuple(ref.shape)}")

        return ref * 2.0 - 1.0

    @torch.no_grad()
    def prepare_train_pack(self, batch, hq_img_m11, lq_img_m11, lq_feature, model_ae, venc, latents_mean=None, latents_std=None):
        lq_latent, hq_latent = self.latent_encoder.encode_lq_hq(
            model_ae, lq_img_m11, hq_img_m11, latents_mean=latents_mean, latents_std=latents_std
        )

        ref_m11 = self._get_ref_m11(batch) if self.use_refldm else None
        ref_z = None
        ref_latent = None

        if ref_m11 is not None and (self.use_ref_feature or self.use_ref_attention):
            ref_z = self.feature_extractor.extract_ref(ref_m11, venc)

        if ref_m11 is not None and self.use_ref_latent:
            ref_latent = self.latent_encoder.encode_ref(model_ae, ref_m11, latents_mean=latents_mean, latents_std=latents_std)

        if self.use_ref_attention:
            z = {"lq": lq_feature, "ref": ref_z}
        elif self.use_ref_feature and ref_z is not None:
            z = self.feature_extractor.fuse_lq_ref(lq_feature, ref_z)
        else:
            z = lq_feature

        return {
            "lq": lq_latent,
            "hq": hq_latent,
            "z": z,
            "z_lq": lq_feature,
            "ref_img": ref_m11,
            "ref_z": ref_z,
            "ref_latent": ref_latent,
        }

    def _identity_enabled_now(self, global_step):
        use = self.use_identity_loss
        start = int(getattr(self.args, "identity_start_step", 0))
        every = int(getattr(self.args, "identity_every_n_steps", 1))
        if global_step is not None:
            use = use and global_step >= start and (global_step % max(every, 1) == 0)
        return use

    def compute_loss(self, vosr, model, model_ae, ref_pack, global_step=None):
        use_stageE = self.use_ref_attention and isinstance(ref_pack.get("z"), dict)
        use_id_now = self._identity_enabled_now(global_step)

        if use_id_now:
            if use_stageE:
                loss_fm, _, extra = vosr.loss_fm_return_extra_stageD(
                    model,
                    ref_pack["lq"],
                    ref_pack["hq"],
                    ref_pack["z"],
                )
            else:
                loss_fm, _, extra = vosr.loss_fm_return_extra(
                    model,
                    ref_pack["lq"],
                    ref_pack["hq"],
                    ref_pack["z"],
                )

            id_losses = self.id_loss(
                model_ae=model_ae,
                hq_pred_latent=extra["hq_pred"],
                hq_gt_latent=ref_pack["hq"],
                t=extra["t"],
                ref_img_m11=ref_pack.get("ref_img", None),
            )

            identity_weight = float(getattr(self.args, "identity_loss_weight", 0.0005))
            total = loss_fm + identity_weight * id_losses["loss_id"]

            log = {
                "loss_fm": loss_fm.detach(),
                "loss_total": total.detach(),
            }

            for k, v in id_losses.items():
                if torch.is_tensor(v):
                    log[k] = v.detach()

            return total, total, log

        if use_stageE:
            loss_fm, loss_backward = vosr.loss_fm_stageD(
                model,
                ref_pack["lq"],
                ref_pack["hq"],
                ref_pack["z"],
            )
        else:
            loss_fm, loss_backward = vosr.loss_fm(
                model,
                ref_pack["lq"],
                ref_pack["hq"],
                ref_pack["z"],
            )

        return loss_fm, loss_backward, {"loss_fm": loss_fm.detach()}
