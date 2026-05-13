import torch

from .features import RefFeatureExtractor
from .latents import RefLatentEncoder
from .losses import TimestepScaledIdentityLoss


class RefLDMVOSRAddon:
    """Modular controller for integrating FFHQ-Ref / Ref-LDM ideas into VOSR.

    Stage-A: use original VOSR condition + timestep-scaled identity loss.
    Stage-B: optionally fuse DINO reference features into VOSR's z condition.
    Ref latent encoding is provided for later extensions, but is disabled by default.
    """

    def __init__(self, args, accelerator, preprocess_raw_image=None):
        self.args = args
        self.accelerator = accelerator
        self.use_refldm = bool(getattr(args, "use_refldm", False))
        self.use_ref_feature = bool(getattr(args, "use_ref_feature", False))
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
        # dataset returns [0, 1], convert to [-1, 1]
        return ref * 2.0 - 1.0

    @torch.no_grad()
    def prepare_train_pack(
        self,
        batch,
        hq_img_m11,
        lq_img_m11,
        lq_feature,
        model_ae,
        venc,
        latents_mean=None,
        latents_std=None,
    ):
        lq_latent, hq_latent = self.latent_encoder.encode_lq_hq(
            model_ae, lq_img_m11, hq_img_m11, latents_mean=latents_mean, latents_std=latents_std
        )

        ref_m11 = self._get_ref_m11(batch) if self.use_refldm else None
        ref_z = None
        ref_latent = None
        z = lq_feature

        if ref_m11 is not None and self.use_ref_feature:
            ref_z = self.feature_extractor.extract_ref(ref_m11, venc)
            z = self.feature_extractor.fuse_lq_ref(lq_feature, ref_z)

        if ref_m11 is not None and self.use_ref_latent:
            ref_latent = self.latent_encoder.encode_ref(
                model_ae, ref_m11, latents_mean=latents_mean, latents_std=latents_std
            )

        return {
            "lq": lq_latent,
            "hq": hq_latent,
            "z": z,
            "ref_img": ref_m11,
            "ref_z": ref_z,
            "ref_latent": ref_latent,
        }

    def compute_loss(self, vosr, model, model_ae, ref_pack):
        if self.use_identity_loss:
            loss_fm, loss_backward, extra = vosr.loss_fm_return_extra(
                model, ref_pack["lq"], ref_pack["hq"], ref_pack["z"]
            )
            id_loss = self.id_loss(
                model_ae=model_ae,
                hq_pred_latent=extra["hq_pred"],
                hq_gt_latent=ref_pack["hq"],
                t=extra["t"],
            )
            weight = float(getattr(self.args, "identity_loss_weight", 0.0001))
            total = loss_fm + weight * id_loss
            return total, total, {
                "loss_fm": loss_fm.detach(),
                "loss_id": id_loss.detach(),
            }

        loss_fm, loss_backward = vosr.loss_fm(
            model, ref_pack["lq"], ref_pack["hq"], ref_pack["z"]
        )
        return loss_fm, loss_backward, {"loss_fm": loss_fm.detach()}
