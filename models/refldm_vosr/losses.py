import torch
import torch.nn as nn


class SimpleIdentityLoss(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        model_path = getattr(args, "identity_model_path", None)
        if model_path is None:
            raise ValueError("identity_model_path must be set when use_identity_loss=true")
        try:
            from .identity_loss import IdentityLoss
        except Exception:
            from identity_loss import IdentityLoss
        self.loss = IdentityLoss(model_path=model_path).to(device).eval()
        for p in self.loss.parameters():
            p.requires_grad_(False)

    def forward(self, sr_m11, target_m11):
        out = self.loss(sr_m11, target_m11)
        if isinstance(out, tuple):
            out = out[0]
        if out.ndim == 0:
            return out
        return out.mean()


class TimestepScaledIdentityLoss(nn.Module):
    def __init__(self, args, device, latent_codec):
        super().__init__()
        self.args = args
        self.latent_codec = latent_codec
        self.identity = SimpleIdentityLoss(args, device)

    def forward(self, model_ae, hq_pred_latent, hq_gt_latent, t):
        sr_m11 = self.latent_codec.decode(model_ae, hq_pred_latent)
        gt_m11 = self.latent_codec.decode(model_ae, hq_gt_latent)
        id_loss = self.identity(sr_m11.float(), gt_m11.float())
        gamma = float(getattr(self.args, "identity_timestep_gamma", 0.5))
        scale = torch.clamp(1.0 - t.float(), min=0.0).pow(gamma).mean()
        return id_loss * scale


@torch.no_grad()
def cosine_identity_score(identity_model, sr_m11, ref_m11):
    return None
