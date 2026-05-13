# Alternative manual patch: paste this method inside class VOSR in vosr.py.
# The generated train_vosr_refldm_modular.py uses a monkey-patch instead, so
# manual editing is optional.

import torch
from einops import rearrange


def loss_fm_return_extra(self, model, lq, hq, z=None, weight_dtype=None):
    B, device = hq.size(0), hq.device
    if self.time_dist[0] == "uniform":
        t = torch.rand(B, device=device)
    elif self.time_dist[0] == "lognorm":
        mu, sigma = self.time_dist[-2], self.time_dist[-1]
        rnd_normal = torch.randn(B, device=device)
        t = torch.sigmoid(rnd_normal * sigma + mu)
    else:
        raise ValueError(f"Unsupported time_dist: {self.time_dist}")
    t_ = rearrange(t, "b -> b 1 1 1")
    cond_strength_aelq = torch.sigmoid(
        torch.randn(B, device=device) * self.args.cond_strength_aelq_list[1]
        + self.args.cond_strength_aelq_list[0]
    )
    cond_strength_aelq = rearrange(cond_strength_aelq, "b -> b 1 1 1")
    eps = torch.randn_like(lq)
    _, _, _, lq_mixed, _, _, z_mixed = self._prepare_cfg_conditions(B, device, lq, z, cond_strength_aelq)
    z_t = (1.0 - t_) * hq + t_ * eps
    inp = torch.cat([lq_mixed, z_t], dim=1)
    v_target = eps - hq
    v_pred = model(inp, t, z=z_mixed)
    loss = ((v_pred - v_target) ** 2).mean()
    hq_pred = z_t - t_ * v_pred
    extra = {"t": t, "z_t": z_t, "v_pred": v_pred, "v_target": v_target, "hq_pred": hq_pred}
    return loss, loss, extra
