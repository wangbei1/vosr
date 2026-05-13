"""Stage-D VOSR partial-conditioning losses and ref-aware sampling.

The original VOSR already uses restoration-oriented partial conditioning: for a
random subset of samples, LQ structure is weakened and semantic condition is
zeroed. This patch extends that behavior to condition dictionaries:

    z = {"lq": z_lq, "ref": ref_z}

Full branch:
    v_cond = model([lq_noised, z_t], t, z={"lq": z_lq, "ref": ref_z})
Partial branch:
    v_pcond = model([lq_weak, z_t], t, z={"lq": 0, "ref": 0})

Training randomly mixes full/partial samples as in VOSR. Sampling performs:
    v_cfg = v_pcond + s * (v_cond - v_pcond)
"""

from __future__ import annotations

import random
from typing import Any

import torch
from einops import rearrange


def _clone_like(x: Any):
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: _clone_like(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [v.clone() for v in x]
    return x.clone()


def _zero_like(x: Any):
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: _zero_like(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [torch.zeros_like(v) for v in x]
    return torch.zeros_like(x)


def _set_indices(dst: Any, idx, src: Any):
    if dst is None or src is None:
        return
    if isinstance(dst, dict):
        for k in dst.keys():
            _set_indices(dst[k], idx, src[k])
    elif isinstance(dst, (list, tuple)):
        for i in range(len(dst)):
            dst[i][idx] = src[i][idx]
    else:
        dst[idx] = src[idx]


def _scale_first_dim(x: Any, scale):
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: _scale_first_dim(v, scale) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_scale_first_dim(v, scale) for v in x]
    while scale.ndim < x.ndim:
        scale = scale.unsqueeze(-1)
    return x * scale


def patch_vosr_stageD(VOSRClass):
    if getattr(VOSRClass, "_stageD_patched", False):
        return

    def _prepare_cfg_conditions_ref(self, B, device, lq, z_pack, cond_strength_aelq):
        cfg_mask = torch.rand(B, device=device) < self.cfg_ratio
        cfg_indices = (cfg_mask > 0).nonzero(as_tuple=True)[0]

        weak_alpha = random.uniform(
            self.args.weak_cond_strength_aelq_list[0],
            self.args.weak_cond_strength_aelq_list[1],
        )
        lq_weak = self.interpolate(lq, torch.zeros_like(lq), weak_alpha, self.interp_type)
        lq_noised = self.interpolate(lq, torch.randn_like(lq), cond_strength_aelq, interp_type="sph")
        lq_mixed = lq_noised.clone()
        lq_mixed[cfg_indices] = lq_weak[cfg_indices]

        z_full = _clone_like(z_pack)
        z_weak = _zero_like(z_pack)
        z_mixed = _clone_like(z_pack)
        _set_indices(z_mixed, cfg_indices, z_weak)
        return cfg_indices, lq_weak, lq_noised, lq_mixed, z_weak, z_full, z_mixed

    def loss_fm_stageD(self, model, lq, hq, z=None, weight_dtype=None):
        loss, loss_backward, _ = self.loss_fm_return_extra_stageD(model, lq, hq, z, weight_dtype)
        return loss, loss_backward

    def loss_fm_return_extra_stageD(self, model, lq, hq, z=None, weight_dtype=None):
        B, device = hq.size(0), hq.device
        if self.time_dist[0] == "uniform":
            t = torch.rand(B, device=device)
        elif self.time_dist[0] == "lognorm":
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            t = torch.sigmoid(torch.randn(B, device=device) * sigma + mu)
        else:
            raise ValueError(f"Unsupported time_dist: {self.time_dist}")

        t_ = rearrange(t, "b -> b 1 1 1")
        cond_strength_aelq = torch.sigmoid(
            torch.randn(B, device=device) * self.args.cond_strength_aelq_list[1]
            + self.args.cond_strength_aelq_list[0]
        )
        cond_strength_aelq = rearrange(cond_strength_aelq, "b -> b 1 1 1")

        eps = torch.randn_like(lq)
        _, lq_weak, lq_noised, lq_mixed, z_weak, z_full, z_mixed = self._prepare_cfg_conditions_ref(
            B, device, lq, z, cond_strength_aelq
        )
        z_t = (1.0 - t_) * hq + t_ * eps
        inp = torch.cat([lq_mixed, z_t], dim=1)
        v_target = eps - hq
        v_pred = model(inp, t, z=z_mixed)
        loss = ((v_pred - v_target) ** 2).mean()
        hq_pred = z_t - t_ * v_pred
        extra = {
            "t": t,
            "t_view": t_,
            "z_t": z_t,
            "v_pred": v_pred,
            "v_target": v_target,
            "hq_pred": hq_pred,
            "z_full": z_full,
            "z_weak": z_weak,
            "lq_noised": lq_noised,
            "lq_weak": lq_weak,
        }
        return loss, loss, extra

    @torch.no_grad()
    def sample_multistep_fm_ref(self, model, lq, n_steps=25, venc_fea=None, ref_fea=None):
        """Ref-aware multi-step FM sampler with VOSR restoration-oriented CFG."""
        z_pack_full = {"lq": venc_fea, "ref": ref_fea}
        z_pack_weak = _zero_like(z_pack_full)

        B, device = lq.size(0), lq.device
        x = torch.randn_like(lq)
        timesteps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

        weak_alpha = float(getattr(self.args, "sample_weak_cond_strength_aelq", 0.1))
        lq_weak = self.interpolate(lq, torch.zeros_like(lq), weak_alpha, self.interp_type)

        for i in range(n_steps):
            t_cur = timesteps[i].repeat(B)
            t_next = timesteps[i + 1].repeat(B)
            dt = (t_cur - t_next).view(B, 1, 1, 1)

            inp_cond = torch.cat([lq, x], dim=1)
            inp_weak = torch.cat([lq_weak, x], dim=1)
            v_cond = model(inp_cond, t_cur, z=z_pack_full)
            v_weak = model(inp_weak, t_cur, z=z_pack_weak)
            v = v_weak + self.cfg_scale * (v_cond - v_weak)
            x = x - dt * v
        return x

    VOSRClass._prepare_cfg_conditions_ref = _prepare_cfg_conditions_ref
    VOSRClass.loss_fm_stageD = loss_fm_stageD
    VOSRClass.loss_fm_return_extra_stageD = loss_fm_return_extra_stageD
    VOSRClass.sample_multistep_fm_ref = sample_multistep_fm_ref
    VOSRClass._stageD_patched = True
