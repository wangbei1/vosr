"""Stage-D Ref cross-attention patch for VOSR LightningDiT.

This module monkey-patches an existing LightningDiT instance so the model can
consume a condition dict:

    z = {
        "lq":  List[Tensor[B, N, C_enc]] or Tensor[B, N, C_enc],
        "ref": List[Tensor[B, K, N, C_enc]] or Tensor[B, K, N, C_enc],
    }

The original VOSR semantic condition is still used through the original
cross-attention. Reference features are projected with the same DINO projector
(layer_norm + mlp_ca), then injected with an additional cross-attention layer in
each DiT block:

    x = x + cross_attn_lq(x, z_lq)
    x = x + ref_gate * cross_attn_ref(x, z_ref_tokens)

The added ref cross-attention is zero-safe: a learnable scalar gate controls its
strength. It does not modify checkpoint key compatibility except for new missing
keys when saving/loading Stage-D checkpoints.
"""
# ref_attention_patch.py is designed to be dependency-light and self-contained, so it can be easily copy-pasted into other codebases. It only depends on PyTorch and the original LightningDiT code.
from __future__ import annotations

import types
import math

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from models.lightningdit import MultiHeadCrossAttention, modulate_adasin

from models.lightningdit import modulate_adasin
from models.rmsnorm import RMSNorm


class CachedRefCrossAttention(nn.Module):
    """
    Cross-attention with optional precomputed K/V cache.

    Parameters are intentionally named like LightningDiT.MultiHeadCrossAttention:
    q_linear, k_linear, v_linear, proj, q_norm, k_norm

    This keeps checkpoint compatibility with previous ref_cross_attn modules.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        qk_norm=False,
        fused_attn=True,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

        # Init proj to zero so the new attention starts with zero contribution.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def build_kv(self, cond):
        """
        cond: [B,M,C]
        return k,v: [B,H,M,D]
        """
        B, M, C = cond.shape

        k = self.k_linear(cond)
        v = self.v_linear(cond)

        k = k.view(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        k = self.k_norm(k)

        return k, v

    def forward(self, x, cond=None, kv_cache=None):
        """
        x:        [B,N,C]
        cond:     [B,M,C], used when kv_cache is None
        kv_cache: tuple(k,v), both [B,H,M,D]
        """
        B, N, C = x.shape

        q = self.q_linear(x)
        q = q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)

        if kv_cache is None:
            if cond is None:
                raise ValueError("CachedRefCrossAttention requires cond or kv_cache")
            k, v = self.build_kv(cond)
        else:
            k, v = kv_cache

        if self.fused_attn:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


def _first_condition(z: Any) -> Optional[torch.Tensor]:
    if z is None:
        return None
    if isinstance(z, (list, tuple)):
        return z[0]
    return z


def _split_z_pack(z: Any) -> Tuple[Any, Any]:
    if isinstance(z, dict):
        return z.get("lq", None), z.get("ref", None)
    return z, None


def _pool_tokens(x: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """
    x: [B,N,C]
    return: [B,M,C], M <= target_tokens
    """
    if x is None:
        return None

    B, N, C = x.shape
    target_tokens = int(target_tokens)

    if target_tokens <= 0 or N <= target_tokens:
        return x

    x = x.transpose(1, 2)  # [B,C,N]
    x = F.adaptive_avg_pool1d(x, target_tokens)
    x = x.transpose(1, 2).contiguous()
    return x


def _as_ref_tensor(ref_z: Any, args=None) -> Optional[torch.Tensor]:
    """
    Convert ref condition to compact multi-ref tokens.

    Input:
        list first element or tensor
        [B,K,N,C] or [B,N,C]

    Output:
        [B,M,C]
        where M = K * ref_attn_tokens_per_ref, capped by ref_attn_max_total_tokens
    """
    if ref_z is None:
        return None
    if isinstance(ref_z, (list, tuple)):
        ref_z = ref_z[0]

    tokens_per_ref = int(getattr(args, "ref_attn_tokens_per_ref", 64)) if args is not None else 64
    max_total = int(getattr(args, "ref_attn_max_total_tokens", 256)) if args is not None else 256

    if ref_z.ndim == 4:
        B, K, N, C = ref_z.shape

        flat = ref_z.reshape(B * K, N, C)
        flat = _pool_tokens(flat, tokens_per_ref)

        M = flat.shape[1]
        ref_z = flat.reshape(B, K * M, C)

        if max_total > 0 and ref_z.shape[1] > max_total:
            ref_z = _pool_tokens(ref_z, max_total)

        return ref_z

    if ref_z.ndim == 3:
        if max_total > 0:
            ref_z = _pool_tokens(ref_z, max_total)
        return ref_z
    raise ValueError(f"ref_z must be [B,K,N,C] or [B,N,C], got {tuple(ref_z.shape)}")


def _split_z_pack(z: Any) -> Tuple[Any, Any]:
    if isinstance(z, dict):
        return z.get("lq", None), z.get("ref", None)
    return z, None


def _project_semantic(model: nn.Module, z: Any) -> Optional[torch.Tensor]:
    z = _first_condition(z)
    if z is None:
        return None
    z = model.layer_norm(z)
    z = model.mlp_ca(z)
    return z


def _project_ref(model: nn.Module, ref_z: Any) -> Optional[torch.Tensor]:
    ref_z = _as_ref_tensor(ref_z, getattr(model, "_ref_attention_args", None))
    if ref_z is None:
        return None
    ref_z = model.layer_norm(ref_z)
    ref_z = model.mlp_ca(ref_z)
    return ref_z


def _build_ref_kv_cache(blocks, ref_z):
    """
    Build per-block K/V cache.

    This is CacheKV at DiT-block level:
    ref_z is projected once into K/V for each block, then reused inside the block.
    """
    if ref_z is None:
        return None

    caches = []
    for block in blocks:
        if getattr(block, "use_ref_cross_attn", False):
            caches.append(block.ref_cross_attn.build_kv(ref_z))
        else:
            caches.append(None)
    return caches


def _make_block_forward(block):
    def forward_stage_e(self, x, c, z=None, ref_z=None, ref_kv=None, feat_rope=None):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + c.reshape(B, 6, -1)
        ).chunk(6, dim=1)

        x = x + gate_msa * self.attn(
            modulate_adasin(self.norm1(x), shift_msa, scale_msa),
            rope=feat_rope,
        )
        if self.z_dims is not None and z is not None:
            x = x + self.cross_attn(x, z)

        if getattr(self, "use_ref_cross_attn", False) and (ref_z is not None or ref_kv is not None):
            ref_out = self.ref_cross_attn(x, cond=ref_z, kv_cache=ref_kv)

            # Use sigmoid instead of tanh for better gradient flow
            # sigmoid(0.0) = 0.5, making reference attention active from start
            gate = torch.sigmoid(self.ref_attn_gate)
            scale = float(getattr(self, "ref_attn_rescale", 0.1))

            x = x + gate * scale * ref_out

        x = x + gate_mlp * self.mlp(
            modulate_adasin(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x

    return types.MethodType(forward_stage_e, block)


def patch_lightningdit_ref_attention(model: nn.Module, args=None) -> nn.Module:
    """Patch a LightningDiT instance in-place and return it.

    Call immediately after model creation and before optimizer creation:

        model = LightningDiT(...)
        if args.use_ref_attention:
            patch_lightningdit_ref_attention(model, args)
        optimizer = AdamW(model.parameters(), ...)
    """
    if getattr(model, "_ref_attention_patched", False):
        return model

    if getattr(model, "z_dims", None) is None:
        raise ValueError("Ref attention requires model.z_dims to be set.")

    init_gate = float(getattr(args, "ref_attn_init_gate", 0.0)) if args is not None else 0.0
    qk_norm = bool(getattr(args, "use_qknorm", False)) if args is not None else False
    ref_attn_scale = float(getattr(args, "ref_attn_scale", 0.1)) if args is not None else 0.1

    model._ref_attention_args = args

    for block in model.blocks:
        if not hasattr(block, "ref_cross_attn"):
            block.ref_cross_attn = CachedRefCrossAttention(
                d_model=model.hidden_size,
                num_heads=model.num_heads,
                qk_norm=qk_norm,
                fused_attn=True,
            )
        if not hasattr(block, "ref_attn_gate"):
            block.ref_attn_gate = nn.Parameter(torch.tensor(init_gate, dtype=torch.float32))
        block.use_ref_cross_attn = True
        block.ref_attn_rescale = ref_attn_scale
        block.forward = _make_block_forward(block)

    def forward_stage_e(self, x, t, r=None, z=None):
        use_checkpoint = self.use_checkpoint
        x = self.x_embedder(x)
        t_raw = t
        t_emb = self.t_embedder(t)

        if self.r_embedder is not None:
            r_emb = self.r_embedder(r) * (t_raw - r).unsqueeze(-1)
        else:
            r_emb = 0

        c = t_emb + r_emb
        c0 = self.t_block(c)

        lq_z, ref_z = _split_z_pack(z)
        if self.z_dims is not None:
            lq_z = _project_semantic(self, lq_z)
            ref_z = _project_ref(self, ref_z)
        else:
            lq_z = None
            ref_z = None

        ref_kv_cache = _build_ref_kv_cache(self.blocks, ref_z)

        for i, block in enumerate(self.blocks):
            ref_kv = None if ref_kv_cache is None else ref_kv_cache[i]

            if use_checkpoint:
                x = checkpoint(
                    block,
                    x,
                    c0,
                    lq_z,
                    ref_z,
                    ref_kv,
                    self.feat_rope,
                    use_reentrant=True,
                )
            else:
                x = block(x, c0, lq_z, ref_z, ref_kv, self.feat_rope)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

    model.forward = types.MethodType(forward_stage_e, model)
    model._ref_attention_patched = True
    return model
