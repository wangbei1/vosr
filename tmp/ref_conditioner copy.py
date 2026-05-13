import torch
import torch.nn as nn
from .ref_fusion import GatedRefFusion


class RefConditioner(nn.Module):
    """
    Fuse a list of visual token features from LQ and REF.

    Example:
        z_lq  = [feat_0, feat_1, feat_2]
        z_ref = [feat_0, feat_1, feat_2]
        z_out = conditioner(z_lq, z_ref)

    Each feat should be [B, N, C].
    """
    def __init__(self, num_layers: int, dim: int, use_residual: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.use_residual = use_residual

        self.fusions = nn.ModuleList([
            GatedRefFusion(dim) for _ in range(num_layers)
        ])

    def forward(self, z_lq, z_ref):
        if not isinstance(z_lq, (list, tuple)) or not isinstance(z_ref, (list, tuple)):
            raise TypeError("z_lq and z_ref must be list/tuple of tensors")

        if len(z_lq) != len(z_ref):
            raise ValueError(f"len mismatch: {len(z_lq)} vs {len(z_ref)}")

        if len(z_lq) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layers, got {len(z_lq)}"
            )

        z_out = []
        for i, (feat_lq, feat_ref) in enumerate(zip(z_lq, z_ref)):
            if feat_lq.shape[-1] != self.dim or feat_ref.shape[-1] != self.dim:
                raise ValueError(
                    f"Layer {i}: expected last dim={self.dim}, "
                    f"got {feat_lq.shape} and {feat_ref.shape}"
                )

            fused = self.fusions[i](feat_lq, feat_ref)
            if self.use_residual:
                fused = fused + feat_lq
            z_out.append(fused)

        return z_out