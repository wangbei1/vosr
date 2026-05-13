import torch
import torch.nn as nn


class GatedRefFusion(nn.Module):
    """
    Fuse LQ feature and REF feature in token space.

    Input:
        lq_feat:  [B, N, C]
        ref_feat: [B, N, C]

    Output:
        fused:    [B, N, C]
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        self.norm_lq = nn.LayerNorm(dim)
        self.norm_ref = nn.LayerNorm(dim)

        self.ref_proj = nn.Linear(dim, dim)

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, lq_feat: torch.Tensor, ref_feat: torch.Tensor) -> torch.Tensor:
        if lq_feat.ndim != 3 or ref_feat.ndim != 3:
            raise ValueError(
                f"GatedRefFusion expects [B, N, C], got {lq_feat.shape} and {ref_feat.shape}"
            )

        if lq_feat.shape != ref_feat.shape:
            raise ValueError(
                f"Shape mismatch: lq_feat={lq_feat.shape}, ref_feat={ref_feat.shape}"
            )

        lq = self.norm_lq(lq_feat)
        ref = self.norm_ref(ref_feat)
        ref = self.ref_proj(ref)

        gate = self.gate(torch.cat([lq, ref], dim=-1))
        fused = lq_feat + gate * ref
        return fused