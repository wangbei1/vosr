import torch


class RefLatentEncoder:
    """VAE latent encoder/decoder helper for VOSR + Ref-LDM experiments."""

    def __init__(self, args):
        self.args = args

    @torch.no_grad()
    def encode_lq_hq(self, model_ae, lq_m11, hq_m11, latents_mean=None, latents_std=None):
        combined = torch.cat([lq_m11, hq_m11], dim=0)
        if self.args.ae_type == "qwen":
            if latents_mean is None or latents_std is None:
                raise ValueError("qwen VAE requires latents_mean and latents_std")
            latent = (model_ae.encode(combined).latent_dist.sample() - latents_mean) * latents_std
        elif self.args.ae_type == "sd2":
            latent = model_ae.encode(combined.to(model_ae.dtype)).latent_dist.sample()
            latent = latent * model_ae.config.scaling_factor
        else:
            raise ValueError(f"Unsupported ae_type: {self.args.ae_type}")
        return latent.chunk(2, dim=0)

    @torch.no_grad()
    def encode_ref(self, model_ae, ref_m11, latents_mean=None, latents_std=None):
        """Encode [B,K,3,H,W] reference images into [B,K,C,h,w] latents."""
        if ref_m11 is None:
            return None
        if ref_m11.ndim != 5:
            raise ValueError(f"ref_m11 must be [B,K,C,H,W], got {tuple(ref_m11.shape)}")
        B, K, C, H, W = ref_m11.shape
        ref_flat = ref_m11.reshape(B * K, C, H, W)
        chunk = int(getattr(self.args, "ref_vae_encode_chunk", 1))
        outs = []
        for start in range(0, ref_flat.shape[0], chunk):
            cur = ref_flat[start:start + chunk]
            if self.args.ae_type == "qwen":
                if latents_mean is None or latents_std is None:
                    raise ValueError("qwen VAE requires latents_mean and latents_std")
                latent = (model_ae.encode(cur).latent_dist.sample() - latents_mean) * latents_std
            elif self.args.ae_type == "sd2":
                latent = model_ae.encode(cur.to(model_ae.dtype)).latent_dist.sample()
                latent = latent * model_ae.config.scaling_factor
            else:
                raise ValueError(f"Unsupported ae_type: {self.args.ae_type}")
            outs.append(latent)
        ref_latent = torch.cat(outs, dim=0)
        return ref_latent.reshape(B, K, *ref_latent.shape[1:])

    # @torch.no_grad()
    def decode(self, model_ae, latent):
        if self.args.ae_type == "sd2":
            x = latent / model_ae.config.scaling_factor
            return model_ae.decode(x.to(model_ae.dtype), return_dict=False)[0].clamp(-1, 1)
        raise NotImplementedError("decode() is currently implemented for sd2 only")
