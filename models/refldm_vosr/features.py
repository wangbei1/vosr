import torch
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD




class RefFeatureExtractor:
    """DINOv2 feature extraction and lightweight LQ/Ref fusion for VOSR.

    Input image tensors are expected in [-1, 1].

    LQ output:
        list of [B,N,C]

    Ref output:
        list of [B,K,N,C]
    """

    def __init__(self, args, preprocess_raw_image=None):
        self.args = args
        self._external_preprocess = preprocess_raw_image

    def _preprocess(self, x_255):
        if self._external_preprocess is not None:
            return self._external_preprocess(x_255, self.args)
        x = x_255 / 255.0
        x = torch.nn.functional.interpolate(
            x, self.args.dinov2_size, mode="bicubic", align_corners=False
        ).clamp(0.0, 1.0)
        return Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)

    # 保留一个别名（兼容旧代码）
    def preprocess_raw_image(self, x_255):
        return self._preprocess(x_255)

    def _select_features(self, features, x_norm):
        selected = []
        for i in self.args.layer_dinov2b_list:
            if i == -1:
                selected.append(x_norm)
            else:
                selected.append(features[f"layer_{i}"])
        return selected

    @torch.no_grad()
    def extract(self, image_m11, venc):
        raw_image = (0.5 * image_m11 + 0.5) * 255.0
        raw_image = self._preprocess(raw_image)
        features, x_norm = venc.forward_with_features(raw_image)
        return self._select_features(features, x_norm)

    @torch.no_grad()
    def extract_ref(self, ref_m11, venc):
        """
        ref_m11:
            [B,K,C,H,W] preferred
            [B,C,H,W] accepted as K=1
        """
        if ref_m11 is None:
            return None

        if ref_m11.ndim == 4:
            ref_m11 = ref_m11.unsqueeze(1)

        if ref_m11.ndim != 5:
            raise ValueError(f"ref_m11 must be [B,K,C,H,W], got {tuple(ref_m11.shape)}")

        B, K, C, H, W = ref_m11.shape
        ref_flat = ref_m11.reshape(B * K, C, H, W)

        chunk = int(getattr(self.args, "ref_dino_forward_chunk", 8))
        outs = None

        for start in range(0, ref_flat.shape[0], chunk):
            cur = ref_flat[start:start + chunk]
            raw_image = (0.5 * cur + 0.5) * 255.0
            raw_image = self._preprocess(raw_image)

            features, x_norm = venc.forward_with_features(raw_image)
            z_cur = self._select_features(features, x_norm)

            if outs is None:
                outs = [[] for _ in range(len(z_cur))]

            for i, feat in enumerate(z_cur):
                outs[i].append(feat)

        z = [torch.cat(parts, dim=0) for parts in outs]

        ref_z = []
        for feat in z:
            # [B*K,N,C] -> [B,K,N,C]
            N, C_feat = feat.shape[1], feat.shape[2]
            ref_z.append(feat.reshape(B, K, N, C_feat))

        return ref_z

    def fuse_lq_ref(self, z_lq, ref_z):
        """Fuse LQ and reference DINO features without changing LightningDiT.

        This is intentionally conservative: ref_z is averaged over K references
        and added to the original LQ condition. Shape must match because both
        LQ and Ref are processed by the same DINO pipeline.
        """
        if ref_z is None or not getattr(self.args, "use_ref_feature", False):
            return z_lq
        weight = float(getattr(self.args, "ref_feature_weight", 0.05))
        fused = []
        for lq_feat, ref_feat in zip(z_lq, ref_z):
            ref_mean = ref_feat.mean(dim=1)  # [B,N,C]

            if ref_mean.shape != lq_feat.shape:
                raise ValueError(
                    "ref feature and lq feature shape mismatch: "
                    f"ref={tuple(ref_mean.shape)}, lq={tuple(lq_feat.shape)}"
                )
            fused.append(lq_feat + weight * ref_mean)
        return fused
