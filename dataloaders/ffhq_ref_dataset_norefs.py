import os
import ast
import random
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CenterCropSquare:
    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, pil_image: Image.Image):
        if min(*pil_image.size) < self.image_size:
            scale = self.image_size / min(*pil_image.size)
            new_size = tuple(round(x * scale) for x in pil_image.size)
            pil_image = pil_image.resize(new_size, resample=Image.Resampling.BICUBIC)

        crop_y = (pil_image.height - self.image_size) // 2
        crop_x = (pil_image.width - self.image_size) // 2
        pil_image = pil_image.crop((
            crop_x, crop_y,
            crop_x + self.image_size, crop_y + self.image_size
        ))
        return pil_image


class RandomCropSquare:
    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, pil_image: Image.Image):
        if min(*pil_image.size) < self.image_size:
            scale = self.image_size / min(*pil_image.size)
            new_size = tuple(round(x * scale) for x in pil_image.size)
            pil_image = pil_image.resize(new_size, resample=Image.Resampling.BICUBIC)

        max_x = pil_image.width - self.image_size
        max_y = pil_image.height - self.image_size
        crop_x = random.randint(0, max_x)
        crop_y = random.randint(0, max_y)

        pil_image = pil_image.crop((
            crop_x, crop_y,
            crop_x + self.image_size, crop_y + self.image_size
        ))
        return pil_image


class FFHQRefDataset(Dataset):
    """
    Dataset adapter for Ref-LDM FFHQ_Ref CSV files.

    Current training-stage goal:
    - return `hq` so VOSR can train without changing the core loss loop
    - optionally also return `ref` / metadata for later reference-aware extension

    Expected CSV columns:
    - train: gt_image, ref_image
    - val/test: gt_image, lq_image, ref_image
    """

    def __init__(self, split="train", args=None):
        super().__init__()
        self.args = args
        self.split = split
        self.max_retry = 10000

        # required
        self.hq_root = args.ffhq_ref_hq_dir

        # split-specific csv
        if split == "train":
            self.csv_path = args.ffhq_ref_train_csv
        elif split == "val":
            self.csv_path = args.ffhq_ref_val_csv
        elif split == "test":
            self.csv_path = args.ffhq_ref_test_csv
        else:
            raise ValueError(f"Unsupported split: {split}")

        # optional
        self.lq_root = getattr(args, "ffhq_ref_lq_dir", None)
        self.num_refs = getattr(args, "ffhq_ref_num_refs", 5)
        self.return_ref = getattr(args, "ffhq_ref_return_ref", True)

        # crop strategy
        crop_mode = getattr(args, "ffhq_ref_crop_mode", "random" if split == "train" else "center")
        if crop_mode == "random":
            self.preproc = RandomCropSquare(args.resolution)
        elif crop_mode == "center":
            self.preproc = CenterCropSquare(args.resolution)
        else:
            raise ValueError(f"Unsupported ffhq_ref_crop_mode: {crop_mode}")

        self.to_tensor = transforms.ToTensor()
        self.df = pd.read_csv(self.csv_path)

        # sanity
        required_cols = {"gt_image", "ref_image"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"{self.csv_path} missing required columns: {missing}")

        print(f"=====> FFHQRefDataset[{split}] loaded {len(self.df)} samples from {self.csv_path}")

    def __len__(self):
        return len(self.df)

    def _load_rgb(self, path):
        with Image.open(path) as im:
            im.load()
            return im.convert("RGB")

    def _parse_ref_list(self, ref_value):
        if isinstance(ref_value, list):
            refs = ref_value
        elif isinstance(ref_value, str):
            refs = ast.literal_eval(ref_value)
        else:
            refs = []

        if not isinstance(refs, list):
            refs = []

        refs = [str(x) for x in refs if str(x).strip() != ""]
        return refs

    def _sample_or_pad_refs(self, ref_names):
        if len(ref_names) == 0:
            return []

        if len(ref_names) >= self.num_refs:
            return random.sample(ref_names, self.num_refs)

        out = list(ref_names)
        while len(out) < self.num_refs:
            out.append(random.choice(ref_names))
        return out

    def __getitem__(self, idx):
        for retry in range(self.max_retry):
            try:
                current_idx = (idx + retry) % len(self.df)
                row = self.df.iloc[current_idx]

                gt_name = row["gt_image"]
                gt_path = os.path.join(self.hq_root, gt_name)
                gt_img = self._load_rgb(gt_path)
                gt_img = self.preproc(gt_img)
                hq_tensor = self.to_tensor(gt_img)

                sample = {
                    "hq": hq_tensor,
                    "gt_path": gt_path,
                    "gt_name": gt_name,
                }

                # Optional: keep ref images for future reference-aware model extension
                if self.return_ref:
                    ref_names = self._parse_ref_list(row["ref_image"])
                    ref_names = self._sample_or_pad_refs(ref_names)

                    ref_tensors = []
                    ref_paths = []
                    for ref_name in ref_names:
                        ref_path = os.path.join(self.hq_root, ref_name)
                        ref_img = self._load_rgb(ref_path)
                        ref_img = self.preproc(ref_img)
                        ref_tensors.append(self.to_tensor(ref_img))
                        ref_paths.append(ref_path)

                    if len(ref_tensors) > 0:
                        sample["ref"] = torch.stack(ref_tensors, dim=0)  # [K, C, H, W]
                    else:
                        sample["ref"] = torch.empty(0)

                    sample["ref_paths"] = ref_paths
                    sample["ref_names"] = ref_names

                # Optional: val/test CSV may contain lq_image. Not used in current training loop.
                if "lq_image" in row and isinstance(row["lq_image"], str):
                    lq_name = row["lq_image"]
                    sample["lq_name"] = lq_name
                    if self.lq_root is not None:
                        sample["lq_path"] = os.path.join(self.lq_root, lq_name)

                return sample

            except Exception as e:
                if retry == 0:
                    print(f"Warning: Failed to load FFHQRef sample idx={idx}: {str(e)}")
                if retry == self.max_retry - 1:
                    raise RuntimeError(
                        f"Failed to load FFHQRef data after {self.max_retry} retries. Last error: {str(e)}"
                    )
                continue

        raise RuntimeError(f"Unexpected error in FFHQRefDataset.__getitem__ for idx {idx}")