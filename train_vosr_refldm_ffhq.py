# Use the dataset of FFHQ_Ref to train
# add new files:
#   dataloaders/__init__.py
#   dataloaders/ffhq_ref_dataset.py
#   models/refldm/cache_kv.py
#   configs/train_yml/multi_step/VOSR_0.5B_ffhq_ref.yml




import os, re
import yaml
import copy, glob
import logging
import gc
import math
import json
import random
import numpy as np
from pathlib import Path
from collections import OrderedDict

import torch
import timm
import torchvision
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed

from safetensors.torch import save_file
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, make_image_grid

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import pyiqa

# from models.lightningdit_cachekv import LightningDiT
# from vosr_ref import VOSR


from dataloaders import TxtPairDataset, build_webdataset_pipeline, FFHQRefDataset  
from dataloaders.realesrgan_gpu import RealESRGAN_degradation

import ast

my_torch_cache_root = 'preset/ckpts/torch_cache'
os.makedirs(my_torch_cache_root, exist_ok=True)
torch.hub.set_dir(my_torch_cache_root)

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

if hasattr(torch, '_C') and hasattr(torch._C, '_cuda_setDeviceAllocator'):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logger = get_logger(__name__)


def resize_to_512(x):
    return torch.nn.functional.interpolate(
        x,
        size=(512, 512),
        mode="bicubic",
        align_corners=False
    ).clamp(0, 1)


def load_model_weights_with_interpolation(accelerator, target_model, state_dict, model_name="model"):
    target_model_state = target_model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in target_model_state:
            if "pos_embed" in k and v.shape != target_model_state[k].shape:
                if accelerator.is_main_process:
                    print(f"[{model_name}] Resizing {k} from {v.shape} to {target_model_state[k].shape}")

                v_len = v.shape[1]
                target_len = target_model_state[k].shape[1]
                dim = v.shape[-1]

                src_size = int(math.sqrt(v_len))
                tgt_size = int(math.sqrt(target_len))

                v_img = v.reshape(1, src_size, src_size, dim).permute(0, 3, 1, 2)
                v_img = torch.nn.functional.interpolate(
                    v_img, size=(tgt_size, tgt_size), mode='bicubic', align_corners=False
                )
                v = v_img.permute(0, 2, 3, 1).reshape(1, tgt_size * tgt_size, dim)

            if "rope" in k or "freqs_cos" in k or "freqs_sin" in k:
                continue

            new_state_dict[k] = v
        else:
            if accelerator.is_main_process:
                print(f"[{model_name}] Skipping key {k} (not in current model)")

    msg = target_model.load_state_dict(new_state_dict, strict=False)
    if accelerator.is_main_process:
        print(f"[{model_name}] Missing keys: {msg.missing_keys}")
        print(f"[{model_name}] Unexpected keys: {msg.unexpected_keys}")


def _resolve_ckpt_dir(weight_file_path):
    cur = os.path.abspath(weight_file_path)
    for _ in range(5):
        cur = os.path.dirname(cur)
        basename = os.path.basename(cur)
        if basename.startswith("checkpoint-"):
            try:
                step = int(basename.split("-")[-1])
            except ValueError:
                step = 0
            return cur, step
    return None, 0


def find_latest_checkpoint(args):
    checkpoint_dir = f"{args.output_dir}/checkpoints"
    resume_ckpt = args.resume_ckpt

    if resume_ckpt is not None and resume_ckpt != "":
        if os.path.isfile(resume_ckpt):
            ckpt_dir, step = _resolve_ckpt_dir(resume_ckpt)
            if ckpt_dir is not None:
                print(f"resume_ckpt file: {resume_ckpt} -> checkpoint dir: {ckpt_dir} (Step {step})")
                return ckpt_dir, step
            raise ValueError(f"Cannot locate checkpoint-XXXXXXXX directory from: {resume_ckpt}")

        if os.path.isdir(resume_ckpt):
            try:
                step = int(os.path.basename(os.path.normpath(resume_ckpt)).split("-")[-1])
            except ValueError:
                step = 0
            return resume_ckpt, step

        print(f"resume_ckpt does not exist: {resume_ckpt}. Will try auto-discovery.")

    elif os.path.exists(checkpoint_dir):
        subdirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
        if len(subdirs) > 0:
            subdirs.sort(key=lambda x: int(x.split("-")[-1]))
            latest_ckpt_name = subdirs[-1]
            resume_path = os.path.join(checkpoint_dir, latest_ckpt_name)

            try:
                ckpt_files = os.listdir(resume_path)
                has_model_file = any(
                    f in ckpt_files for f in ["pytorch_model.bin", "model.safetensors", "pytorch_model.bin.index.json"]
                )
                if not has_model_file:
                    raise ValueError(f"Checkpoint directory missing model files: {resume_path}")

                global_step = int(latest_ckpt_name.split("-")[-1])
                print(f"Found latest checkpoint: {resume_path} (Step {global_step})")
                return resume_path, global_step

            except (ValueError, Exception) as e:
                if len(subdirs) >= 2:
                    second_latest_ckpt_name = subdirs[-2]
                    resume_path = os.path.join(checkpoint_dir, second_latest_ckpt_name)
                    try:
                        global_step = int(second_latest_ckpt_name.split("-")[-1])
                    except ValueError:
                        global_step = 0
                    print(f"Latest checkpoint failed ({str(e)}). Using second-latest: {resume_path} (Step {global_step})")
                    return resume_path, global_step
                else:
                    try:
                        global_step = int(latest_ckpt_name.split("-")[-1])
                    except ValueError:
                        global_step = 0
                    print(f"Warning: latest checkpoint validation failed ({str(e)}), still using: {resume_path} (Step {global_step})")
                    return resume_path, global_step

    return None, 0


def normalize_report_to(report_to):
    if report_to is None:
        return None
    if isinstance(report_to, str):
        value = report_to.strip()
        if value.lower() in {"", "none", "null", "false", "off", "no"}:
            return None
        return value
    return report_to


def report_to_wandb(report_to):
    if report_to is None:
        return False
    if isinstance(report_to, str):
        return report_to.lower() in {"wandb", "all"}
    return any(str(item).lower() in {"wandb", "all"} for item in report_to)


def filter_collate_fn(batch):
    if not batch:
        return {}

    filtered_batch = []
    for item in batch:
        clean_item = {}
        for k, v in item.items():
            if isinstance(v, (torch.Tensor, int, float, np.ndarray)):
                clean_item[k] = v
        filtered_batch.append(clean_item)

    return torch.utils.data.default_collate(filtered_batch)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    return logging.getLogger(__name__)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        name = name.replace("_orig_mod.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def load_dinov2(args, device):
    local_repo = getattr(args, "dinov2_local_repo", None)
    if local_repo is None:
        raise ValueError("Please set dinov2_local_repo in config for offline loading.")

    if args.enc_type == 'dinov2b':
        model_name = 'dinov2_vitb14'
    elif args.enc_type == 'dinov2l':
        model_name = 'dinov2_vitl14'
    elif args.enc_type == 'dinov2g':
        model_name = 'dinov2_vitg14'
    else:
        raise ValueError(f"Unsupported enc_type: {args.enc_type}")

    encoder = torch.hub.load(
        repo_or_dir=local_repo,
        model=model_name,
        source='local'
    )

    del encoder.head
    encoder.head = torch.nn.Identity()

    def forward_with_features(self, x, masks=None):
        features = {}
        layer_indices = list(range(len(self.blocks)))

        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in layer_indices:
                features[f'layer_{i}'] = x[:, 1:]

        x_norm = self.norm(x)
        return features, x_norm[:, 1:]

    import types
    encoder.forward_with_features = types.MethodType(forward_with_features, encoder)

    encoder = encoder.to(device)
    encoder.eval()
    return encoder


def preprocess_raw_image(x, args):
    x = x / 255.
    x = torch.nn.functional.interpolate(x, args.dinov2_size, mode='bicubic').clip(0., 1.)
    x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    return x

def parse_ref_names(ref_value):
    """
    Parse ref_image field from FFHQ-Ref csv.
    Expected format can be:
        "['00001.png', '00002.png']"
        "00001.png"
    """
    if isinstance(ref_value, list):
        refs = ref_value
    elif isinstance(ref_value, str):
        try:
            refs = ast.literal_eval(ref_value)
        except Exception:
            refs = [ref_value]
    else:
        refs = []

    if not isinstance(refs, list):
        refs = [str(refs)]

    refs = [str(x) for x in refs if str(x).strip() != ""]
    return refs

@torch.no_grad()
def build_ref_condition_for_eval(
    ref_path,
    args,
    accelerator,
    to_tensor,
    unwrapped_venc,
    unwrapped_model_ae,
    latents_mean=None,
    latents_std=None,
):
    """
    Build ref_eval, ref_z_eval, ref_latent for reference-aware eval.

    Returns:
        ref_eval: [-1,1], [1,3,H,W]
        ref_z_eval: list of DINO features
        ref_latent: VAE latent
    """
    with Image.open(ref_path) as img:
        ref_img = img.convert("RGB")

    ref_eval = to_tensor(ref_img).unsqueeze(0).to(accelerator.device)
    ref_eval = ref_eval * 2.0 - 1.0

    ref_eval = torch.nn.functional.interpolate(
        ref_eval,
        size=(args.resolution, args.resolution),
        mode="bicubic",
        align_corners=False,
    ).clamp(-1, 1)

    with accelerator.autocast():
        raw_ref = (0.5 * ref_eval + 0.5) * 255
        raw_ref_ = preprocess_raw_image(raw_ref, args)

        ref_features, ref_x_norm = unwrapped_venc.forward_with_features(raw_ref_)
        ref_z_eval = [v for k, v in ref_features.items() if k.startswith("layer_")]
        ref_z_eval[-1] = ref_x_norm
        ref_z_eval = [ref_z_eval[i] for i in args.layer_dinov2b_list]

        if args.ae_type == "qwen":
            ref_latent = (
                unwrapped_model_ae.encode(ref_eval).latent_dist.sample() - latents_mean
            ) * latents_std

        elif args.ae_type == "sd2":
            ref_latent = (
                unwrapped_model_ae.encode(
                    ref_eval.to(unwrapped_model_ae.dtype)
                ).latent_dist.sample()
                * unwrapped_model_ae.config.scaling_factor
            )

        else:
            raise ValueError(f"Unsupported ae_type: {args.ae_type}")

    return ref_eval, ref_z_eval, ref_latent

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_dataset_config(path):
    txt_paths = []
    probs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 1:
                dataset_path, repeat = parts[0], 1
            elif len(parts) == 2:
                dataset_path, repeat = parts[0], int(parts[1])
            else:
                raise ValueError(f"Invalid line format: {line}")
            txt_paths.append(dataset_path)
            probs.append(repeat)
    return txt_paths, probs


def main(config_path):
    from argparse import Namespace

    config = load_config(config_path)
    args = Namespace(**config)
    args.report_to = normalize_report_to(getattr(args, "report_to", None))

    use_refldm = getattr(args, "use_refldm", False)

    if use_refldm:
        from models.lightningdit_cachekv import LightningDiT
        from vosr_ref import VOSR
        print("===> use_refldm=True: using Ref-LDM cache-kv VOSR path")
    else:
        from models.lightningdit import LightningDiT
        from vosr import VOSR
        print("===> use_refldm=False: using original VOSR path")

    train_dataset_config = getattr(args, "train_dataset_config", None)
    if train_dataset_config is not None:
        txt_paths, probs = load_dataset_config(train_dataset_config)
        args.train_dataset_txt_paths_list = txt_paths
        args.train_dataset_prob_paths_list = probs

    weak_cond_strength_aelq = f'{args.weak_cond_strength_aelq_list[0]}-{args.weak_cond_strength_aelq_list[1]}'
    cfg_param = f's{args.cfg_scale}-r{args.cfg_ratio}-wc{weak_cond_strength_aelq}'

    if args.time_dist[0] == 'uniform':
        td = 'uni'
    elif args.time_dist[0] == 'lognorm':
        td = f'ln_{args.time_dist[1]}_{args.time_dist[2]}'
    else:
        raise ValueError(f"Unsupported time_dist: {args.time_dist}")

    resume = '_resume' if args.resume_ckpt is not None else ''

    downsample_scale = 8
    base_channel = 4 if args.ae_type == 'sd2' else 16
    args.exp_name = (
        f'ldit_fm_bs{args.train_batch_size * args.gradient_accumulation_steps:03d}'
        f'_{args.ae_type}f{downsample_scale}c{base_channel}'
        f'_size{args.resolution}'
        f'_ps{args.patch_size}'
        f'_d{args.dim}'
        f'_b{args.depth}'
        f'_h{args.num_heads}'
        f'_cfg{cfg_param}'
        f'_edr{args.encdim_ratio}'
        f'_td{td}'
        f'_type{args.dataset_type}'
        f"{resume}"
        f"{args.suffix}"
    )

    print(f'===> expname is {args.exp_name}')
    args.output_dir = os.path.join(args.output_dir, args.exp_name)

    logging_out_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=str(args.logging_dir)
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    print(f'===> choose {len(args.layer_dinov2b_list)} layer fea for CA.')
    venc = load_dinov2(args, device=accelerator.device)
    print('===> loading vision encoder')

    checkpoint_dir = f"{args.output_dir}/checkpoints"

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        args_dict = vars(args)
        json_dir = os.path.join(args.output_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        os.makedirs(checkpoint_dir, exist_ok=True)

        global logger
        logger = create_logger(args.output_dir)
        logger.info(f"Experiment directory created at {args.output_dir}")

        iqa_lpips = pyiqa.create_metric('lpips', device='cuda')
        iqa_musiq = pyiqa.create_metric('musiq', device='cuda')

    if args.ae_type == "qwen":
        from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
        model_ae = AutoencoderKLQwenImage2D.from_pretrained(args.ae_path)
    elif args.ae_type == "sd2":
        from diffusers import AutoencoderKL
        model_ae = AutoencoderKL.from_pretrained(args.ae_path, subfolder="vae")
    else:
        raise ValueError(f"Unsupported ae_type: {args.ae_type}")

    model_ae = model_ae.to(accelerator.device)
    model_ae.eval()

    model = LightningDiT(
        input_size=args.resolution // 8,
        patch_size=args.patch_size,
        in_channels=2 * base_channel,
        out_channels=base_channel,
        hidden_size=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        z_dims=args.enc_dim,
        encdim_ratio=args.encdim_ratio,
        auxiliary_time_cond=False,
        use_qknorm=args.use_qknorm,
        use_swiglu=args.use_swiglu,
        use_rope=args.use_rope,
        use_rmsnorm=args.use_rmsnorm,
        wo_shift=args.wo_shift,
        num_fused_layers=len(args.layer_dinov2b_list),
    )

    total_params = sum(p.numel() for p in model.parameters())
    total_params_in_billion = total_params / 1e9
    if accelerator.is_main_process:
        logger.info(f"===========> Total parameters: {total_params} ({total_params_in_billion:.3f} B)")

    model.train()

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install bitsandbytes: `pip install bitsandbytes`")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    degradation = RealESRGAN_degradation('params_realsr.yml', device=accelerator.device)

    global_step = 0
    resume_path, current_step = find_latest_checkpoint(args)

    if args.seed is not None:
        seed_to_use = args.seed + current_step
        set_seed(seed_to_use)
        if accelerator.is_main_process:
            logger.info(f"Seed set to {seed_to_use} (Base {args.seed} + Step {current_step})")

    local_batch_size = int(args.train_batch_size // accelerator.num_processes)

    if accelerator.is_main_process:
        logger.info(f"===========> Batch Size Debug Info (DDP):")
        logger.info(f"  args.train_batch_size (global) = {args.train_batch_size}")
        logger.info(f"  accelerator.num_processes (total GPUs) = {accelerator.num_processes}")
        logger.info(f"  local_batch_size (per GPU) = {local_batch_size}")
        logger.info(f"  gradient_accumulation_steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size per GPU = {local_batch_size * args.gradient_accumulation_steps}")
        logger.info(f"  Total effective batch size = {local_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
        logger.info(f"  Expected total batch size = {args.train_batch_size * args.gradient_accumulation_steps}")
        if local_batch_size * accelerator.num_processes * args.gradient_accumulation_steps != args.train_batch_size * args.gradient_accumulation_steps:
            logger.warning("  ⚠️ WARNING: Batch size mismatch! This may cause training issues.")

    num_workers = args.dataloader_num_workers
    prefetch_factor = 4 if args.dataloader_num_workers > 0 else None

    if args.dataset_type == 'txt':
        train_dataset = TxtPairDataset(split='train', args=args)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=local_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor
        )

    elif args.dataset_type == 'webdataset':
        train_dataset = build_webdataset_pipeline(args, split='train')
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=local_batch_size,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if args.dataloader_num_workers > 0 else False,
            prefetch_factor=4,
            collate_fn=filter_collate_fn,
        )

    elif args.dataset_type == 'ffhq_ref':
        train_dataset = FFHQRefDataset(split='train', args=args)

        # print("[TRAIN DEBUG] dataset class:", train_dataset.__class__)
        # print("[TRAIN DEBUG] dataset module:", train_dataset.__class__.__module__)

        # _debug_sample = train_dataset[0]
        # print("[TRAIN DEBUG] first sample keys:", list(_debug_sample.keys()))
        # if "ref" in _debug_sample:
        #     print("[TRAIN DEBUG] first sample ref shape:", _debug_sample["ref"].shape)
        # else:
        #     print("[TRAIN DEBUG] WARNING: first sample has no 'ref'")


        train_dataloader = DataLoader(
            train_dataset,
            batch_size=local_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor,
            collate_fn=filter_collate_fn,
        )

    else:
        raise ValueError(f"Unsupported dataset_type: {args.dataset_type}")

    if accelerator.is_main_process:
        to_tensor = transforms.ToTensor()
        to_pil = transforms.ToPILImage()

    vosr = VOSR(
        time_dist=args.time_dist,
        cfg_ratio=args.cfg_ratio,
        cfg_scale=args.cfg_scale,
        interp_type=args.interp_type,
        a=args.a,
        b=args.b,
        accelerator=accelerator,
        t_start=args.t_start,
        t_end=args.t_end,
        args=args
    )

    if args.use_ema:
        ema = copy.deepcopy(model)
        ema = ema.to(accelerator.device)
        requires_grad(ema, False)
        ema.eval()
        accelerator.register_for_checkpointing(ema)
    else:
        ema = None

    model, model_ae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, model_ae, optimizer, train_dataloader, lr_scheduler
    )

    if resume_path:
        global_step = current_step
        if accelerator.is_main_process:
            logger.info(f"Loading state from {resume_path}")
        accelerator.load_state(resume_path)

    elif args.pretrained_ckpt:
        pretrained_path = args.pretrained_ckpt
        if accelerator.is_main_process:
            print(f'=======> Loading pretrained weights from {pretrained_path}')

        from safetensors.torch import load_file
        if pretrained_path.endswith(".safetensors"):
            state_dict = load_file(pretrained_path)
        else:
            state_dict = torch.load(pretrained_path, map_location="cpu")

        unwrapped_model = accelerator.unwrap_model(model)
        load_model_weights_with_interpolation(accelerator, unwrapped_model, state_dict, model_name="Main Model")

        if args.use_ema:
            if accelerator.is_main_process:
                print("=======> Re-initializing EMA from loaded model weights")
            ema.load_state_dict(unwrapped_model.state_dict())

        global_step = 0
        if accelerator.is_main_process:
            print("=======> Optimizer and Scheduler reset. Global step set to 0.")

    else:
        global_step = 0

    if accelerator.is_main_process and args.report_to is not None:
        tracker_config = vars(copy.deepcopy(args))
        for key, value in tracker_config.items():
            if isinstance(value, list):
                tracker_config[key] = str(value)

        init_kwargs = {}
        if report_to_wandb(args.report_to):
            wandb_id_file = os.path.join(args.output_dir, "wandb_id.txt")
            wandb_id = None

            if os.path.exists(wandb_id_file):
                with open(wandb_id_file, 'r') as f:
                    wandb_id = f.read().strip()
                print(f"=======> Resuming WandB run with ID: {wandb_id}")

            if wandb_id is None or wandb_id == "":
                import uuid
                wandb_id = uuid.uuid4().hex
                os.makedirs(args.output_dir, exist_ok=True)
                with open(wandb_id_file, 'w') as f:
                    f.write(wandb_id)
                print(f"=======> Created new WandB run with ID: {wandb_id}")

            init_kwargs["wandb"] = {
                "name": f"{args.exp_name}",
                "dir": args.output_dir,
                "id": wandb_id,
                "resume": "allow",
            }

        accelerator.init_trackers(
            project_name=args.tracker_project_name,
            config=tracker_config,
            init_kwargs=init_kwargs,
        )

    total_batch_size = args.train_batch_size
    if accelerator.is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Instantaneous batch size per device = {local_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        logger.info(f"  Number of processes (GPUs) = {accelerator.num_processes}")
        logger.info(f"  Number of nodes = {accelerator.num_processes // 8 if accelerator.num_processes >= 8 else 1} (assuming 8 GPUs per node)")
        logger.info(f"  Distributed type = {accelerator.distributed_type}")
        logger.info(f" EXP = {args.exp_name}")

    initial_global_step = global_step
    progress_bar = tqdm(
        total=args.max_train_steps,
        initial=initial_global_step,
        desc="Training",
        dynamic_ncols=True,
        mininterval=1.0,
        leave=True,
        disable=not accelerator.is_local_main_process,
    )

    epoch = 0
    unwrapped_model_ae = accelerator.unwrap_model(model_ae)
    unwrapped_venc = accelerator.unwrap_model(venc)

    if args.ae_type == "qwen":
        latents_mean = (
            torch.tensor(unwrapped_model_ae.config.latents_mean)
            .view(1, unwrapped_model_ae.config.z_dim, 1, 1)
            .to(unwrapped_model_ae.device, unwrapped_model_ae.dtype)
        )
        latents_std = 1.0 / torch.tensor(unwrapped_model_ae.config.latents_std).view(
            1, unwrapped_model_ae.config.z_dim, 1, 1
        ).to(unwrapped_model_ae.device, unwrapped_model_ae.dtype)

    gc.disable()

    while global_step < args.max_train_steps:
        epoch += 1

        if epoch > 1:
            gc.collect()
            torch.cuda.empty_cache()

        for batch_idx, batch in enumerate(train_dataloader):
            debug_this_step = global_step < 3 and accelerator.is_main_process

            if debug_this_step and accelerator.is_main_process:
                tqdm.write(f"[DEBUG] got batch {batch_idx}")
                tqdm.write(f"[DEBUG] batch keys: {list(batch.keys())}")

            hq = batch["hq"]
            if use_refldm:
                if "ref" not in batch:
                    raise KeyError(
                        f"use_refldm=True but batch has no 'ref'. "
                        f"Current batch keys: {list(batch.keys())}"
                    )
                ref = batch["ref"]

                use_multi_ref = getattr(args, "use_multi_ref", False)

                # Dataset may always return [B,K,C,H,W].
                # If multi-ref is disabled, use only the first reference.
                if use_refldm and (not use_multi_ref) and ref.dim() == 5:
                    ref = ref[:, 0]   # [B,C,H,W]
            else:
                ref = None

            if debug_this_step and accelerator.is_main_process:
                tqdm.write(
                    f"[DEBUG] hq cpu shape={hq.shape}, dtype={hq.dtype}, "
                    f"min={hq.min().item():.6f}, max={hq.max().item():.6f}"
                )
                if use_refldm:
                    tqdm.write(
                        f"[DEBUG] ref cpu shape={ref.shape}, dtype={ref.dtype}, "
                        f"min={ref.min().item():.6f}, max={ref.max().item():.6f}"
                    )

            if not torch.isfinite(hq).all():
                raise ValueError("[DEBUG] Non-finite values found in CPU hq batch")
            if use_refldm and not torch.isfinite(ref).all():
                raise ValueError("[DEBUG] Non-finite values found in CPU ref batch")

            hq = hq.to(accelerator.device, non_blocking=True)
            if use_refldm:
                ref = ref.to(accelerator.device, non_blocking=True)

            if debug_this_step and accelerator.is_main_process:
                tqdm.write(f"[DEBUG] hq/ref moved to {hq.device}")

            if not torch.isfinite(hq).all():
                raise ValueError("[DEBUG] Non-finite values found in GPU hq batch")
            if use_refldm and not torch.isfinite(ref).all():
                raise ValueError("[DEBUG] Non-finite values found in GPU ref batch")

            with torch.no_grad():
                if debug_this_step and accelerator.is_main_process:
                    tqdm.write("[DEBUG] before degradation")

                _, lq = degradation.degrade_process(hq, resize_bak=True)

                if debug_this_step and accelerator.is_main_process:
                    tqdm.write("[DEBUG] after degradation")

                if not torch.isfinite(lq).all():
                    raise ValueError("[DEBUG] Non-finite values found in lq after degradation")

                hq = hq * 2 - 1
                lq = lq * 2 - 1

                if debug_this_step and accelerator.is_main_process:
                    tqdm.write(f"[DEBUG] hq range after norm: {hq.min().item():.6f} ~ {hq.max().item():.6f}")
                    tqdm.write(f"[DEBUG] lq range after norm: {lq.min().item():.6f} ~ {lq.max().item():.6f}")

            with torch.no_grad():
                if debug_this_step and accelerator.is_main_process:
                    tqdm.write("[DEBUG] before venc preprocess (lq/ref)")

                raw_image = (0.5 * lq + 0.5) * 255
                raw_image_ = preprocess_raw_image(raw_image, args)
                if not torch.isfinite(raw_image_).all():
                    raise ValueError("[DEBUG] Non-finite values found in raw_image_")

                ref_z = None
                ref_flat = None
                B_ref, K_ref = None, None

                if use_refldm:
                    use_multi_ref = getattr(args, "use_multi_ref", False)

                    if use_multi_ref:
                        if ref.dim() == 4:
                            # fallback: single ref -> fake K=1
                            ref = ref.unsqueeze(1)

                        B_ref, K_ref, C_ref, H_ref, W_ref = ref.shape
                        ref_flat = ref.reshape(B_ref * K_ref, C_ref, H_ref, W_ref)

                    else:
                        if ref.dim() == 5:
                            ref = ref[:, 0]   # [B,C,H,W]

                        B_ref = ref.shape[0]
                        K_ref = 1
                        ref_flat = ref

                    ref_m11_flat = ref_flat * 2 - 1
                    raw_ref = (0.5 * ref_m11_flat + 0.5) * 255
                    raw_ref_ = preprocess_raw_image(raw_ref, args)

                    if not torch.isfinite(raw_ref_).all():
                        raise ValueError("[DEBUG] Non-finite values found in raw_ref_")

                with accelerator.autocast():
                    if debug_this_step and accelerator.is_main_process:
                        tqdm.write("[DEBUG] before venc forward (lq)")
                    features, x_norm = unwrapped_venc.forward_with_features(raw_image_)
                    
                    if use_refldm:
                        if debug_this_step and accelerator.is_main_process:
                            tqdm.write("[DEBUG] before venc forward (ref)")
                        ref_features, ref_x_norm = unwrapped_venc.forward_with_features(raw_ref_)

                    z = [v for k, v in features.items() if k.startswith('layer_')]
                    z[-1] = x_norm
                    z = [z[i] for i in args.layer_dinov2b_list]
                    if use_refldm:
                        ref_z = [v for k, v in ref_features.items() if k.startswith('layer_')]
                        ref_z[-1] = ref_x_norm
                        ref_z = [ref_z[i] for i in args.layer_dinov2b_list]

                    if debug_this_step and accelerator.is_main_process:
                        for zi, feat in enumerate(z):
                            tqdm.write(f"[DEBUG] z[{zi}] shape={feat.shape}, dtype={feat.dtype}")
                        if use_refldm:
                            for zi, feat in enumerate(ref_z):
                                tqdm.write(f"[DEBUG] ref_z[{zi}] shape={feat.shape}, dtype={feat.dtype}")

            with accelerator.accumulate(model):
                if global_step >= args.max_train_steps:
                    break

                with torch.no_grad():
                    if use_refldm:
                        use_multi_ref = getattr(args, "use_multi_ref", False)

                        if use_multi_ref:
                            if ref.dim() == 4:
                                ref = ref.unsqueeze(1)

                            B_ref, K_ref, C_ref, H_ref, W_ref = ref.shape
                            ref_flat = ref.reshape(B_ref * K_ref, C_ref, H_ref, W_ref)

                        else:
                            if ref.dim() == 5:
                                ref = ref[:, 0]

                            B_ref = ref.shape[0]
                            K_ref = 1
                            ref_flat = ref

                        ref_m11_flat = ref_flat * 2 - 1
                        combined = torch.cat([lq, hq, ref_m11_flat], dim=0)
                    else:
                        combined = torch.cat([lq, hq], dim=0)
                    if args.ae_type == 'qwen':
                        combined_latent = (
                            unwrapped_model_ae.encode(combined).latent_dist.sample() - latents_mean
                        ) * latents_std
                    elif args.ae_type == 'sd2':
                        combined_latent = (
                            unwrapped_model_ae.encode(
                                combined.to(unwrapped_model_ae.dtype)
                            ).latent_dist.sample()
                            * unwrapped_model_ae.config.scaling_factor
                        )
                    else:
                        raise ValueError(f"Unsupported ae_type: {args.ae_type}")

                    if use_refldm:
                        B = lq.shape[0]
                        lq_latent = combined_latent[:B]
                        hq_latent = combined_latent[B:2 * B]
                        ref_latent = combined_latent[2 * B:]  # [B*K,C,h,w] if multi-ref
                    else:
                        lq_latent, hq_latent = combined_latent.chunk(2, dim=0)
                        ref_latent = None

                with accelerator.autocast():
                    if use_refldm:
                        loss, loss_backward = vosr.loss_fm(
                            model,
                            lq_latent,
                            hq_latent,
                            z,
                            ref_latent=ref_latent,
                            ref_z=ref_z,
                        )
                    else:
                        loss, loss_backward = vosr.loss_fm(
                            model,
                            lq_latent,
                            hq_latent,
                            z,
                        )

                accelerator.backward(loss_backward)

                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients and args.use_ema:
                    update_ema(ema, accelerator.unwrap_model(model), args.ema_decay)

            if accelerator.sync_gradients:
                lr_scheduler.step()
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0 and global_step > 0:
                    accelerator.wait_for_everyone()
                    gc.collect()
                    torch.cuda.empty_cache()

                    ckpt_dir = f"{checkpoint_dir}/checkpoint-{global_step:08d}"
                    accelerator.save_state(ckpt_dir)

                    if accelerator.is_main_process:
                        import threading

                        unwrapped_model = accelerator.unwrap_model(model)
                        model_state_cpu = {k: v.cpu().clone() for k, v in unwrapped_model.state_dict().items()}
                        ema_state_cpu = {k: v.cpu().clone() for k, v in ema.state_dict().items()} if args.use_ema else None

                        def save_clean_weights(model_state, ema_state, save_dir):
                            os.makedirs(save_dir, exist_ok=True)
                            clean_model_dir = f"{save_dir}/clean_weights"
                            os.makedirs(clean_model_dir, exist_ok=True)

                            save_file(model_state, f"{clean_model_dir}/model.safetensors")
                            if ema_state is not None:
                                save_file(ema_state, f"{clean_model_dir}/ema_model.safetensors")

                        save_thread = threading.Thread(
                            target=save_clean_weights,
                            args=(model_state_cpu, ema_state_cpu, ckpt_dir),
                            daemon=False,
                        )
                        save_thread.start()

                    accelerator.wait_for_everyone()
                    torch.cuda.empty_cache()
                #
                if global_step % args.inference_steps == 1 and global_step > 0:
                    if accelerator.is_main_process:
                        model.eval()
                        eval_model = ema if args.use_ema else accelerator.unwrap_model(model)

                        sr_dir = Path(args.output_dir, "sr", f"{global_step:08d}")
                        sr_dir.mkdir(parents=True, exist_ok=True)

                        lpips_sum, musiq_sum, n_img = 0.0, 0.0, 0
                        use_ffhq_csv_eval = getattr(args, "ffhq_ref_test_use_csv_for_eval", False)

                        with torch.no_grad():
                            if use_ffhq_csv_eval:
                                import pandas as pd

                                test_df = pd.read_csv(args.ffhq_ref_test_csv)
                                hq_dir = Path(args.ffhq_ref_eval_hq_dir)
                                lq_dir = Path(args.ffhq_ref_eval_lq_dir)

                                logger.info(
                                    f"[Eval @ step {global_step}] Using FFHQ_Ref CSV eval: "
                                    f"csv={args.ffhq_ref_test_csv}, lq_dir={lq_dir}, hq_dir={hq_dir}, "
                                    f"use_refldm={use_refldm}"
                                )

                                eval_bar = tqdm(
                                    test_df.iterrows(),
                                    total=len(test_df),
                                    desc=f"Eval@{global_step}",
                                    dynamic_ncols=True,
                                    leave=False,
                                    disable=not accelerator.is_main_process,
                                )

                                for _, row in eval_bar:
                                    gt_name = row["gt_image"]
                                    lq_name = row["lq_image"]

                                    gt_path = hq_dir / gt_name
                                    lq_path = lq_dir / lq_name

                                    if not gt_path.exists() or not lq_path.exists():
                                        continue

                                    ref_z_eval, ref_latent = None, None
                                    if use_refldm:
                                        ref_names = parse_ref_names(row["ref_image"])
                                        if len(ref_names) == 0:
                                            logger.warning(f"[Eval] No ref found for {gt_name}, skip.")
                                            continue

                                        ref_name = ref_names[0]
                                        ref_path = hq_dir / ref_name

                                        if not ref_path.exists():
                                            logger.warning(f"[Eval] ref missing: {ref_path}, skip.")
                                            continue

                                    name = Path(gt_name).stem

                                    with Image.open(lq_path) as img:
                                        lq_img = img.convert("RGB")

                                    if hasattr(args, "test_upscale") and args.test_upscale > 1:
                                        w, h = lq_img.size
                                        lq_img = lq_img.resize(
                                            (w * args.test_upscale, h * args.test_upscale),
                                            Image.LANCZOS
                                        )

                                    lq_eval = to_tensor(lq_img).unsqueeze(0).to(accelerator.device)
                                    lq_eval = lq_eval * 2.0 - 1.0

                                    with Image.open(gt_path) as img:
                                        gt_img = img.convert("RGB")
                                    gt_01 = to_tensor(gt_img).unsqueeze(0).to(accelerator.device)

                                    # LQ DINO feature
                                    with accelerator.autocast():
                                        raw_image = (0.5 * lq_eval + 0.5) * 255
                                        raw_image_ = preprocess_raw_image(raw_image, args)

                                        features, x_norm = unwrapped_venc.forward_with_features(raw_image_)
                                        z_eval = [v for k, v in features.items() if k.startswith("layer_")]
                                        z_eval[-1] = x_norm
                                        z_eval = [z_eval[i] for i in args.layer_dinov2b_list]

                                    # REF DINO feature + REF latent
                                    if use_refldm:
                                        _, ref_z_eval, ref_latent = build_ref_condition_for_eval(
                                            ref_path=ref_path,
                                            args=args,
                                            accelerator=accelerator,
                                            to_tensor=to_tensor,
                                            unwrapped_venc=unwrapped_venc,
                                            unwrapped_model_ae=unwrapped_model_ae,
                                            latents_mean=latents_mean if args.ae_type == "qwen" else None,
                                            latents_std=latents_std if args.ae_type == "qwen" else None,
                                        )

                                    # LQ latent + sampling
                                    with accelerator.autocast():
                                        if args.ae_type == "qwen":
                                            lq_latent = (
                                                unwrapped_model_ae.encode(lq_eval).latent_dist.sample() - latents_mean
                                            ) * latents_std

                                            if use_refldm:
                                                sr_m11 = vosr.sample_multistep_fm_ref(
                                                    eval_model,
                                                    lq_latent,
                                                    ref_latent,
                                                    n_steps=args.infer_steps,
                                                    venc_fea=z_eval,
                                                    ref_fea=ref_z_eval,
                                                )
                                            else:
                                                sr_m11 = vosr.sample_multistep_fm(
                                                    eval_model,
                                                    lq_latent,
                                                    n_steps=args.infer_steps,
                                                    venc_fea=z_eval,
                                                )

                                            sr_m11 = sr_m11 / latents_std + latents_mean
                                            sr_m11 = unwrapped_model_ae.decode(
                                                sr_m11,
                                                return_dict=False
                                            )[0].clamp(-1, 1)

                                        elif args.ae_type == "sd2":
                                            lq_latent = (
                                                unwrapped_model_ae.encode(
                                                    lq_eval.to(unwrapped_model_ae.dtype)
                                                ).latent_dist.sample()
                                                * unwrapped_model_ae.config.scaling_factor
                                            )

                                            if use_refldm:
                                                sr_m11 = vosr.sample_multistep_fm_ref(
                                                    eval_model,
                                                    lq_latent,
                                                    ref_latent,
                                                    n_steps=args.infer_steps,
                                                    venc_fea=z_eval,
                                                    ref_fea=ref_z_eval,
                                                )
                                            else:
                                                sr_m11 = vosr.sample_multistep_fm(
                                                    eval_model,
                                                    lq_latent,
                                                    n_steps=args.infer_steps,
                                                    venc_fea=z_eval,
                                                )

                                            sr_m11 = sr_m11 / unwrapped_model_ae.config.scaling_factor
                                            sr_m11 = unwrapped_model_ae.decode(
                                                sr_m11,
                                                return_dict=False
                                            )[0].clamp(-1, 1)

                                        else:
                                            raise ValueError(f"Unsupported ae_type: {args.ae_type}")

                                    sr_01 = (sr_m11 + 1.0) / 2.0
                                    sr_01 = sr_01.clamp(0, 1)
                                    sr_01 = resize_to_512(sr_01)
                                    gt_01 = resize_to_512(gt_01)

                                    if accelerator.is_main_process and n_img < 3:
                                        logger.info(
                                            f"[Eval Debug] {name} | "
                                            f"sr_m11=({sr_m11.min().item():.4f}, {sr_m11.max().item():.4f}) | "
                                            f"sr_01=({sr_01.min().item():.4f}, {sr_01.max().item():.4f}, "
                                            f"mean={sr_01.mean().item():.4f})"
                                        )

                                    to_pil(sr_01.squeeze(0).cpu()).save(sr_dir / f"{name}.png")

                                    sr_metric = (sr_01 * 255).clamp(0, 255).byte().float() / 255.0
                                    lpips_sum += iqa_lpips(gt_01, sr_metric.clip(0, 1))
                                    musiq_sum += iqa_musiq(sr_metric.clip(0, 1))
                                    n_img += 1

                            else:
                                # folder eval does not contain ref_image metadata.
                                # Keep original VOSR eval path here.
                                if args.test_lq_dir is None or args.test_gt_dir is None:
                                    logger.warning(
                                        f"[Eval @ step {global_step}] Skip folder-based eval because "
                                        f"test_lq_dir={args.test_lq_dir}, test_gt_dir={args.test_gt_dir}"
                                    )
                                else:
                                    lq_dir = Path(args.test_lq_dir)
                                    gt_dir = Path(args.test_gt_dir)

                                    logger.info(
                                        f"[Eval @ step {global_step}] Folder eval without reference: "
                                        f"lq_dir={lq_dir}, gt_dir={gt_dir}"
                                    )

                                    for lq_path in sorted(lq_dir.glob("*.png")):
                                        name = lq_path.stem
                                        gt_path = gt_dir / f"{name}.png"
                                        if not gt_path.exists():
                                            continue

                                        with Image.open(lq_path) as img:
                                            lq_img = img.convert("RGB")

                                        if hasattr(args, "test_upscale") and args.test_upscale > 1:
                                            w, h = lq_img.size
                                            lq_img = lq_img.resize(
                                                (w * args.test_upscale, h * args.test_upscale),
                                                Image.LANCZOS
                                            )

                                        lq_eval = to_tensor(lq_img).unsqueeze(0).to(accelerator.device)
                                        lq_eval = lq_eval * 2.0 - 1.0

                                        with Image.open(gt_path) as img:
                                            gt_img = img.convert("RGB")
                                        gt_01 = to_tensor(gt_img).unsqueeze(0).to(accelerator.device)

                                        with accelerator.autocast():
                                            raw_image = (0.5 * lq_eval + 0.5) * 255
                                            raw_image_ = preprocess_raw_image(raw_image, args)

                                            features, x_norm = unwrapped_venc.forward_with_features(raw_image_)
                                            z_eval = [v for k, v in features.items() if k.startswith("layer_")]
                                            z_eval[-1] = x_norm
                                            z_eval = [z_eval[i] for i in args.layer_dinov2b_list]

                                            if args.ae_type == "qwen":
                                                lq_latent = (
                                                    unwrapped_model_ae.encode(lq_eval).latent_dist.sample()
                                                    - latents_mean
                                                ) * latents_std

                                                sr_m11 = vosr.sample_multistep_fm(
                                                    eval_model,
                                                    lq_latent,
                                                    n_steps=args.infer_steps,
                                                    venc_fea=z_eval,
                                                )

                                                sr_m11 = sr_m11 / latents_std + latents_mean
                                                sr_m11 = unwrapped_model_ae.decode(
                                                    sr_m11,
                                                    return_dict=False
                                                )[0].clamp(-1, 1)

                                            elif args.ae_type == "sd2":
                                                lq_latent = (
                                                    unwrapped_model_ae.encode(
                                                        lq_eval.to(unwrapped_model_ae.dtype)
                                                    ).latent_dist.sample()
                                                    * unwrapped_model_ae.config.scaling_factor
                                                )

                                                sr_m11 = vosr.sample_multistep_fm(
                                                    eval_model,
                                                    lq_latent,
                                                    n_steps=args.infer_steps,
                                                    venc_fea=z_eval,
                                                )

                                                sr_m11 = sr_m11 / unwrapped_model_ae.config.scaling_factor
                                                sr_m11 = unwrapped_model_ae.decode(
                                                    sr_m11,
                                                    return_dict=False
                                                )[0].clamp(-1, 1)

                                            else:
                                                raise ValueError(f"Unsupported ae_type: {args.ae_type}")

                                        sr_01 = (sr_m11 + 1.0) / 2.0
                                        sr_01 = sr_01.clamp(0, 1)
                                        sr_01 = resize_to_512(sr_01)
                                        gt_01 = resize_to_512(gt_01)

                                        to_pil(sr_01.squeeze(0).cpu()).save(sr_dir / f"{name}.png")

                                        sr_metric = (sr_01 * 255).clamp(0, 255).byte().float() / 255.0
                                        lpips_sum += iqa_lpips(gt_01, sr_metric.clip(0, 1))
                                        musiq_sum += iqa_musiq(sr_metric.clip(0, 1))
                                        n_img += 1

                        if n_img > 0:
                            lpips_avg = lpips_sum.float().item() / n_img
                            musiq_avg = musiq_sum.float().item() / n_img

                            logger.info(
                                f"[Eval @ step {global_step}] LPIPS: {lpips_avg:.4f}  MUSIQ: {musiq_avg:.4f}"
                            )

                            accelerator.log(
                                {
                                    "eval/lpips": lpips_avg,
                                    "eval/musiq": musiq_avg,
                                },
                                step=global_step
                            )
                        else:
                            logger.warning(f"[Eval @ step {global_step}] No valid eval image pairs found.")

                        model.train()
                        if args.use_ema:
                            ema.eval()

            if global_step % 50 == 0:
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "v_loss": loss.item()
                }
                progress_bar.set_postfix(**logs)

            if accelerator.is_main_process and global_step % 250 == 0:
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "v_loss": loss.item()
                }
                accelerator.log(logs, step=global_step)

    gc.enable()
    gc.collect()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("🎉 Training completed successfully!")
        logger.info(f"Total steps: {global_step}/{args.max_train_steps}")
        logger.info(f"Final checkpoint dir: {checkpoint_dir}")
    accelerator.end_training()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_yml/vosr_fm_256_tar.yml', help='Path to config file')
    args = parser.parse_args()

    main(args.config)


"""
export CUDA_VISIBLE_DEVICES=2

nohup torchrun --nproc_per_node=1 train_vosr_refldm_ffhq.py \
  --config configs/train_yml/multi_step/VOSR_0.5B_ffhq_refldm_smoke.yml \
  > train_vosr_refldm_ffhq_smoke.log 2>&1 &


  nohup torchrun --nproc_per_node=1 train_vosr_refldm_ffhq.py \
  --config configs/train_yml/multi_step/VOSR_0.5B_ffhq_refldm.yml \
  > train_vosr_refldm_ffhq.log 2>&1 &



"""