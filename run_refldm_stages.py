# -*- coding: utf-8 -*-
import os
import yaml
import subprocess
from copy import deepcopy
from datetime import datetime

# ====== 你的基础 config 路径 ======
BASE_CONFIG = "configs/train_yml/multi_step/VOSR_0.5B_ffhq_test.yml"

# ====== 输出目录 ======
RUN_ROOT = "exp_refldm_stages"

# ====== 训练命令 ======
TRAIN_CMD = "torchrun --nproc_per_node=1 train_vosr_refldm_modular.py --config {config}"

# ====== 阶段定义 ======
STAGES = {
    "B_read_ref": {
        "use_refldm": True,
        "use_ref_cachekv": False,
        "use_ref_sampling": False,
        "use_identity_loss": False,
    },
    "C_cachekv_only": {
        "use_refldm": True,
        "use_ref_cachekv": True,
        "use_ref_sampling": False,
        "use_identity_loss": False,
    },
    "D_identity_only": {
        "use_refldm": True,
        "use_ref_cachekv": False,
        "use_ref_sampling": False,
        "use_identity_loss": True,
        "identity_loss_weight": 2e-5,
        "identity_loss_interval": 4,
    },
    "E_cachekv_identity": {
        "use_refldm": True,
        "use_ref_cachekv": True,
        "use_ref_sampling": False,
        "use_identity_loss": True,
        "identity_loss_weight": 2e-5,
        "identity_loss_interval": 4,
    },
    "F_full": {
        "use_refldm": True,
        "use_ref_cachekv": True,
        "use_ref_sampling": True,
        "use_identity_loss": True,
        "identity_loss_weight": 2e-5,
        "identity_loss_interval": 4,
        "save_ref_ablation": True,
    },
}

# ====== 工具函数 ======
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(cfg, path):
    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

def run_stage(stage_name, base_cfg):
    print(f"\n========== Running Stage: {stage_name} ==========")

    cfg = deepcopy(base_cfg)
    stage_cfg = STAGES[stage_name]

    # 覆盖参数
    for k, v in stage_cfg.items():
        cfg[k] = v

    # 每个阶段单独输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_dir = os.path.join(RUN_ROOT, f"{stage_name}_{timestamp}")
    os.makedirs(stage_dir, exist_ok=True)

    cfg["output_dir"] = stage_dir
    cfg["logging_dir"] = os.path.join(stage_dir, "logs")

    # 保存 config
    cfg_path = os.path.join(stage_dir, "config.yaml")
    save_yaml(cfg, cfg_path)

    # 日志文件
    log_path = os.path.join(stage_dir, "train.log")

    # 构造命令
    cmd = TRAIN_CMD.format(config=cfg_path)

    print(f"Command: {cmd}")
    print(f"Log: {log_path}")

    # 执行训练
    with open(log_path, "w") as f:
        process = subprocess.Popen(cmd, shell=True, stdout=f, stderr=f)
        process.wait()

    if process.returncode != 0:
        print(f"❌ Stage {stage_name} FAILED")
        return False

    print(f"✅ Stage {stage_name} DONE")
    return True

# ====== 主函数 ======
def main():
    base_cfg = load_yaml(BASE_CONFIG)

    for stage in ["B_read_ref", "C_cachekv_only", "D_identity_only", "E_cachekv_identity", "F_full"]:
        ok = run_stage(stage, base_cfg)
        if not ok:
            print("❌ Stop pipeline due to failure")
            break

    print("\n🎯 All stages finished")

if __name__ == "__main__":
    main()


    # python run_refldm_stages.py