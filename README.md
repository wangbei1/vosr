# RefLDM-VOSR

基于 VOSR 与 Ref-LDM 的参考图像人脸超分辨率训练项目。

本项目在原始 VOSR（Flow-Matching Super-Resolution）基础上，引入 Ref-LDM 风格的：

- Reference Cross Attention
- Reference Feature Conditioning
- Identity Loss
- FFHQ-Ref Dataset

用于实现：

```text
Reference-based Face Super-Resolution
```

---

# 1. 训练入口

训练从以下文件开始：

```bash
train_vosr_refldm_modular.py
```

启动方式：

```bash
torchrun --nproc_per_node=1 \
  train_vosr_refldm_modular.py \
  --config configs/train_yml/multi_step/ffhq_config/test_stage1.yml
```

---

# 2. 整体训练流程

```text
FFHQRefDataset
    ↓
读取 HQ / Reference 图像
    ↓
RealESRGAN degradation
    ↓
生成 LQ 图像
    ↓
DINOv2 提取 LQ feature
    ↓
Reference DINO feature
    ↓
LightningDiT
    ├── cross_attn(lq_z)
    └── ref_cross_attn(ref_z)
    ↓
Flow Matching Loss
    ↓
(Optional) Identity Loss
    ↓
反向传播与参数更新
```

---

# 3. 关键文件说明

---

## 3.1 train_vosr_refldm_modular.py

训练主入口。

负责：

- 读取 yml 配置；
- 加载 DINOv2；
- 加载 VAE；
- 构建 LightningDiT；
- patch Ref Attention；
- 构建 FFHQRefDataset；
- 构建 VOSR；
- 构建 RefLDM addon；
- 训练循环；
- checkpoint 保存；
- eval 验证。

这是整个项目最核心的训练文件。

---

## 3.2 dataloaders/ffhq_ref_dataset.py

FFHQ-Ref 数据集读取。

负责：

- 读取 FFHQ-Ref csv；
- 加载 HQ 图像；
- 加载 reference 图像；
- 返回：

```python
{
    "hq": ...,
    "ref": ...
}
```

---

## 3.3 models/refldm_vosr/ref_attention_patch.py

Reference Attention 核心结构修改。

该文件会 patch 原始 LightningDiT：

原始 VOSR：

```python
x = x + self.cross_attn(x, z)
```

当前项目：

```python
x = x + self.cross_attn(x, lq_z)
x = x + gate * scale * self.ref_cross_attn(x, ref_z)
```

即：

```text
LQ feature  → 原始 cross_attn
Ref feature → 新增 ref_cross_attn
```

这是 Ref-LDM 接入 VOSR 的核心文件。

---

## 3.4 models/refldm_vosr/addon_stageE.py

Stage-E Reference 训练逻辑。

核心函数：

```python
prepare_train_pack()
compute_loss()
```

负责：

- Reference feature 提取；
- latent 编码；
- 构造训练输入；
- 计算 loss；
- 调用 Identity Loss。

---

## 3.5 models/refldm_vosr/vosr_stageE_patch.py

修改原始 VOSR：

- Flow Matching Loss
- CFG Sampling
- Ref-aware Sampling

使其支持：

```python
z = {
    "lq": lq_z,
    "ref": ref_z
}
```

并新增：

```python
sample_multistep_fm_ref()
```

用于 reference-aware 推理。

---

## 3.6 models/refldm_vosr/losses_stageE.py

Stage-E Identity Loss。

支持：

```yaml
identity_target:
  - gt
  - ref
  - both
```

用于：

```text
IDS_ref
IDS_gt
```

训练。

---

## 3.7 models/refldm_vosr/identity_loss.py

ArcFace / InsightFace Identity Loss 封装。

加载：

```yaml
identity_model_path
```

用于身份一致性计算。

---

## 3.8 models/refldm_vosr/eval_refaware.py

Reference-aware eval。

负责：

- 读取测试 csv；
- 构造 ref condition；
- build_eval_condition()；
- 推理时使用：

```python
sample_multistep_fm_ref()
```

---

# 4. 三阶段训练

---

## Stage 1：Ref-only Warmup
位置：configs/train_yml/multi_step/ffhq_config/test_stage1.yml
目标：

- 学习 Reference Attention；
- 不破坏原始 VOSR。

配置：

```yaml
freeze_backbone: true
train_ref_only: true
use_identity_loss: false
learning_rate: 0.0001
```

重点观察：

```text
ref_attn_gate
ref_cross_attn grad
LPIPS
MUSIQ
```

---

## Stage 2：Reference Identity Alignment
位置：configs/train_yml/multi_step/ffhq_config/test_stage2.yml

目标：

- 提升 IDS_ref；
- 学习身份一致性。

配置：

```yaml
freeze_backbone: true
train_ref_only: true
use_identity_loss: true
identity_target: ref
identity_loss_weight: 0.00005
learning_rate: 0.00005
```

重点观察：

```text
IDS_ref
LPIPS
```

---

## Stage 3：Full Finetune
位置：configs/train_yml/multi_step/ffhq_config/test_stage3.yml

目标：

- 联合优化超分与身份；
- 微调 backbone。

配置：

```yaml
freeze_backbone: false
train_ref_only: false
identity_target: both
learning_rate: 0.0000005
```

重点观察：

```text
IDS_ref
IDS_gt
LPIPS
MUSIQ
```
