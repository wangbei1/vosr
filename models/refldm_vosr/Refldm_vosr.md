## `refldm_vosr` 模块总体作用

`refldm_vosr` 是在原始 VOSR 基础上加入 Ref-LDM 风格参考图条件的扩展模块。它的核心目标是：在不破坏原 VOSR 超分主干能力的前提下，引入参考图像的身份和外观信息，使模型在 FFHQ-Ref 任务中同时保持输入低质量图像的结构，以及参考图像的人脸身份特征。

整体结构可以概括为：

```text
LQ 图像
  → VAE encode
  → LQ latent

LQ 图像
  → DINOv2
  → lq_z
  → 原 VOSR cross_attn

Reference 图像
  → DINOv2
  → ref_z
  → 新增 ref_cross_attn

LQ latent + noisy HQ latent
  → LightningDiT
  → Flow Matching loss

预测 HQ latent
  → VAE decode
  → SR image
  → identity loss
```

也就是说，该模块没有直接替换 VOSR 原来的条件分支，而是在原有 LQ 条件之外，额外加入 reference 条件分支。

---

## 1. `addon_stageE.py`

这是 Stage-E 训练的主控模块。

核心类：

```python
class RefLDMVOSRStageEAddon
```

主要功能：

```text
1. 从 batch 中读取 reference 图像；
2. 编码 LQ/HQ 图像为 latent；
3. 提取 reference 图像的 DINOv2 特征；
4. 构造 z = {"lq": lq_feature, "ref": ref_z}；
5. 根据配置决定是否计算 identity loss；
6. 调用 VOSR 的 ref-aware flow matching loss。
```

关键逻辑：

```python
if self.use_ref_attention:
    z = {"lq": lq_feature, "ref": ref_z}
```

这说明模型输入的条件被拆成两路：

```text
lq_feature → 原始 cross_attn
ref_z      → 新增 ref_cross_attn
```

这是当前改造最重要的部分。

---

## 2. `ref_attention_patch.py`

这是最核心的结构修改文件。

它通过 monkey patch 的方式修改原始 `LightningDiT`，为每个 DiT block 新增一个 reference cross-attention 分支。

原始 VOSR block 大致是：

```python
x = x + self.attn(...)
x = x + self.cross_attn(x, z)
x = x + self.mlp(...)
```

修改后变成：

```python
x = x + self.attn(...)
x = x + self.cross_attn(x, lq_z)
x = x + gate * scale * self.ref_cross_attn(x, ref_z)
x = x + self.mlp(...)
```

也就是：

```text
原 cross_attn：继续处理 LQ-DINO 条件
新增 ref_cross_attn：专门处理 Reference-DINO 条件
```

核心代码：

```python
if self.z_dims is not None and z is not None:
    x = x + self.cross_attn(x, z)

if getattr(self, "use_ref_cross_attn", False) and (ref_z is not None or ref_kv is not None):
    ref_out = self.ref_cross_attn(x, cond=ref_z, kv_cache=ref_kv)
    gate = torch.sigmoid(self.ref_attn_gate)
    scale = float(getattr(self, "ref_attn_rescale", 0.1))
    x = x + gate * scale * ref_out
```

这里使用的是：

```python
gate = torch.sigmoid(self.ref_attn_gate)
```

因此即使：

```yaml
ref_attn_init_gate: 0.0
```

也有：

```text
sigmoid(0) = 0.5
```

所以 reference 分支不是完全关闭的。

不过 `CachedRefCrossAttention` 中：

```python
nn.init.zeros_(self.proj.weight)
nn.init.zeros_(self.proj.bias)
```

说明 ref attention 的最终投影层一开始输出为 0。这样做的好处是：训练初始时模型行为接近原 VOSR，不会突然破坏原模型输出。随后 `proj` 会先获得梯度，逐渐激活 ref 分支。

---

## 3. `vosr_stageE_patch.py`

这个文件扩展了 VOSR 的训练 loss 和采样逻辑，使其支持：

```python
z = {
    "lq": lq_z,
    "ref": ref_z
}
```

核心函数：

```python
loss_fm_stageD()
loss_fm_return_extra_stageD()
sample_multistep_fm_ref()
```

虽然名字里仍然叫 `stageD`，但实际也被 Stage-E 使用。

训练时，它保持原 VOSR 的 Flow Matching 逻辑：

```python
z_t = (1.0 - t) * hq + t * eps
v_target = eps - hq
v_pred = model(inp, t, z=z_mixed)
loss = ((v_pred - v_target) ** 2).mean()
```

同时扩展 CFG 条件：

```text
full condition:
  z = {"lq": lq_z, "ref": ref_z}

weak condition:
  z = {"lq": 0, "ref": 0}
```

采样时使用：

```python
v = v_weak + self.cfg_scale * (v_cond - v_weak)
```

因此它保留了原 VOSR 的 CFG 推理方式，同时让 reference 条件参与采样。

---

## 4. `features.py`

这个文件负责 reference 图像的 DINOv2 特征提取。

核心类：

```python
class RefFeatureExtractor
```

主要作用：

```text
1. 输入 ref 图像；
2. 将 [B,K,3,H,W] 展平成 [B*K,3,H,W]；
3. 送入 DINOv2；
4. 提取指定层特征；
5. 再 reshape 回 [B,K,N,C]。
```

这样可以保证多张 reference 的特征顺序不会乱。

---

## 5. `latents.py`

这个文件负责图像和 latent 之间的编码/解码。

核心类：

```python
class RefLatentEncoder
```

主要功能：

```text
1. encode LQ image；
2. encode HQ image；
3. encode reference image；
4. decode predicted HQ latent；
5. 兼容 sd2 VAE 和 qwen AE 的 latent mean/std。
```

identity loss 需要先把预测 latent decode 成图像，因此 `latents.py` 会被 `losses_stageE.py` 调用。

---

## 6. `losses_stageE.py`

这是 Stage-E 的身份损失模块。

核心类：

```python
class TimestepScaledIdentityLoss
```

支持：

```yaml
identity_target: gt
identity_target: ref
identity_target: both
```

含义分别是：

```text
gt   ：预测 SR 图像与 GT 图像做 identity loss
ref  ：预测 SR 图像与 reference 图像做 identity loss
both ：同时约束 GT identity 和 Ref identity
```

其核心逻辑是：

```python
loss_id = id_loss * scale
```

其中：

```python
scale = (1 - t) ** gamma
```

也就是 timestep-scaled identity loss。

这意味着 identity loss 在不同扩散/flow 时间步上的权重不同。通常越接近最终图像，identity loss 权重越大。

---

## 7. `identity_loss.py`

这个文件封装了 ArcFace / InsightFace 风格的人脸身份损失。

它通过：

```yaml
identity_model_path: /data/.../insightface_webface_r50.onnx
```

加载身份特征提取模型，然后计算 SR 图像和目标图像之间的身份距离。

在训练中用于：

```text
提升 IDS_ref 或 IDS_gt
```

---

## 8. `eval_refaware.py`

这个文件用于评估阶段构造 reference-aware condition。

核心作用是：在推理/评估时，从 FFHQ-Ref 的 csv 中读取 reference 图像，然后提取 reference feature，并构造：

```python
z_pack = {
    "lq": lq_feature,
    "ref": ref_feature
}
```

如果没有这个文件，训练时可能用了 reference，但 eval 时仍然只用 LQ 条件，导致看不出 ref 模块效果。

---

## 9. `eval.py`

这个文件是 FFHQ-Ref 评估入口相关逻辑。

主要作用：

```text
1. 读取 FFHQ-Ref 测试 csv；
2. 加载 LQ / HQ / Ref；
3. 调用 ref-aware sampling；
4. 保存 SR 结果；
5. 计算指标。
```

具体指标是否完整取决于主训练脚本如何调用它。

---

## 10. `addon.py`、`losses.py`、`vosr_patch.py`

早期版本。

其中：

```text
addon.py
losses.py
vosr_patch.py
```

主要对应较早的 Ref-LDM 接入方式，偏向 identity loss 或简单 feature 融合。

当前如果你使用的是 Stage-E ref attention，核心应优先看：

```text
addon_stageE.py
ref_attention_patch.py
vosr_stageE_patch.py
losses_stageE.py
features.py
latents.py
eval_refaware.py
```

---

## 当前代码实现的训练逻辑

第一阶段：

```yaml
freeze_backbone: true
train_ref_only: true
use_identity_loss: false
```

实际训练的是：

```text
ref_cross_attn
ref_attn_gate
```

目标是让 reference attention 先学会接入。

第二阶段：

```yaml
freeze_backbone: true
train_ref_only: true
use_identity_loss: true
identity_target: ref
```

仍然只训练 reference 分支，同时加入 ref identity loss，使输出更接近参考图身份。

第三阶段：

```yaml
freeze_backbone: false
train_ref_only: false
identity_target: both
```

解冻主干，用极小学习率联合微调，使超分质量和 identity 保持同时优化。

---

## 需要注意的点

当前代码里有几个需要注意的地方。

第一，`ref_attention_patch.py` 的注释里有些地方还写着：

```text
gate = tanh(...)
```

但实际代码已经改成：

```python
gate = torch.sigmoid(self.ref_attn_gate)
```

所以应以实际代码为准。

第二，`ref_cross_attn.proj` 是零初始化的。因此训练初期 reference 分支输出为 0，这是有意设计，用于保护原始 VOSR 输出。

第三，`vosr_stageE_patch.py` 中函数名仍然叫：

```python
patch_vosr_stageD
loss_fm_stageD
loss_fm_return_extra_stageD
```

但实际也服务于 Stage-E。命名上容易误解，但逻辑上是支持 Stage-E 的。

第四，三阶段 yml 需要是完整配置文件，不能只写差异项，否则训练脚本可能缺少：

```text
resolution
patch_size
dim
depth
ae_path
pretrained_ckpt
dataset_type
ffhq_ref_train_csv
dinov2_local_repo
```

---

## 总结

这个 `refldm_vosr` 模块实现的是：

```text
在 VOSR 原有 LQ 条件超分框架上，
额外加入 Ref-LDM 风格的 reference attention 和 identity supervision。
```

最核心的实现是：

```python
x = x + self.cross_attn(x, lq_z)
x = x + gate * scale * self.ref_cross_attn(x, ref_z)
```
