from .addon import RefLDMVOSRAddon
from .vosr_patch import patch_vosr_loss_return_extra

__all__ = ["RefLDMVOSRAddon", "patch_vosr_loss_return_extra"]

from .ref_attention_patch import patch_lightningdit_ref_attention
from .vosr_stageD_patch import patch_vosr_stageD
from .addon_stageD import RefLDMVOSRStageDAddon
from .addon_stageE import RefLDMVOSRStageEAddon

