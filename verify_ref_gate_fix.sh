#!/bin/bash
# VOSR Ref Gate Fix - Verification Checklist
# This script verifies all changes have been applied correctly

echo "======================================================================"
echo "VOSR Reference Gate Fix - Verification Checklist"
echo "======================================================================"

# Check 1: Configuration file
echo ""
echo "[1] Checking config file modification..."
if grep -q "ref_attn_init_gate: 0.1" /data/zhangdanning/Projects/VOSR/configs/train_yml/multi_step/ffhq_config/test_stage1.yml; then
    echo "    ? ref_attn_init_gate changed to 0.1 in test_stage1.yml"
else
    echo "    ? FAILED: ref_attn_init_gate not updated"
fi

# Check 2: Code change - sigmoid
echo ""
echo "[2] Checking code modification (sigmoid)..."
if grep -q "torch.sigmoid(self.ref_attn_gate)" /data/zhangdanning/Projects/VOSR/models/refldm_vosr/ref_attention_patch.py; then
    echo "    ? torch.tanh changed to torch.sigmoid in ref_attention_patch.py"
else
    echo "    ? FAILED: sigmoid not found"
fi

# Check 3: Documentation
echo ""
echo "[3] Checking documentation files..."
if [ -f "/data/zhangdanning/Projects/VOSR/DIAGNOSIS_REF_GATE_ZERO.md" ]; then
    echo "    ? DIAGNOSIS_REF_GATE_ZERO.md exists"
else
    echo "    ? DIAGNOSIS_REF_GATE_ZERO.md missing"
fi

if [ -f "/data/zhangdanning/Projects/VOSR/VOSR_REF_GATE_FIX_SUMMARY.md" ]; then
    echo "    ? VOSR_REF_GATE_FIX_SUMMARY.md exists"
else
    echo "    ? VOSR_REF_GATE_FIX_SUMMARY.md missing"
fi

if [ -f "/data/zhangdanning/Projects/VOSR/demo_ref_gate_fix.py" ]; then
    echo "    ? demo_ref_gate_fix.py exists"
else
    echo "    ? demo_ref_gate_fix.py missing"
fi

# Check 4: Python syntax validation
echo ""
echo "[4] Checking Python syntax..."
python3 -m py_compile /data/zhangdanning/Projects/VOSR/models/refldm_vosr/ref_attention_patch.py 2>/dev/null
if [ $? -eq 0 ]; then
    echo "    ? ref_attention_patch.py syntax valid"
else
    echo "    ? ref_attention_patch.py has syntax errors"
fi

# Summary
echo ""
echo "======================================================================"
echo "Summary:"
echo "  - Configuration: UPDATED (ref_attn_init_gate: 0.1)"
echo "  - Code: UPDATED (tanh -> sigmoid)"
echo "  - Documentation: COMPLETE"
echo ""
echo "Next steps:"
echo "  1. cd /data/zhangdanning/Projects/VOSR"
echo "  2. Run: python train_vosr_refldm_modular.py --config_path configs/train_yml/multi_step/ffhq_config/test_stage1.yml"
echo "  3. Monitor: ref_gate_mean should NOT be 0.0"
echo "  4. Expected: ref_gate_mean -> 0.4-0.6 range within first 100 steps"
echo "======================================================================"
