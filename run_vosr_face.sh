#!/usr/bin/env bash
# run_vosr_face.sh
# Run VOSR inference on three face LR datasets with 4 model variants
# (0.5B / 1.4B  x  multi-step / one-step). Includes ModelScope download.
# Usage:  bash run_vosr_face.sh
set -euo pipefail

# =========================================================
# 0. Paths (edit as needed)
# =========================================================
VOSR_ROOT="/data/wubin/VOSR-main"
CKPT_ROOT="${VOSR_ROOT}/preset/ckpts"
OUTPUT_ROOT="${VOSR_ROOT}/output"

# Three LR face directories to restore
INPUT_DIRS=(
    "/data/wubin/VOSR-main/FFHQ-Ref/test_images/moderate_degrad"
    "/data/wubin/VOSR-main/FFHQ-Ref/test_images/severe_degrad"
    "/data/wubin/VOSR-main/CelebA-Test-Ref/celeba_test_images/lq"
)

# Short tags matching INPUT_DIRS (used in output folder names)
INPUT_TAGS=(
    "FFHQ_moderate"
    "FFHQ_severe"
    "CelebA_lq"
)

# 4x super-resolution (change to 1 if your LR is already 512 and you only want restoration)
UPSCALE=4

# ---------------------------------------------------------------------------
# Smoke-test switches.
#
# Because models/lightningdit.py uses multiple @torch.compile decorators,
# the FIRST image after loading each model incurs several minutes of
# _inductor compilation — tqdm's ETA will look absurd (e.g. 80+ hours)
# until compile warm-up finishes. Use these switches to verify the full
# pipeline on a tiny subset before launching the full 12-job sweep.
#
#   SMOKE_TEST=1          only run the first dataset x first model combo
#   ONLY_DATASET=<tag>    only run this dataset tag (e.g. FFHQ_moderate)
#   ONLY_MODEL=<tag>      only run this model tag   (e.g. 0.5B_os_1step)
#
# Examples:
#   SMOKE_TEST=1 bash run_vosr_face.sh
#   ONLY_MODEL=0.5B_os_1step bash run_vosr_face.sh
# ---------------------------------------------------------------------------
SMOKE_TEST="${SMOKE_TEST:-0}"
ONLY_DATASET="${ONLY_DATASET:-}"
ONLY_MODEL="${ONLY_MODEL:-}"

# =========================================================
# 1. Download all weights from ModelScope mirror
# =========================================================
mkdir -p "${CKPT_ROOT}"

if ! command -v modelscope &>/dev/null; then
    echo "[INFO] Installing modelscope CLI ..."
    pip install -U modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

# VOSR_CKPT is one repo containing all 4 models + Qwen VAE + SD2.1 VAE
# + lightweight decoder + dinov2 cache.
#
# NOTE: modelscope download is idempotent and supports resume:
#   * completed files are verified and skipped
#   * partially-downloaded files resume via HTTP Range
#   * transient network errors are retried internally
# So we always call it — re-running after a crash will simply continue
# where it left off. Retry up to 4 times at the shell level for hard
# failures (DNS, auth, etc.) with exponential backoff.
echo "[INFO] Downloading / resuming VOSR weights from ModelScope to ${CKPT_ROOT} ..."
MAX_TRIES=4
SLEEP=2
for attempt in $(seq 1 ${MAX_TRIES}); do
    if modelscope download \
            --model LULALULALU/VOSR_CKPT \
            --local_dir "${CKPT_ROOT}"; then
        echo "[INFO] ModelScope download finished on attempt ${attempt}."
        break
    fi
    if [ "${attempt}" -eq "${MAX_TRIES}" ]; then
        echo "[ERROR] ModelScope download failed after ${MAX_TRIES} attempts."
        exit 1
    fi
    echo "[WARN] Download attempt ${attempt} failed, retrying in ${SLEEP}s ..."
    sleep "${SLEEP}"
    SLEEP=$((SLEEP * 2))
done

# Verify expected entries
for d in \
    "VOSR_0.5B_ms" "VOSR_0.5B_os" "VOSR_1.4B_ms" "VOSR_1.4B_os" \
    "Qwen-Image-vae-2d" "stable-diffusion-2-1-base" "torch_cache"; do
    if [ ! -e "${CKPT_ROOT}/${d}" ]; then
        echo "[WARN] Missing ${CKPT_ROOT}/${d}, please check the download."
    fi
done
[ ! -f "${CKPT_ROOT}/sd21_lwdecoder.pth" ] && echo "[WARN] Missing sd21_lwdecoder.pth"

# =========================================================
# 2. Inference configs: 4 (size x steps) variants
#    Format (pipe-separated so EXTRA can contain spaces):
#       <script>|<ckpt subdir>|<infer_steps>|<extra args>|<short tag>
# =========================================================
# Multi-step uses cfg_scale=0.5 (README's ScreenSR / face-friendly default).
# For very heavy degradation, try --cfg_scale -0.5 for more generative detail.
RUN_CONFIGS=(
    "inference_vosr.py|VOSR_0.5B_ms|25|--cfg_scale 0.5|0.5B_ms_25step"
    "inference_vosr.py|VOSR_1.4B_ms|25|--cfg_scale 0.5|1.4B_ms_25step"
    "inference_vosr_onestep.py|VOSR_0.5B_os|1||0.5B_os_1step"
    "inference_vosr_onestep.py|VOSR_1.4B_os|1||1.4B_os_1step"
)

# =========================================================
# 3. Run all combinations: 3 datasets x 4 models = 12 jobs
# =========================================================
mkdir -p "${OUTPUT_ROOT}"
LOG_DIR="${OUTPUT_ROOT}/_logs"
mkdir -p "${LOG_DIR}"

cd "${VOSR_ROOT}"

for i in "${!INPUT_DIRS[@]}"; do
    IN_DIR="${INPUT_DIRS[$i]}"
    IN_TAG="${INPUT_TAGS[$i]}"

    if [ ! -d "${IN_DIR}" ]; then
        echo "[SKIP] Input dir not found: ${IN_DIR}"
        continue
    fi

    # Dataset filter
    if [ -n "${ONLY_DATASET}" ] && [ "${ONLY_DATASET}" != "${IN_TAG}" ]; then
        continue
    fi
    # Smoke test: only first dataset
    if [ "${SMOKE_TEST}" = "1" ] && [ "${i}" != "0" ]; then
        break
    fi

    for cfg in "${RUN_CONFIGS[@]}"; do
        # Pipe-split so EXTRA may legitimately contain spaces (e.g. "--cfg_scale 0.5").
        IFS='|' read -r SCRIPT CKPT_NAME STEPS EXTRA TAG <<< "${cfg}"

        # Model filter
        if [ -n "${ONLY_MODEL}" ] && [ "${ONLY_MODEL}" != "${TAG}" ]; then
            continue
        fi
        # Smoke test: only first model
        if [ "${SMOKE_TEST}" = "1" ] && [ "${TAG}" != "0.5B_ms_25step" ]; then
            continue
        fi

        OUT_DIR="${OUTPUT_ROOT}/${IN_TAG}/${TAG}"
        mkdir -p "${OUT_DIR}"

        LOG_FILE="${LOG_DIR}/${IN_TAG}__${TAG}.log"

        echo "================================================================"
        echo "[RUN] dataset=${IN_TAG}  model=${TAG}"
        echo "      script=${SCRIPT}"
        echo "      ckpt  =${CKPT_ROOT}/${CKPT_NAME}"
        echo "      steps =${STEPS}  extra='${EXTRA}'"
        echo "      out   =${OUT_DIR}"
        echo "      log   =${LOG_FILE}"
        echo "================================================================"

        # shellcheck disable=SC2086
        python "${SCRIPT}" \
            -c "${CKPT_ROOT}/${CKPT_NAME}" \
            -i "${IN_DIR}" \
            -o "${OUT_DIR}" \
            -u "${UPSCALE}" \
            --infer_steps "${STEPS}" \
            ${EXTRA} 2>&1 | tee "${LOG_FILE}"
    done
done

echo
echo "[DONE] All inferences finished. Results under: ${OUTPUT_ROOT}"
