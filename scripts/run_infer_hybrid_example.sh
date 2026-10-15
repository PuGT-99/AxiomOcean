#!/usr/bin/env bash
# Hybrid 1d+5d schedule (only public entrypoint of this package).
set -euo pipefail

RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${RELEASE_ROOT}/models"
SCRIPTS_DIR="${RELEASE_ROOT}/scripts"

PYTHON_SCRIPT="${SCRIPTS_DIR}/infer_hybrid_rollout.py"
RUNTIME_MODULE="${SCRIPTS_DIR}/onnx_runtime.py"

ENCODER_1D="${ENCODER_1D:-${MODELS_DIR}/axiomocean_1d_encoder_backbone_fp32.onnx}"
DECODER_1D="${DECODER_1D:-${MODELS_DIR}/axiomocean_1d_decoder_head_chunk_fp32.onnx}"
ENCODER_5D="${ENCODER_5D:-${MODELS_DIR}/axiomocean_5d_encoder_backbone_fp32.onnx}"
DECODER_5D="${DECODER_5D:-${MODELS_DIR}/axiomocean_5d_decoder_head_chunk_fp32.onnx}"

# Required data roots (no machine-specific defaults).
: "${TEST_DATA_DIR:?Set TEST_DATA_DIR to the ocean initial-condition root}"
: "${ERA5_DATA_DIR:?Set ERA5_DATA_DIR to the normalised ERA5 root}"
: "${CLIMATOLOGY_DIR:?Set CLIMATOLOGY_DIR to the climatology root}"

RESULTS_DIR="${RESULTS_DIR:-${RELEASE_ROOT}/outputs/results_hybrid}"
NC_OUTPUT_DIR="${NC_OUTPUT_DIR:-${RELEASE_ROOT}/outputs/nc_hybrid}"

START_DATE="${START_DATE:-2020-01-01}"
END_DATE="${END_DATE:-2020-01-01}"
MAX_LEADTIME="${MAX_LEADTIME:-10}"
ONE_DAY_THROUGH_LEAD="${ONE_DAY_THROUGH_LEAD:-5}"
DECODER_CHUNK_SIZE="${DECODER_CHUNK_SIZE:-4096}"
SESSION_MODE="${SESSION_MODE:-switch}"
SAVE_LEADTIMES="${SAVE_LEADTIMES:-1,5,10}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ORT_GPU_ID="${ORT_GPU_ID:-0}"
export HDF5_USE_FILE_LOCKING="FALSE"
PYTHON="${PYTHON:-python3}"

for required_path in \
  "${PYTHON_SCRIPT}" "${RUNTIME_MODULE}" \
  "${ENCODER_1D}" "${ENCODER_1D}.data" "${DECODER_1D}" "${DECODER_1D}.data" \
  "${ENCODER_5D}" "${ENCODER_5D}.data" "${DECODER_5D}" "${DECODER_5D}.data" \
  "${TEST_DATA_DIR}" "${ERA5_DATA_DIR}" "${CLIMATOLOGY_DIR}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "ERROR: required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

mkdir -p "${RESULTS_DIR}" "${NC_OUTPUT_DIR}"

EXTRA_ARGS=()
[[ "${DEBUG:-1}" == "1" ]] && EXTRA_ARGS+=(--debug)
[[ "${FAIL_FAST:-1}" == "1" ]] && EXTRA_ARGS+=(--fail_fast)
[[ -n "${SAVE_LEADTIMES}" ]] && EXTRA_ARGS+=(--save_leadtimes "${SAVE_LEADTIMES}")

echo "Hybrid ONNX rollout ${START_DATE}..${END_DATE} max_lead=${MAX_LEADTIME}"

"${PYTHON}" "${PYTHON_SCRIPT}" \
  --encoder_1d "${ENCODER_1D}" \
  --decoder_1d "${DECODER_1D}" \
  --encoder_5d "${ENCODER_5D}" \
  --decoder_5d "${DECODER_5D}" \
  --runtime_module "${RUNTIME_MODULE}" \
  --decoder_chunk_size "${DECODER_CHUNK_SIZE}" \
  --ort_gpu_id "${ORT_GPU_ID}" \
  --one-day-through-lead "${ONE_DAY_THROUGH_LEAD}" \
  --session-mode "${SESSION_MODE}" \
  --test_data_dir "${TEST_DATA_DIR}" \
  --era5_data_dir "${ERA5_DATA_DIR}" \
  --climatology_dir "${CLIMATOLOGY_DIR}" \
  --output_dir "${RESULTS_DIR}" \
  --nc_output_dir "${NC_OUTPUT_DIR}" \
  --max-leadtime "${MAX_LEADTIME}" \
  --start_date "${START_DATE}" \
  --end_date "${END_DATE}" \
  "${EXTRA_ARGS[@]}"

echo "[Completed] NC under ${NC_OUTPUT_DIR}"
