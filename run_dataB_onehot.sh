#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_EXE="$(readlink -f "$(command -v "${PYTHON_BIN}")")"
PYTHON_PREFIX="$(cd "$(dirname "${PYTHON_EXE}")/.." && pwd)"
GPU_LIST="${GPUS:-0,1}"
OUTPUT_ROOT="${PROJECT_ROOT}/output/DataB_ConditionOneHot"
LOG_ROOT="${OUTPUT_ROOT}/logs"
DRY_RUN=0

if [[ ${1:-} == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -ne 0 ]]; then
  echo "Usage: GPUS=0,1 bash run_dataB_onehot.sh [--dry-run]" >&2
  exit 2
fi

IFS=',' read -r -a GPU_IDS <<<"${GPU_LIST}"
if [[ ${#GPU_IDS[@]} -ne 2 ]]; then
  echo "Exactly two GPU IDs are required, for example GPUS=0,1" >&2
  exit 2
fi

TASKS=(dataB_classification dataB_regression)
CONFIGS=(configs/dataB_classification_onehot.yaml configs/dataB_regression_onehot.yaml)
mkdir -p "${LOG_ROOT}" "${PROJECT_ROOT}/cache/pycache"
export PYTHONHASHSEED=42
export PYTHONPYCACHEPREFIX="${PROJECT_ROOT}/cache/pycache"
export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import yaml

for path in (Path("configs/dataB_classification_onehot.yaml"), Path("configs/dataB_regression_onehot.yaml")):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["seed"] == 42
    assert config["split"]["num_folds"] == 10
    assert config["training"]["epochs"] == 200
    assert config["training"]["batch_size"] == 128
    assert config["training"]["selection_source"] == "validation"
    assert config["data"]["condition_encoding"] == "fixed_one_hot"
    assert "drop_zero_ddg" not in config["data"]
    assert Path(config["data"]["data_dir"]).is_dir()
print("One-hot sensitivity configuration checks passed.")
PY

echo "Task assignment:"
for index in 0 1; do
  echo "  GPU ${GPU_IDS[$index]}: ${TASKS[$index]}"
done
if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "Dry run complete; training was not started."
  exit 0
fi

rm -f "${OUTPUT_ROOT}/PIPELINE_COMPLETE" "${OUTPUT_ROOT}/PIPELINE_FAILED"
pids=()
for index in 0 1; do
  task="${TASKS[$index]}"
  config="${CONFIGS[$index]}"
  gpu="${GPU_IDS[$index]}"
  result_dir="${OUTPUT_ROOT}/${task}"
  log="${LOG_ROOT}/${task}.log"
  (
    if [[ -f "${result_dir}/RUN_COMPLETE" ]]; then
      echo "SKIP ${task}: RUN_COMPLETE exists"
      exit 0
    fi
    echo "START ${task} on GPU ${gpu}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON_BIN}" scripts/run_task.py --config "${config}" --device cuda >"${log}" 2>&1
    status=$?
    if [[ ${status} -eq 0 ]]; then
      "${PYTHON_BIN}" scripts/check_protocol.py "${result_dir}" >>"${log}" 2>&1
      status=$?
    fi
    if [[ ${status} -ne 0 ]]; then
      echo "FAILED ${task}; see ${log}" >&2
      exit "${status}"
    fi
    "${PYTHON_BIN}" - "${result_dir}" <<'PY'
from pathlib import Path
import sys

for checkpoint in Path(sys.argv[1]).glob("fold_*/best_model.pt"):
    checkpoint.unlink()
PY
    touch "${result_dir}/RUN_COMPLETE"
    echo "DONE ${task}"
  ) &
  pids+=("$!")
done

failed=0
for index in 0 1; do
  if ! wait "${pids[$index]}"; then
    failed=1
  fi
done
if [[ ${failed} -ne 0 ]]; then
  touch "${OUTPUT_ROOT}/PIPELINE_FAILED"
  exit 1
fi

"${PYTHON_BIN}" scripts/collect_results.py \
  --root "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/chimera_onehot_summary.csv" \
  --datasets DataB
touch "${OUTPUT_ROOT}/PIPELINE_COMPLETE"
echo "Data B one-hot sensitivity experiments complete."
