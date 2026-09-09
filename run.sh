#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_EXE="$(readlink -f "$(command -v "${PYTHON_BIN}")")"
PYTHON_PREFIX="$(cd "$(dirname "${PYTHON_EXE}")/.." && pwd)"
GPU_LIST="${GPUS:-0,1,2,3}"
OUTPUT_ROOT="${PROJECT_ROOT}/output"
LOG_ROOT="${OUTPUT_ROOT}/logs"
DRY_RUN=0

if [[ ${1:-} == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -ne 0 ]]; then
  echo "Usage: GPUS=0,1,2,3 bash run.sh [--dry-run]" >&2
  exit 2
fi

IFS=',' read -r -a GPU_IDS <<<"${GPU_LIST}"
if [[ ${#GPU_IDS[@]} -ne 4 ]]; then
  echo "Exactly four GPU IDs are required, for example GPUS=0,1,2,3" >&2
  exit 2
fi

TASKS=(dataA_classification dataA_regression dataB_classification dataB_regression)
mkdir -p "${LOG_ROOT}" "${PROJECT_ROOT}/cache/pycache"
export PYTHONHASHSEED=42
export PYTHONPYCACHEPREFIX="${PROJECT_ROOT}/cache/pycache"
# PyTorch wheels may load the system libstdc++ before RDKit is imported. Put
# the selected conda environment's runtime libraries first so both packages use
# the same C++ runtime, independent of their Python import order.
export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import yaml
import torch
from rdkit import Chem
import torch_geometric

assert torch.cuda.is_available(), "CUDA is not available in the selected Python environment"
assert Chem.MolFromSmiles("CCO") is not None

tasks = ("dataA_classification", "dataA_regression", "dataB_classification", "dataB_regression")
for task in tasks:
    path = Path("configs") / f"{task}.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["seed"] == 42
    assert config["split"]["num_folds"] == 10
    assert config["training"]["epochs"] == 200
    assert config["training"]["batch_size"] == 128
    assert config["model"]["chirality_alpha"] == 2.0
    assert Path(config["data"]["data_dir"]).exists()
print("Configuration and dataset checks passed.")
PY

echo "Task assignment:"
for index in 0 1 2 3; do
  echo "  GPU ${GPU_IDS[$index]}: ${TASKS[$index]}"
done
if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "Dry run complete; training was not started."
  exit 0
fi

rm -f "${OUTPUT_ROOT}/PIPELINE_COMPLETE" "${OUTPUT_ROOT}/PIPELINE_FAILED"
pids=()
for index in 0 1 2 3; do
  task="${TASKS[$index]}"
  gpu="${GPU_IDS[$index]}"
  config="configs/${task}.yaml"
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
    # Checkpoints are required during training and final evaluation, but are not
    # retained in the reproducibility output.
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
for index in 0 1 2 3; do
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
  --output "${OUTPUT_ROOT}/chimera_summary.csv"
touch "${OUTPUT_ROOT}/PIPELINE_COMPLETE"
echo "All tasks complete. Final table: ${OUTPUT_ROOT}/chimera_summary.csv"
