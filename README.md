# CHIMERA

## Environment

Create a fresh environment from the project root. The example below places the
environment and package caches in an `envs/` directory next to this repository;
the environment name is independent of the CHIMERA project name.

```bash
ENV_BASE="$(dirname "$PWD")/envs"
ENV_PREFIX="${ENV_BASE}/paper_repro_check"
mkdir -p "${ENV_BASE}/.pkgs_paper_repro_check" "${ENV_BASE}/.pip_paper_repro_check"

CONDA_PKGS_DIRS="${ENV_BASE}/.pkgs_paper_repro_check" \
PIP_CACHE_DIR="${ENV_BASE}/.pip_paper_repro_check" \
conda env create --prefix "${ENV_PREFIX}" --file environment.yml
```

The dependency versions in `environment.yml` are pinned to the versions used
for the reported experiments. `run.sh` also selects the C++ runtime from this
environment so that PyTorch and RDKit work regardless of import order.

## Train CHIMERA

The four main experiments are Data A/Data B classification and regression. They use seed 42 and the same 10-fold rotating protocol: one outer fold is used for testing, the next fold is used for validation, and the remaining eight folds are used for training. Each model is trained for 200 epochs with batch size 128. Classification checkpoints are selected by validation AUC, regression checkpoints by validation RMSE, and the test fold is evaluated only after checkpoint selection. CDAN is not used.

Run the four tasks in parallel on four GPUs:

```bash
PYTHON_BIN="${ENV_PREFIX}/bin/python" GPUS=0,1,2,3 bash run.sh
```

Each GPU runs one task. Training outputs are written to `output/`, including the final metrics for all ten folds. To check the configurations and GPU assignment without starting training:

```bash
PYTHON_BIN="${ENV_PREFIX}/bin/python" bash run.sh --dry-run
```

`run.sh` resolves all project, data, configuration, cache, log, and output paths
relative to its own location. The repository directory may therefore be renamed
or moved without editing the configurations.

A successful run creates `output/PIPELINE_COMPLETE` and
`output/chimera_summary.csv`. Before marking a task complete, the runner checks
the fixed-seed 8/1/1 split, all 200 training epochs, validation-only checkpoint
selection, and the absence of test overlap. Checkpoints are then removed; the
fold metrics and protocol records are retained.

The four training configurations are:

```text
configs/dataA_classification.yaml
configs/dataA_regression.yaml
configs/dataB_classification.yaml
configs/dataB_regression.yaml
```

## Generate Tables 4–7

The reported fold-level results are stored under `results/`:

```text
results/CHIMERA/          main CHIMERA results
results/COMPONENT_SPLIT/ component-based splits for Table 4
results/BASELINE/         baseline results for Tables 5 and 6
results/ABLATION/         ablation results for Table 7
```

Generate all four tables directly from the saved `fold_metrics.csv` files:

```bash
"${ENV_PREFIX}/bin/python" scripts/reproduce_tables.py
```

The generated tables are written to:

```text
results/TABLES/table4.csv
results/TABLES/table5.csv
results/TABLES/table6.csv
results/TABLES/table7.csv
```
