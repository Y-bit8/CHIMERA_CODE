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

## Data B preprocessing

The released `data/dataB/Ir-Ni reaction.xlsx` file contains 886 reactions. No
reaction is removed by the Data B classification or regression dataset builder,
and a numerical `ddG` value of zero is retained as a valid observation. In the
released file there are 0 exact-zero `ddG` values, 0 missing `ddG` values, and 0
missing `ee` values, so the input and final sample counts are both 886. Using the
published classification rule `abs(ee) >= 90`, the dataset contains 431 high-ee
and 455 low-ee reactions. The absolute-ee distribution is: minimum 1, first
quartile 70, median 89, third quartile 93, and maximum 99. Because neither
`ddG` nor `ee` contains an exact zero in the released source file, removing the
historical zero filter changes neither this distribution nor the class balance;
the code now explicitly retains any zero-valued observation.

The two continuous reaction conditions are temperature (`tem`) and reaction
time (`Time`). The five remaining condition columns (`metal`, `solvent`,
`additive`, `gm`, and `elsi`) are fixed category identifiers, not physical
quantities; their complete code-to-category mapping and category counts are
provided in `data/dataB/DataB_code_key.csv`.

## Data B condition-encoding sensitivity

The main reported experiments retain the historical seven-value input: two
continuous values followed by five scalar category identifiers. Because the
five identifiers are nominal and their numerical spacing has no chemical
meaning, the release also provides a fixed one-hot sensitivity analysis. It
keeps temperature and time unchanged and expands every released level of
`metal`, `solvent`, `additive`, `gm`, and `elsi`, producing a 155-dimensional
condition vector (2 continuous + 153 one-hot). The codebook is fixed before any
fold is formed, uses no target labels, and is identical for every fold and every
compared model.

Run the CHIMERA sensitivity experiment on two GPUs:

```bash
PYTHON_BIN="${ENV_PREFIX}/bin/python" GPUS=0,1 bash run_dataB_onehot.sh
```

The corresponding configurations are
`configs/dataB_classification_onehot.yaml` and
`configs/dataB_regression_onehot.yaml`. The scalar and one-hot CHIMERA results
use the same seed, rotating 8/1/1 folds, 200 epochs, model architecture,
optimizer, and validation-only checkpoint rule; only the condition
representation changes.

| Encoding | REC | F1 | AUC | R² | RMSE |
|---|---:|---:|---:|---:|---:|
| Seven scalar values | 0.773 ± 0.037 | 0.770 ± 0.039 | 0.857 ± 0.033 | 0.650 ± 0.092 | 1.572 ± 0.219 |
| Fixed 155-D one-hot | 0.778 ± 0.025 | 0.773 ± 0.028 | 0.867 ± 0.027 | 0.653 ± 0.099 | 1.564 ± 0.245 |

The same fixed 155-dimensional matrix was supplied to CHIMERA and all compared
models. Their one-hot results are archived under `results/CONDITION_ENCODING/`
in the same per-model, per-task `fold_metrics.csv` format as the main results.

Generate the encoding summary and the numerical one-hot-minus-original
differences from those fold files:

```bash
"${ENV_PREFIX}/bin/python" scripts/summarize_condition_encoding.py
```

The outputs remain under `results/CONDITION_ENCODING/` and are not part of
manuscript Tables 4–7.

Audit the released Data B file, configurations, codebook, encodings, and saved
sensitivity summaries without starting training:

```bash
"${ENV_PREFIX}/bin/python" scripts/audit_dataB_release.py
```

## Generate Tables 4–7

The reported fold-level results are stored under `results/`:

```text
results/CHIMERA/          main CHIMERA results
results/COMPONENT_SPLIT/ component-based splits for Table 4
results/BASELINE/         baseline results for Tables 5 and 6
results/ABLATION/         ablation results for Table 7
results/DataB_CHIMERA_NoAux328_ConditionsKept/
                         descriptor-free Data B control for Table 5
results/CONDITION_ENCODING/
                         Data B fixed one-hot fold results and response table
```

Only the ten-fold metric files needed to regenerate the tables are archived.
Table 5 includes the Data B `CHIMERA (w/o 328-D descriptors)` control. Table 6
reports descriptive mean paired differences only; it does not calculate or
report confidence intervals or hypothesis-test p-values.

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
