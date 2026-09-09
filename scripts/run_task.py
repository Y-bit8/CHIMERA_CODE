#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

                                                                               
                                                                            
                                                                                  
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from chimera.common.config import load_config, save_json
from chimera.common.device import init_runtime, cleanup_runtime
from chimera.common.trainer import train_cross_validate


def main():
    parser = argparse.ArgumentParser(description="Run reviewer-safe CHIMERA training.")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    parser.add_argument("--data-dir", default=None, help="Override data directory")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"], help="Override device")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task_module = cfg["task_module"]
    task_type = cfg["task_type"]
    data_dir = Path(args.data_dir or cfg.get("data", {}).get("data_dir", "data/irni_regression"))
    output_dir = Path(args.output_dir or cfg.get("output_dir", f"results/{cfg['name']}"))
    device_name = args.device or cfg.get("device", "auto")

    ctx = init_runtime(device_name, seed=int(cfg.get("seed", 42)))
    try:
        build_mod = importlib.import_module(f"{task_module}.build_dataset")
        model_mod = importlib.import_module(f"{task_module}.model")
        dataset, metadata = build_mod.build_dataset(data_dir, cfg)
        split_groups = metadata.pop("_split_groups", None)
        if ctx.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_json(metadata, output_dir / "dataset_metadata.json")
            print(f"Loaded {len(dataset)} samples from {data_dir}")
            print(f"Writing results to {output_dir}")
        train_cross_validate(model_mod.Transformer, dataset, cfg, task_type, output_dir, ctx, split_groups=split_groups)
    finally:
        cleanup_runtime(ctx)


if __name__ == "__main__":
    main()
