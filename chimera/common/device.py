from __future__ import annotations

from dataclasses import dataclass
import os
import random
import numpy as np
import torch


@dataclass
class RuntimeContext:
    device: torch.device
    is_distributed: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def init_runtime(requested_device: str = "auto", seed: int = 42) -> RuntimeContext:
    set_seed(seed)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1

    if requested_device == "cpu":
        device = torch.device("cpu")
        backend = "gloo"
    elif torch.cuda.is_available():
        if is_distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if is_distributed and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend)

    return RuntimeContext(device=device, is_distributed=is_distributed, rank=rank, local_rank=local_rank, world_size=world_size)


def cleanup_runtime(ctx: RuntimeContext) -> None:
    if ctx.is_distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def move_to_device(obj, device: torch.device):
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if hasattr(obj, "to"):
        return obj.to(device)
    return obj
