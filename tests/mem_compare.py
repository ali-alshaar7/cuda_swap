"""
Allocates a fixed amount of GPU memory and reports peak VRAM and host RAM.
Used to compare managed-only vs threshold (mixed) allocation strategies.

Set COMPARE_TARGET_GB to control total allocation (default: 1.5x VRAM).
CUDA_SWAP_THRESHOLD_MB controls which allocations use cuMemAllocManaged.

Per-step STEP lines go to stdout and are parseable for plotting:
  STEP <label> vram_free_gb=<f> container_gb=<f>
"""

import os
import subprocess
import sys
import threading
import time
import torch
from utils import available_ram, MemoryMonitor


# ── per-step snapshot helpers ──────────────────────────────────────────────────

def _nvidia_free_vram_gb() -> float:
    """True free VRAM from nvidia-smi — bypasses our cuMemGetInfo hook."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
        return int(out) / 1024  # MiB → GB
    except Exception:
        return float("nan")


def _container_gb() -> float:
    """Container-scoped RAM from cgroup (v2 then v1 fallback)."""
    for path in ("/sys/fs/cgroup/memory.current",
                 "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            v = open(path).read().strip()
            if v and v != "max":
                return int(v) / 1024**3
        except (FileNotFoundError, ValueError):
            pass
    return float("nan")


def snapshot(label: str):
    """Synchronise GPU, then print a parseable STEP line."""
    torch.cuda.synchronize()
    vf  = _nvidia_free_vram_gb()
    cm  = _container_gb()
    print(f"STEP {label} vram_free_gb={vf:.3f} container_gb={cm:.3f}", flush=True)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device found")
        sys.exit(2)

    props      = torch.cuda.get_device_properties(0)
    total_vram = props.total_memory
    gb         = 1024 ** 3

    print(f"Device: {props.name}")
    print(f"Total VRAM:  {total_vram/gb:.2f} GB")

    host_avail = available_ram()
    max_spill  = max(0, host_avail - 2 * gb)

    target_gb = float(os.environ.get("COMPARE_TARGET_GB", "0"))
    target = int(target_gb * gb) if target_gb > 0 else \
             min(int(total_vram * 1.5), total_vram + max_spill)

    threshold_mb = int(os.environ.get("CUDA_SWAP_THRESHOLD_MB", 512))
    print(f"CUDA_SWAP_THRESHOLD_MB={threshold_mb}")
    print(f"host_avail={host_avail/gb:.1f} GB  max_spill={max_spill/gb:.1f} GB")
    print(f"Target total: {target/gb:.2f} GB")

    tensor_bytes = max(total_vram // 8, 256 * 1024 * 1024)
    n        = max(1, (target + tensor_bytes - 1) // tensor_bytes)
    elements = tensor_bytes // 4

    print(f"Allocating {n} tensors × {tensor_bytes/gb:.2f} GB = {n*tensor_bytes/gb:.2f} GB")

    snapshot("init")

    tensors = []
    for i in range(n):
        t = torch.ones(elements, dtype=torch.float32, device="cuda")
        tensors.append(t)
        snapshot(f"alloc_{i+1}")

    print("\nRunning elementwise sum...")
    result = tensors[0].clone()
    for t in tensors[1:]:
        result.add_(t)

    snapshot("sum_done")

    got      = result.mean().item()
    expected = float(n)
    print(f"Result mean = {got:.1f}  (expected {expected:.1f})")

    if abs(got - expected) > 0.5:
        print("FAIL: math result is wrong!")
        sys.exit(1)
    print("SUCCESS")


if __name__ == "__main__":
    main()
