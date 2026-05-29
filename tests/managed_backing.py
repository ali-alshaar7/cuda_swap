"""
Tests whether cuMemAllocManaged keeps host RAM consumed while pages are
resident in VRAM, or whether host physical pages are freed on migration.

Two runs:
  # regular allocation (baseline):
  python tests/managed_backing.py

  # forced managed (CUDA_SWAP_THRESHOLD_MB=0 makes every alloc managed):
  CUDA_SWAP_THRESHOLD_MB=0 LD_PRELOAD=build/cuda_swap.so python tests/managed_backing.py

In each run we:
  1. Record baseline host RAM and VRAM.
  2. Allocate a fixed chunk and touch every element from the GPU,
     forcing any migration to VRAM to complete.
  3. Sync and record host RAM + VRAM again.
  4. Report the delta for both.

If managed memory keeps a host backing store while pages are in VRAM,
the managed run will show VRAM up by ~ALLOC_GB and host RAM also up by ~ALLOC_GB.
If it's pure migration (host physical pages freed on GPU fault), host RAM
should return to baseline after the GPU touches all pages.
"""

import os, subprocess, time
import torch

ALLOC_GB = 2.0


def host_used_mb() -> float:
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            info[k.strip()] = int(v.split()[0]) * 1024
    return (info["MemTotal"] - info["MemAvailable"]) / 1024**2


def vram_used_mb() -> float:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return float(out.strip().splitlines()[0])


def main():
    managed = os.environ.get("CUDA_SWAP_THRESHOLD_MB") == "0"
    mode = "forced-managed (CUDA_SWAP_THRESHOLD_MB=0)" if managed else "regular cuMemAlloc"

    props = torch.cuda.get_device_properties(0)
    print(f"Device : {props.name}")
    print(f"Mode   : {mode}")
    print(f"Alloc  : {ALLOC_GB:.1f} GB")
    print()

    # Warm up the CUDA context so it doesn't skew the baseline.
    _ = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    time.sleep(0.5)

    host_before = host_used_mb()
    vram_before = vram_used_mb()
    print(f"Before alloc:  host RAM used = {host_before:.0f} MB   VRAM used = {vram_before:.0f} MB")

    n_elements = int(ALLOC_GB * 1024**3 / 4)  # float32
    t = torch.ones(n_elements, dtype=torch.float32, device="cuda")

    # Touch every element from the GPU to force all page faults to complete.
    # If pages were on host, they migrate to VRAM here.
    _ = t.sum()
    torch.cuda.synchronize()
    time.sleep(0.5)  # give the driver a moment to settle

    host_after = host_used_mb()
    vram_after = vram_used_mb()
    print(f"After  alloc:  host RAM used = {host_after:.0f} MB   VRAM used = {vram_after:.0f} MB")
    print()
    print(f"Delta host RAM : {host_after - host_before:+.0f} MB")
    print(f"Delta VRAM     : {vram_after - vram_before:+.0f} MB")

    del t
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
