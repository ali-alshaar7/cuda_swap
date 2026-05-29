"""
Compares allocation + compute cost for regular vs managed CUDA memory.

Allocates tensors that fit entirely in VRAM (~50% of total), runs matmuls,
and measures wall time. Run twice:

  # regular alloc (no cuda_swap):
  python tests/alloc_cost.py

  # forced managed alloc (threshold=0 means every alloc is managed):
  CUDA_SWAP_THRESHOLD_MB=0 LD_PRELOAD=build/cuda_swap.so python tests/alloc_cost.py
"""

import os, sys, time
import torch

def main():
    if not torch.cuda.is_available():
        sys.exit("ERROR: no CUDA device")

    props      = torch.cuda.get_device_properties(0)
    total_vram = props.total_memory
    gb         = 1024 ** 3
    managed    = os.environ.get("CUDA_SWAP_THRESHOLD_MB") == "0"

    print(f"Device : {props.name}  ({total_vram/gb:.1f} GB VRAM)")
    print(f"Mode   : {'forced-managed (CUDA_SWAP_THRESHOLD_MB=0)' if managed else 'regular (no cuda_swap)'}")

    # a @ b allocates a, b, and c simultaneously — use 25% of free VRAM each.
    free_vram  = torch.cuda.mem_get_info()[0]
    target     = int(free_vram * 0.25)
    dim        = int((target / 4) ** 0.5)
    dim        = (dim // 256) * 256
    alloc_size = dim * dim * 4
    print(f"Matrix : {dim}×{dim}  ({alloc_size/gb:.2f} GB)\n")

    N_WARMUP = 3
    N_TRIALS = 10

    def one_trial():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        a = torch.randn(dim, dim, device="cuda")
        b = torch.randn(dim, dim, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        del a, b, c
        return elapsed

    print(f"Warming up ({N_WARMUP} runs)...")
    for _ in range(N_WARMUP):
        one_trial()

    print(f"Timing {N_TRIALS} trials...")
    times = [one_trial() for _ in range(N_TRIALS)]

    mean = sum(times) / len(times)
    std  = (sum((t - mean)**2 for t in times) / (len(times) - 1)) ** 0.5
    print(f"\nResult: {mean*1000:.1f} ms ± {std*1000:.1f} ms  "
          f"(alloc + matmul + sync, {N_TRIALS} trials)")

if __name__ == "__main__":
    main()
