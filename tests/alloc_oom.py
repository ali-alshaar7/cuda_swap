"""
Allocates up to 2x the GPU's total VRAM in float32 tensors, keeps them all alive,
then does an elementwise sum across all of them and verifies the result.

Without cuda_swap: dies with CUDA out of memory.
With cuda_swap preloaded: completes successfully (slowly, due to host swapping).
"""

import os
import sys
import torch
from utils import available_ram, MemoryMonitor


def main():
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device found")
        sys.exit(2)

    monitor = MemoryMonitor().start()
    props = torch.cuda.get_device_properties(0)
    total_vram = props.total_memory
    print(f"Device: {props.name}")
    print(f"Total VRAM: {total_vram / 1024**3:.2f} GB")

    threshold  = int(os.environ.get("CUDA_SWAP_THRESHOLD_MB", 512)) * 1024 * 1024
    host_avail = available_ram()
    max_spill  = max(0, host_avail - 2 * 1024**3)
    target     = min(int(total_vram * 2.0), (total_vram - threshold) + max_spill)

    tensor_bytes = max(total_vram // 8, 256 * 1024 * 1024)
    n        = max(1, (target + tensor_bytes - 1) // tensor_bytes)
    elements = tensor_bytes // 4  # float32

    print(f"host_avail={host_avail/1024**3:.1f} GB  max_spill={max_spill/1024**3:.1f} GB")
    print(f"Allocating {n} tensors × {tensor_bytes/1024**3:.2f} GB "
          f"= {n*tensor_bytes/1024**3:.2f} GB total")

    tensors = []
    for i in range(n):
        t = torch.ones(elements, dtype=torch.float32, device="cuda")
        tensors.append(t)
        print(f"  [{i+1:2d}/{n}] cuda_mem_allocated="
              f"{torch.cuda.memory_allocated()/1024**3:.2f} GB")

    print("\nAll tensors allocated. Running elementwise sum...")
    result = tensors[0].clone()
    for t in tensors[1:]:
        result.add_(t)
    torch.cuda.synchronize()

    got      = result.mean().item()
    expected = float(n)
    print(f"\nResult mean = {got:.1f}  (expected {expected:.1f})")

    monitor.stop()
    monitor.report()

    if abs(got - expected) > 0.5:
        print("FAIL: math result is wrong!")
        sys.exit(1)
    print("SUCCESS")


if __name__ == "__main__":
    main()
