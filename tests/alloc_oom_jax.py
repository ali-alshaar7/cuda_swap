"""
Allocates ~1.5x the GPU's total VRAM in float32 JAX arrays, keeps them alive,
sums them, and verifies the result.

Requires XLA_PYTHON_CLIENT_ALLOCATOR=platform so XLA calls cudaMalloc per
tensor rather than pre-allocating a fixed slab.

Without cuda_swap: dies with XlaRuntimeError / ResourceExhausted.
With cuda_swap preloaded: completes successfully (slowly, due to host swapping).
"""

import os
import subprocess
import sys
from utils import available_ram, MemoryMonitor


def main():
    import jax
    import jax.numpy as jnp

    devices = jax.devices("gpu")
    if not devices:
        print("ERROR: no GPU device found")
        sys.exit(2)

    device = devices[0]

    def jax_vram():
        try:
            return device.memory_stats().get("bytes_in_use", 0)
        except Exception:
            return 0

    monitor = MemoryMonitor(vram_fn=jax_vram).start()

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0]
        total_vram = int(out) * 1024 * 1024
    except Exception:
        total_vram = device.memory_stats().get("bytes_limit", 8 * 1024**3) // 2
        print("Warning: could not query VRAM via nvidia-smi, using estimate")

    print(f"Device: {device}")
    print(f"Total VRAM: {total_vram / 1024**3:.2f} GB")

    threshold  = int(os.environ.get("CUDA_SWAP_THRESHOLD_MB", 512)) * 1024 * 1024
    host_avail = available_ram()
    max_spill  = max(0, host_avail - 2 * 1024**3)
    target     = min(int(total_vram * 1.5), (total_vram - threshold) + max_spill)

    tensor_bytes = max(total_vram // 8, 256 * 1024 * 1024)
    n        = max(1, (target + tensor_bytes - 1) // tensor_bytes)
    elements = tensor_bytes // 4

    print(f"host_avail={host_avail/1024**3:.1f} GB  max_spill={max_spill/1024**3:.1f} GB")
    print(f"Allocating {n} arrays × {tensor_bytes/1024**3:.2f} GB "
          f"= {n*tensor_bytes/1024**3:.2f} GB total")

    arrays = []
    for i in range(n):
        a = jnp.ones(elements, dtype=jnp.float32)
        a.block_until_ready()
        arrays.append(a)
        print(f"  [{i+1:2d}/{n}] live_vram≈{jax_vram()/1024**3:.2f} GB")

    print("\nAll arrays allocated. Running elementwise sum...")
    result = arrays[0]
    for a in arrays[1:]:
        result = result + a
    result.block_until_ready()

    got      = float(jnp.mean(result))
    expected = float(n)
    print(f"\nResult mean = {got:.1f}  (expected {expected:.1f})")

    monitor.stop()
    monitor.report(vram_label="jax")

    if abs(got - expected) > 0.5:
        print("FAIL: math result is wrong!")
        sys.exit(1)
    print("SUCCESS")


if __name__ == "__main__":
    main()
