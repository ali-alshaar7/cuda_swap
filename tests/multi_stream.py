"""
Runs multiple independent CUDA streams concurrently, each holding a large weight
matrix and streaming batches through it.  The combined live memory across all
streams exceeds VRAM, forcing cuda_swap to spill some weight tensors to host RAM
while the GPU continues computing on the others.

Without cuda_swap: OOM when allocating the weight matrices.
With cuda_swap preloaded: all streams complete and every result is verified.

Usage:
    LD_PRELOAD=build/cuda_swap.so python tests/multi_stream.py
"""

import os
import sys
import threading
import torch
from utils import available_ram, MemoryMonitor


def worker(stream_id: int, weight: torch.Tensor, n_steps: int,
           results: list, errors: list):
    """Runs n_steps matmuls on a dedicated CUDA stream and stores per-step means."""
    stream = torch.cuda.Stream()
    dim    = weight.shape[0]
    step_means = []

    with torch.cuda.stream(stream):
        for step in range(n_steps):
            x   = torch.randn(256, dim, device="cuda")
            out = x @ weight
            stream.synchronize()
            step_means.append(out.mean().item())

    results[stream_id] = step_means


def main():
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device found")
        sys.exit(2)

    props      = torch.cuda.get_device_properties(0)
    total_vram = props.total_memory
    gb         = 1024 ** 3
    print(f"Device: {props.name}")
    print(f"Total VRAM: {total_vram/gb:.2f} GB")

    # Size each weight matrix so N streams together exceed VRAM.
    # Each weight is (dim × dim) float32.
    threshold  = int(os.environ.get("CUDA_SWAP_THRESHOLD_MB", 512)) * 1024 * 1024
    host_avail = available_ram()
    max_spill  = max(0, host_avail - 2 * gb)
    total_target = min(total_vram * 1.5, (total_vram - threshold) + max_spill)

    n_streams = 4
    weight_bytes = int(total_target) // n_streams
    # Round dim down to nearest 256 for efficient matmul
    dim = int((weight_bytes / 4) ** 0.5)
    dim = (dim // 256) * 256
    dim = max(dim, 256)
    weight_bytes = dim * dim * 4

    print(f"Streams: {n_streams}  weight per stream: {weight_bytes/gb:.2f} GB  "
          f"(dim={dim})  total: {n_streams*weight_bytes/gb:.2f} GB")
    print(f"host_avail={host_avail/gb:.1f} GB  max_spill={max_spill/gb:.1f} GB")

    monitor = MemoryMonitor().start()

    print("\nAllocating weight matrices...")
    weights = []
    for i in range(n_streams):
        w = torch.randn(dim, dim, device="cuda")
        weights.append(w)
        print(f"  stream {i}: weight allocated  "
              f"cuda_mem={torch.cuda.memory_allocated()/gb:.2f} GB")

    n_steps = 5
    results = [None] * n_streams
    errors  = []

    print(f"\nLaunching {n_streams} concurrent streams ({n_steps} steps each)...")
    threads = [
        threading.Thread(
            target=worker,
            args=(i, weights[i], n_steps, results, errors),
        )
        for i in range(n_streams)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    torch.cuda.synchronize()
    monitor.stop()
    monitor.report()

    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        sys.exit(1)

    print("\nPer-stream results (mean output per step):")
    all_ok = True
    for i, step_means in enumerate(results):
        if step_means is None:
            print(f"  stream {i}: NO RESULT")
            all_ok = False
            continue
        # weights are randn, inputs are randn → output mean ≈ 0 by CLT
        ok = all(abs(m) < 5.0 for m in step_means)
        status = "OK" if ok else "FAIL (mean out of range)"
        means_str = "  ".join(f"{m:+.3f}" for m in step_means)
        print(f"  stream {i}: [{means_str}]  {status}")
        if not ok:
            all_ok = False

    if not all_ok:
        print("\nFAIL")
        sys.exit(1)
    print("\nSUCCESS")


if __name__ == "__main__":
    main()
