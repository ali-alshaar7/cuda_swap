"""
Simulates a training step where activations spike beyond VRAM during forward/backward.
Uses simple linear layers (no attention) for fast execution.

Without cuda_swap: OOMs during forward pass.
With cuda_swap preloaded: completes all steps successfully.
"""

import os
import sys
import torch
import torch.nn as nn
from utils import available_ram, MemoryMonitor


def main():
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device found")
        sys.exit(2)

    props      = torch.cuda.get_device_properties(0)
    total_vram = props.total_memory
    gb         = 1024 ** 3
    print(f"Device: {props.name}")
    print(f"Total VRAM: {total_vram/gb:.2f} GB")

    in_features  = 8192
    out_features = 8192

    threshold  = int(os.environ.get("CUDA_SWAP_THRESHOLD_MB", 512)) * 1024 * 1024
    host_avail = available_ram()
    max_spill  = max(0, host_avail - 2 * gb)
    # Peak spill ≈ 3 activation tensors beyond VRAM threshold
    max_tensor = min(2 * gb, (total_vram - threshold + max_spill) // 3)
    batch_size = max(1, int(max_tensor) // (out_features * 4))
    print(f"host_avail={host_avail/gb:.1f} GB  max_spill={max_spill/gb:.1f} GB")

    model = nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(),
        nn.Linear(out_features, out_features),
        nn.ReLU(),
        nn.Linear(out_features, 1),
    ).cuda()

    model_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"Model: {in_features}→{out_features}→1  params={model_mb:.0f} MB")
    print(f"Batch size: {batch_size}  "
          f"(activation spike ≈ {batch_size * out_features * 4 / gb:.2f} GB)")

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    monitor   = MemoryMonitor().start()

    n_steps   = 3
    prev_loss = None
    print(f"\nTraining for {n_steps} steps...")

    for step in range(n_steps):
        x = torch.randn(batch_size, in_features, device="cuda")
        y = torch.randn(batch_size, 1, device="cuda")

        optimizer.zero_grad()
        out  = model(x)
        loss = (out - y).pow(2).mean()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        print(f"  step {step+1}/{n_steps}  loss={loss.item():.4f}  "
              f"vram={torch.cuda.memory_allocated()/gb:.2f} GB")
        prev_loss = loss.item()

    monitor.stop()
    monitor.report()

    if prev_loss is None or not (0 <= prev_loss < 1e6):
        print("FAIL")
        sys.exit(1)
    print("SUCCESS")


if __name__ == "__main__":
    main()
