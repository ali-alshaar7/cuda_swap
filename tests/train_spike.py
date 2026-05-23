"""
Simulates a training step where activations spike beyond VRAM during forward/backward.
Uses simple linear layers (no attention) for fast execution.

Without cuda_swap: OOMs during forward pass.
With cuda_swap preloaded: completes all steps successfully.
"""

import sys
import threading
import time
import torch
import torch.nn as nn


def read_proc_meminfo():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            info[k.strip()] = int(v.split()[0]) * 1024
    return info


class MemoryMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.peak_vram = 0
        self.peak_ram_used = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._ram_total = read_proc_meminfo()["MemTotal"]
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _run(self):
        while not self._stop.is_set():
            try:
                vram = torch.cuda.memory_allocated()
                self.peak_vram = max(self.peak_vram, vram)
                m = read_proc_meminfo()
                self.peak_ram_used = max(self.peak_ram_used, m["MemTotal"] - m["MemAvailable"])
            except Exception:
                pass
            time.sleep(self.interval)

    def report(self):
        gb = 1024**3
        print(f"\n--- Memory peak report ---")
        print(f"  Peak VRAM (torch):   {self.peak_vram/gb:.2f} GB")
        print(f"  Peak system RAM used:{self.peak_ram_used/gb:.2f} GB"
              f"  (of {self._ram_total/gb:.1f} GB total)")


def main():
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device found")
        sys.exit(2)

    props = torch.cuda.get_device_properties(0)
    total_vram = props.total_memory
    gb = 1024**3
    print(f"Device: {props.name}")
    print(f"Total VRAM: {total_vram/gb:.2f} GB")

    in_features  = 8192
    out_features = 8192

    # Size batch so each activation tensor is ~2 GB individually (safe to allocate),
    # but with 3 hidden layers alive simultaneously the peak exceeds VRAM.
    # peak ≈ model(0.5 GB) + input(2 GB) + 3×hidden(2 GB each) = 8.5 GB > VRAM
    tensor_target = 2 * gb  # 2 GB per activation tensor
    batch_size = tensor_target // (out_features * 4)

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
    monitor = MemoryMonitor().start()

    n_steps = 3
    prev_loss = None
    print(f"\nTraining for {n_steps} steps...")

    for step in range(n_steps):
        x = torch.randn(batch_size, in_features, device="cuda")
        y = torch.randn(batch_size, 1, device="cuda")

        optimizer.zero_grad()
        out = model(x)
        loss = (out - y).pow(2).mean()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        vram_now = torch.cuda.memory_allocated() / gb
        print(f"  step {step+1}/{n_steps}  loss={loss.item():.4f}  vram={vram_now:.2f} GB")
        prev_loss = loss.item()

    monitor.stop()
    monitor.report()

    if prev_loss is None or not (0 <= prev_loss < 1e6):
        print("FAIL")
        sys.exit(1)

    print("SUCCESS")


if __name__ == "__main__":
    main()
