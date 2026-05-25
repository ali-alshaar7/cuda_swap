"""
Allocates ~1.5x the GPU's total VRAM in float32 tensors, keeps them all alive,
then does an elementwise sum across all of them and verifies the result.

Without cuda_swap: dies with CUDA out of memory.
With cuda_swap preloaded: completes successfully (slowly, due to host swapping).
"""

import os
import sys
import threading
import time
import torch


def read_proc_meminfo():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            info[k.strip()] = int(v.split()[0]) * 1024  # kB -> bytes
    return info


def effective_available_ram():
    """Returns available RAM respecting Docker/cgroup memory limits."""
    info = read_proc_meminfo()
    host_avail = info["MemAvailable"]

    # cgroup v2
    for path in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            val = open(path).read().strip()
            if val != "max":
                limit = int(val)
                used = info["MemTotal"] - host_avail
                return max(0, min(host_avail, limit - used))
        except (FileNotFoundError, ValueError):
            continue

    return host_avail


class MemoryMonitor:
    """Samples VRAM and system RAM usage in a background thread."""
    def __init__(self, interval=0.2):
        self.interval = interval
        self.peak_vram = 0
        self.peak_ram_used = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        baseline = read_proc_meminfo()
        self._ram_total = baseline["MemTotal"]
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
                used = m["MemTotal"] - m["MemAvailable"]
                self.peak_ram_used = max(self.peak_ram_used, used)
            except Exception:
                pass
            time.sleep(self.interval)

    def report(self):
        gb = 1024**3
        print(f"\n--- Memory peak report ---")
        print(f"  Peak VRAM (torch):   {self.peak_vram/gb:.2f} GB")
        print(f"  Peak system RAM used:{self.peak_ram_used/gb:.2f} GB  "
              f"(of {self._ram_total/gb:.1f} GB total)")


def main():
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device found")
        sys.exit(2)

    monitor = MemoryMonitor().start()
    props = torch.cuda.get_device_properties(0)
    total_vram = props.total_memory
    print(f"Device: {props.name}")
    print(f"Total VRAM: {total_vram / 1024**3:.2f} GB")

    # Target 2x VRAM, capped by what host RAM can actually absorb.
    # Only allocations beyond (VRAM - threshold) spill to host RAM.
    threshold = int(os.environ.get("CUDA_SWAP_THRESHOLD_MB", 512)) * 1024 * 1024
    host_avail = effective_available_ram()
    host_safety = 2 * 1024 * 1024 * 1024
    max_spill = max(0, host_avail - host_safety)
    max_total = (total_vram - threshold) + max_spill
    target_bytes = min(int(total_vram * 2.0), max_total)

    tensor_bytes = max(total_vram // 8, 256 * 1024 * 1024)  # at least 256 MB
    n = max(1, (target_bytes + tensor_bytes - 1) // tensor_bytes)
    elements = tensor_bytes // 4  # float32

    print(f"host_avail={host_avail/1024**3:.1f} GB  max_spill={max_spill/1024**3:.1f} GB")
    print(f"Allocating {n} tensors × {tensor_bytes / 1024**3:.2f} GB "
          f"= {n * tensor_bytes / 1024**3:.2f} GB total")

    tensors = []
    for i in range(n):
        t = torch.ones(elements, dtype=torch.float32, device="cuda")
        tensors.append(t)
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"  [{i+1:2d}/{n}] allocated, cuda_mem_allocated={allocated:.2f} GB")

    print("\nAll tensors allocated. Running elementwise sum...")

    # Sum all tensors into result; each addition launches a CUDA kernel.
    result = tensors[0].clone()
    for t in tensors[1:]:
        result.add_(t)

    torch.cuda.synchronize()

    expected = float(n)
    got = result.mean().item()
    print(f"\nResult mean = {got:.1f}  (expected {expected:.1f})")

    if abs(got - expected) > 0.5:
        monitor.stop()
        monitor.report()
        print("FAIL: math result is wrong!")
        sys.exit(1)

    monitor.stop()
    monitor.report()
    print("SUCCESS")


if __name__ == "__main__":
    main()
