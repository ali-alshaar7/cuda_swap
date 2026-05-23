"""
Allocates ~2x the GPU's total VRAM in float32 JAX arrays, keeps them alive,
sums them, and verifies the result.

Without cuda_swap: dies with XlaRuntimeError / ResourceExhausted (VRAM OOM).
With cuda_swap preloaded: completes successfully (slowly, due to host swapping).
"""

import sys
import threading
import time


def read_proc_meminfo():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            info[k.strip()] = int(v.split()[0]) * 1024  # kB -> bytes
    return info


class MemoryMonitor:
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
        import jax
        while not self._stop.is_set():
            try:
                d = jax.devices("gpu")[0]
                vram = d.memory_stats().get("bytes_in_use", 0)
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
        print(f"  Peak VRAM (jax):     {self.peak_vram/gb:.2f} GB")
        print(f"  Peak system RAM used:{self.peak_ram_used/gb:.2f} GB  "
              f"(of {self._ram_total/gb:.1f} GB total)")


def main():
    import jax
    import jax.numpy as jnp

    devices = jax.devices("gpu")
    if not devices:
        print("ERROR: no GPU device found")
        sys.exit(2)

    monitor = MemoryMonitor().start()

    device = devices[0]
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True
        ).strip().splitlines()[0]
        total_vram = int(out) * 1024 * 1024  # MiB -> bytes
    except Exception:
        try:
            # bytes_limit may be inflated by cuda_swap's cuMemGetInfo hook;
            # use it only as a last resort and halve it as a rough correction.
            total_vram = device.memory_stats()["bytes_limit"] // 2
        except Exception:
            total_vram = 8 * 1024**3
            print("Warning: could not query VRAM size, assuming 8 GB")

    print(f"Device: {device}")
    print(f"Total VRAM: {total_vram / 1024**3:.2f} GB")

    target_bytes = int(total_vram * 1.5)
    tensor_bytes = max(total_vram // 8, 256 * 1024 * 1024)
    n = (target_bytes + tensor_bytes - 1) // tensor_bytes
    elements = tensor_bytes // 4  # float32

    print(f"Allocating {n} arrays × {tensor_bytes / 1024**3:.2f} GB "
          f"= {n * tensor_bytes / 1024**3:.2f} GB total")

    arrays = []
    for i in range(n):
        a = jnp.ones(elements, dtype=jnp.float32)
        a.block_until_ready()
        arrays.append(a)
        try:
            vram_live = device.memory_stats().get("bytes_in_use", 0) / 1024**3
        except Exception:
            vram_live = float("nan")
        print(f"  [{i+1:2d}/{n}] allocated, live_vram≈{vram_live:.2f} GB")

    print("\nAll arrays allocated. Running elementwise sum...")

    result = arrays[0]
    for a in arrays[1:]:
        result = result + a

    result.block_until_ready()

    expected = float(n)
    got = float(jnp.mean(result))
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
