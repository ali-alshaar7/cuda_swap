"""Shared helpers for cuda_swap test scripts."""

import threading
import time


def available_ram() -> int:
    """Returns available host RAM in bytes, respecting cgroup limits (Docker)."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            info[k.strip()] = int(v.split()[0]) * 1024

    host_avail = info["MemAvailable"]

    try:
        val = open("/sys/fs/cgroup/memory.max").read().strip()
        if val != "max":
            used = info["MemTotal"] - host_avail
            return max(0, min(host_avail, int(val) - used))
    except (FileNotFoundError, ValueError):
        pass

    return host_avail


class MemoryMonitor:
    """Polls peak VRAM and system RAM in a background thread.

    vram_fn: callable returning current VRAM bytes; defaults to torch.
    """
    def __init__(self, interval: float = 0.2, vram_fn=None):
        self.interval = interval
        self._vram_fn = vram_fn
        self.peak_vram = 0
        self.peak_ram_used = 0
        self._ram_total = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    self._ram_total = int(line.split()[1]) * 1024
                    break
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _vram(self) -> int:
        if self._vram_fn:
            return self._vram_fn()
        import torch
        return torch.cuda.memory_allocated()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.peak_vram = max(self.peak_vram, self._vram())
                with open("/proc/meminfo") as f:
                    info = {}
                    for line in f:
                        k, v = line.split(":")
                        info[k.strip()] = int(v.split()[0]) * 1024
                used = info["MemTotal"] - info["MemAvailable"]
                self.peak_ram_used = max(self.peak_ram_used, used)
            except Exception:
                pass
            time.sleep(self.interval)

    def report(self, vram_label: str = "torch"):
        gb = 1024 ** 3
        print(f"\n--- Memory peak report ---")
        print(f"  Peak VRAM ({vram_label}):".ljust(28) + f"{self.peak_vram/gb:.2f} GB")
        print(f"  Peak system RAM used:".ljust(28) +
              f"{self.peak_ram_used/gb:.2f} GB  (of {self._ram_total/gb:.1f} GB total)")
