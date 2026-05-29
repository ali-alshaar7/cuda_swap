"""
Generates the two-panel allocation figure for the blog post.
Data from actual docker run output (STEP lines, nvidia-smi VRAM, cgroup RAM).

5 GB workload, RTX 3070 Ti (7.65 GB VRAM), 6 GB Docker container.
Both runs post-sync (torch.cuda.synchronize before each reading).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── raw data from docker runs ──────────────────────────────────────────────────
labels = ["init", "alloc 1", "alloc 2", "alloc 3", "alloc 4", "alloc 5", "alloc 6", "sum done"]

# True free VRAM from nvidia-smi (post-sync) — GB
t512_vram   = [6.633, 5.651, 4.694, 3.737, 2.780, 1.823, 0.866, 0.078]
tall_vram   = [6.668, 5.643, 4.643, 3.768, 2.768, 1.768, 0.893, 0.074]

# Container RAM from /sys/fs/cgroup/memory.current — GB
# Normalised to each run's init reading so runs can be compared.
t512_cgroup_raw   = [1.616, 1.640, 1.641, 1.641, 1.640, 1.640, 1.640, 2.559]
tall_cgroup_raw   = [0.447, 0.456, 0.449, 0.450, 0.449, 0.449, 0.449, 0.587]

t512_cgroup  = [v - t512_cgroup_raw[0] for v in t512_cgroup_raw]
tall_cgroup  = [v - tall_cgroup_raw[0] for v in tall_cgroup_raw]

xs = np.arange(len(labels))

# ── colours ────────────────────────────────────────────────────────────────────
C512 = "#2563EB"   # blue  — threshold=512
CALL = "#D97706"   # amber — all-managed

# ── figure ─────────────────────────────────────────────────────────────────────
fig, (ax_vram, ax_ram) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.patch.set_facecolor("white")

for ax in (ax_vram, ax_ram):
    ax.set_facecolor("#fafafa")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8, zorder=0)

# ── top panel: free VRAM ───────────────────────────────────────────────────────
ax_vram.plot(xs, t512_vram, color=C512, linewidth=2.5, marker="o", markersize=6,
             label="CUDA_SWAP_THRESHOLD_MB=512", zorder=3)
ax_vram.plot(xs, tall_vram, color=CALL, linewidth=2.5, marker="o", markersize=6,
             linestyle="--", label="CUDA_SWAP_THRESHOLD_MB=999999 (all-managed)", zorder=3)

ax_vram.axhline(7.65, color="#6B7280", linewidth=1, linestyle=":",
                label="Total VRAM (7.65 GB)", zorder=2)

ax_vram.set_ylabel("Free VRAM (GB)", fontsize=11)
ax_vram.set_ylim(-0.3, 8.2)
ax_vram.legend(loc="upper right", fontsize=9, framealpha=0.95)
ax_vram.set_title(
    "Per-step memory — 5 GB workload, RTX 3070 Ti, 6 GB container\n"
    "(readings taken post torch.cuda.synchronize — pages committed before each sample)",
    fontsize=10, pad=8)

# annotation: why both lines look the same
ax_vram.annotate(
    "Both drop identically post-sync:\nregular commits pages immediately;\nmanaged commits on first GPU touch\n(fill kernel runs before we sample).",
    xy=(2, 4.694), xytext=(4.5, 6.3),
    fontsize=8, color="#374151",
    arrowprops=dict(arrowstyle="->", color="#9CA3AF", lw=1.0),
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
              edgecolor="#D1D5DB", alpha=0.95))

# ── bottom panel: container RAM delta ─────────────────────────────────────────
ax_ram.plot(xs, t512_cgroup, color=C512, linewidth=2.5, marker="o", markersize=6,
            label="threshold=512", zorder=3)
ax_ram.plot(xs, tall_cgroup, color=CALL, linewidth=2.5, marker="o", markersize=6,
            linestyle="--", label="all-managed", zorder=3)

ax_ram.set_ylabel("Container RAM Δ from init (GB)", fontsize=11)
ax_ram.set_ylim(-0.1, 1.15)

# annotate the spike
ax_ram.annotate(
    "+0.94 GB\n(clone can't fit\nin VRAM — regular\ntensors immovable)",
    xy=(7, t512_cgroup[7]), xytext=(5.2, 0.75),
    fontsize=8, color=C512,
    arrowprops=dict(arrowstyle="->", color=C512, lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF",
              edgecolor=C512, alpha=0.95))

ax_ram.annotate(
    "+0.14 GB\n(managed tensors\nyield VRAM for clone)",
    xy=(7, tall_cgroup[7]), xytext=(5.1, 0.28),
    fontsize=8, color=CALL,
    arrowprops=dict(arrowstyle="->", color=CALL, lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFBEB",
              edgecolor=CALL, alpha=0.95))

ax_ram.set_xticks(xs)
ax_ram.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)

# shared note
ax_ram.text(0, -0.095,
    "Container RAM normalised to each run's init reading "
    "(absolute baselines differ between docker runs due to host state).",
    fontsize=7.5, color="#6B7280")

plt.tight_layout(rect=[0, 0.02, 1, 1])

out = "/home/alia/projects/portfolio/static/images/cuda-swap-vram.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved → {out}")
