#!/usr/bin/env bash
# Benchmarks each test N times with and without cuda_swap.
# Each trial is a fresh subprocess for a cold allocator state.
# Usage: ./tests/benchmark.sh [N]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -f "${ROOT}/.venv/bin/python" ]; then
    PYTHON="${ROOT}/.venv/bin/python"
else
    PYTHON="$(command -v python3 || command -v python)"
fi
SO="${ROOT}/build/cuda_swap.so"
N="${1:-5}"

if [ ! -f "$SO" ]; then
    echo "cuda_swap.so not found — run: cmake --build build -j\$(nproc)"
    exit 1
fi

# Returns wall-clock seconds for one silent run of a command.
time_run() {
    local start end
    start=$(date +%s%N)
    "$@" > /dev/null 2>&1
    local rc=$?
    end=$(date +%s%N)
    echo "scale=3; ($end - $start) / 1000000000" | bc
    return $rc
}

run_bench() {
    local label="$1"; shift
    local trials=()
    local failed=0
    echo "  [$label]"
    for i in $(seq 1 "$N"); do
        local t
        if t=$(time_run "$@"); then
            trials+=("$t")
            printf "    trial %d: %.2fs\n" "$i" "$t"
        else
            echo "    trial $i: OOM / FAILED (expected without cuda_swap)"
            failed=$((failed + 1))
        fi
    done

    if [ ${#trials[@]} -eq 0 ]; then
        echo "    result: all $N trials failed (OOM as expected)"
        return
    fi

    # Compute mean and stddev via awk.
    printf '%s\n' "${trials[@]}" | awk -v n="${#trials[@]}" '
    { sum += $1; sumsq += $1*$1 }
    END {
        mean = sum/n
        std  = (n > 1) ? sqrt((sumsq - n*mean*mean)/(n-1)) : 0
        printf "    result: mean=%.2fs  stddev=%.2fs  (%d/%d succeeded)\n", mean, std, n, '"$N"'
    }'
}

echo "================================================================"
echo " cuda_swap benchmark  (N=$N trials per condition)"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo " VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
echo " RAM: $(grep MemTotal /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')"
echo "================================================================"

echo ""
echo "--- alloc_oom.py (2x VRAM allocation + sum) ---"
run_bench "without cuda_swap" "$PYTHON" "$SCRIPT_DIR/alloc_oom.py"
run_bench "with    cuda_swap" env LD_PRELOAD="$SO" "$PYTHON" "$SCRIPT_DIR/alloc_oom.py"

echo ""
echo "--- train_spike.py (training with VRAM spike) ---"
run_bench "without cuda_swap" "$PYTHON" "$SCRIPT_DIR/train_spike.py"
run_bench "with    cuda_swap" env LD_PRELOAD="$SO" "$PYTHON" "$SCRIPT_DIR/train_spike.py"

echo ""
echo "--- alloc_oom_jax.py (1.5x VRAM, JAX) ---"
run_bench "without cuda_swap" env XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    "$PYTHON" "$SCRIPT_DIR/alloc_oom_jax.py"
run_bench "with    cuda_swap" env XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    LD_PRELOAD="$SO" "$PYTHON" "$SCRIPT_DIR/alloc_oom_jax.py"

echo ""
echo "--- multi_stream.py (concurrent streams, total > VRAM) ---"
run_bench "without cuda_swap" "$PYTHON" "$SCRIPT_DIR/multi_stream.py"
run_bench "with    cuda_swap" env LD_PRELOAD="$SO" "$PYTHON" "$SCRIPT_DIR/multi_stream.py"

echo ""
echo "Done."
