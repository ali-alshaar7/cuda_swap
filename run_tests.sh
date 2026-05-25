#!/usr/bin/env bash
# Builds the Docker image and runs the benchmark suite inside a container
# with RAM capped at 10 GB so tests can't exhaust host system memory.
#
# Usage: ./run_tests.sh [N]   (N = trials per condition, default 3)

set -euo pipefail
N="${1:-3}"

IMAGE="cuda_swap_test"

echo "Building Docker image..."
docker build -t "$IMAGE" .

echo ""
echo "Running benchmark (RAM cap: 10 GB, N=$N trials)..."
docker run --rm \
    --gpus all \
    --memory=10g \
    --shm-size=1g \
    "$IMAGE" \
    bash tests/benchmark.sh "$N"
