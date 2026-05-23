#!/usr/bin/env bash
# Run the OOM test with and without cuda_swap and verify the expected outcomes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$ROOT/build"
TEST_PY="$SCRIPT_DIR/alloc_oom.py"
SWAP_LIB="$BUILD_DIR/cuda_swap.so"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo "=== Building cuda_swap ==="
cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$BUILD_DIR" -j"$(nproc)"
echo ""

if [[ ! -f "$SWAP_LIB" ]]; then
    echo "ERROR: $SWAP_LIB not found after build"
    exit 1
fi

# ---------------------------------------------------------------------------
# Test 1: baseline — expect OOM (non-zero exit)
# ---------------------------------------------------------------------------
echo "=== Test 1: WITHOUT cuda_swap  (expect OOM / non-zero exit) ==="
if python "$TEST_PY"; then
    echo ""
    echo "FAIL: program succeeded without cuda_swap — try a smaller GPU or larger"
    echo "      tensor size so that it actually OOMs."
    exit 1
else
    echo ""
    echo "PASS: got non-zero exit (OOM as expected)"
fi

echo ""

# ---------------------------------------------------------------------------
# Test 2: with LD_PRELOAD — expect success
# ---------------------------------------------------------------------------
echo "=== Test 2: WITH cuda_swap     (expect SUCCESS) ==="
# Lower threshold to 256 MB so eviction kicks in aggressively.
# Log level 2 (INFO) so we can see swapping activity.
CUDA_SWAP_THRESHOLD_MB=256 CUDA_SWAP_LOG=2 LD_PRELOAD="$SWAP_LIB" python "$TEST_PY"
echo ""
echo "PASS: completed successfully with cuda_swap"
