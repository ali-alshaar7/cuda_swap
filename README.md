# cuda_swap

`LD_PRELOAD` library that prevents CUDA OOM by transparently spilling GPU memory to host RAM. Works with any CUDA program — no code changes.

## How it works

Intercepts `cudaMalloc`/`cuMemAlloc`. While VRAM is above the threshold, allocations use regular device memory. When VRAM runs low, it switches to `cuMemAllocManaged` — CUDA unified memory — which the driver automatically pages between GPU and RAM via hardware page faults.

## Build

```bash
cmake -S . -B build -DCUDA_INCLUDE_DIR=/usr/local/cuda/include
cmake --build build -j$(nproc)
# produces build/cuda_swap.so
```

## Usage

```bash
LD_PRELOAD=/path/to/cuda_swap.so python train.py
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `CUDA_SWAP_THRESHOLD_MB` | `512` | Free VRAM floor (MB). Below this, new allocations use unified memory. |
| `CUDA_SWAP_MAX_HOST_MB` | auto | Cap on host RAM used as overflow. Defaults to available RAM minus 1 GB. |
| `CUDA_SWAP_LOG` | `1` | Verbosity: `0`=off, `1`=warn, `2`=info, `3`=debug. |

## Test

Allocates 2× VRAM in tensors, sums them, verifies the result.

```bash
pip install torch
LD_PRELOAD=build/cuda_swap.so python tests/alloc_oom.py
```
