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

## Tests

Run all tests in an isolated Docker container (10 GB RAM cap):
```bash
./run_tests.sh
```

Or individually:

**`tests/alloc_oom.py`** — Allocates up to 2× VRAM in PyTorch tensors, keeps them all live, sums them and verifies the result. Without cuda_swap: OOM. With: completes successfully.
```bash
LD_PRELOAD=build/cuda_swap.so python tests/alloc_oom.py
```

**`tests/train_spike.py`** — Runs a training loop where model weights fit in VRAM but large activations during forward/backward cause a spike beyond VRAM. Without cuda_swap: OOM on first forward pass. With: all steps complete and loss is valid.
```bash
LD_PRELOAD=build/cuda_swap.so python tests/train_spike.py
```

**`tests/alloc_oom_jax.py`** — Same as alloc_oom but using JAX. Requires `XLA_PYTHON_CLIENT_ALLOCATOR=platform` so XLA calls `cudaMalloc` per tensor rather than managing a fixed pre-allocated slab.
```bash
XLA_PYTHON_CLIENT_ALLOCATOR=platform LD_PRELOAD=build/cuda_swap.so python tests/alloc_oom_jax.py
```

## Results

Benchmarked on RTX 3070 Ti (8 GB VRAM, 16 GB RAM), 3 trials each, run in a Docker container with a 10 GB RAM cap.

| Test | without cuda_swap | with cuda_swap |
|---|---|---|
| alloc_oom.py (2× VRAM PyTorch) | OOM (all trials) | **19.8s ± 0.8s** |
| train_spike.py (VRAM spike training) | OOM (all trials) | **289.9s ± 13.1s** |
| alloc_oom_jax.py (1.5× VRAM JAX) | OOM (all trials) | **28.5s ± 0.9s** |

The overhead comes from hardware page faulting between GPU and host RAM — no explicit copy operations, no code changes required.
