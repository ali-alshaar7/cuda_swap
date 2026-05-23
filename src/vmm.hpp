#pragma once
#include <cuda.h>
#include <cstddef>

// Hybrid allocator:
//   free_vram > threshold  →  cuMemAlloc   (regular, pure VRAM, zero RAM overhead)
//   free_vram < threshold  →  cuMemAllocManaged (driver pages overflow to host RAM)
//
// Only allocations made under VRAM pressure consume system RAM, so system RAM
// usage is proportional to the overflow, not total allocation size.

bool vmm_supported();
CUresult vmm_alloc(CUdeviceptr* out, size_t size);
CUresult vmm_free(CUdeviceptr ptr);
CUresult vmm_memgetinfo(size_t* free, size_t* total);
