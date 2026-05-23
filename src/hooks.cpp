//
// cuda_swap — CUDA runtime + driver API interposer
//
// Usage: LD_PRELOAD=/path/to/cuda_swap.so <any cuda program>
//
// Env vars:
//   CUDA_SWAP_MAX_HOST_MB  hard cap on CPU RAM used as overflow (default 4096)
//   CUDA_SWAP_LOG          verbosity: 0=off 1=warn 2=info 3=debug (default 1)
//

#include "vmm.hpp"
#include "util.hpp"

#include <cuda.h>
#include <dlfcn.h>
#include <mutex>
#include <cstring>

// cuda.h #defines unversioned names onto _v2 variants. Undefine so we can
// export both names as separate interposer symbols.
#ifdef cuMemAlloc
#  undef cuMemAlloc
#endif
#ifdef cuMemFree
#  undef cuMemFree
#endif

// ---------------------------------------------------------------------------
// Runtime API types — defined manually, no cudart header needed.
// ---------------------------------------------------------------------------
typedef int cudaError_t;
using PFN_cudaFree = cudaError_t(*)(void*);

// ---------------------------------------------------------------------------
// Allocation hooks — redirect to managed memory allocator
// ---------------------------------------------------------------------------

extern "C" cudaError_t cudaMalloc(void** devPtr, size_t size) {
    CS_DEBUG("cudaMalloc(%zu bytes)", size);
    CUresult rc = vmm_alloc((CUdeviceptr*)devPtr, size);
    CS_DEBUG("cudaMalloc -> ptr=0x%llx rc=%d", (unsigned long long)(uintptr_t)*devPtr, rc);
    return (cudaError_t)rc;
}

extern "C" cudaError_t cudaFree(void* devPtr) {
    CS_DEBUG("cudaFree(0x%llx)", (unsigned long long)(uintptr_t)devPtr);
    CUresult rc = vmm_free((CUdeviceptr)(uintptr_t)devPtr);
    if (rc == CUDA_ERROR_INVALID_VALUE) {
        // Not our pointer — pass through to real cudaFree.
        static PFN_cudaFree real;
        if (!real) {
            static std::mutex m;
            std::lock_guard<std::mutex> lg(m);
            if (!real) real = (PFN_cudaFree)dlsym(RTLD_NEXT, "cudaFree");
        }
        return real ? real(devPtr) : 0;
    }
    return (cudaError_t)rc;
}

extern "C" CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
    CS_DEBUG("cuMemAlloc_v2(%zu bytes)", bytesize);
    return vmm_alloc(dptr, bytesize);
}

extern "C" CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    return cuMemAlloc_v2(dptr, bytesize);
}

extern "C" CUresult cuMemFree_v2(CUdeviceptr dptr) {
    CS_DEBUG("cuMemFree_v2(0x%llx)", (unsigned long long)dptr);
    CUresult rc = vmm_free(dptr);
    if (rc == CUDA_ERROR_INVALID_VALUE) {
        // Not our pointer — pass through.
        using Fn = CUresult(*)(CUdeviceptr);
        static Fn real;
        if (!real) {
            static std::mutex m;
            std::lock_guard<std::mutex> lg(m);
            if (!real) real = (Fn)dlsym(RTLD_NEXT, "cuMemFree_v2");
        }
        return real ? real(dptr) : rc;
    }
    return rc;
}

extern "C" CUresult cuMemFree(CUdeviceptr dptr) {
    return cuMemFree_v2(dptr);
}

extern "C" CUresult cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pPitch,
                                        size_t widthBytes, size_t height,
                                        unsigned int /*elementSize*/) {
    size_t pitch = (widthBytes + 511) & ~(size_t)511;
    *pPitch = pitch;
    return vmm_alloc(dptr, pitch * height);
}

// ---------------------------------------------------------------------------
// cuMemGetInfo hook — report available host RAM as additional "free" VRAM so
// that allocators (e.g. XLA's BFC) don't bail out before attempting cudaMalloc.
// Our cudaMalloc hook will then satisfy the request with managed memory.
// ---------------------------------------------------------------------------
extern "C" CUresult cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream /*stream*/) {
    CS_DEBUG("cuMemAllocAsync(%zu bytes) → redirecting to vmm_alloc", bytesize);
    return vmm_alloc(dptr, bytesize);
}

extern "C" CUresult cuMemGetInfo_v2(size_t* free, size_t* total) {
    CUresult rc = vmm_memgetinfo(free, total);
    if (rc == CUDA_SUCCESS && free && total) {
        size_t host_headroom = max_host_bytes();
        CS_DEBUG("cuMemGetInfo: real_free=%zu MB  adding host_headroom=%zu MB",
                 *free/(1024*1024), host_headroom/(1024*1024));
        *free  += host_headroom;
        *total += host_headroom;
    }
    return rc;
}

// ---------------------------------------------------------------------------
// dlsym hook — catches libraries that resolve CUDA symbols via
// dlopen()+dlsym(), which would otherwise bypass LD_PRELOAD.
//
// We only redirect explicit library-handle lookups. RTLD_NEXT/RTLD_DEFAULT
// go through the normal linker chain and already find our symbols first.
// Redirecting RTLD_NEXT would cause our own internal dlsym calls to loop.
// ---------------------------------------------------------------------------
extern "C" void* dlsym(void* handle, const char* symbol) {
    static void* (*real)(void*, const char*) = nullptr;
    if (!real) {
        static std::mutex m;
        std::lock_guard<std::mutex> lg(m);
        if (!real)
            real = (void*(*)(void*, const char*))
                   dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    }

    if (handle != RTLD_NEXT && handle != RTLD_DEFAULT && symbol) {
#define REDIRECT(sym) \
        if (strcmp(symbol, #sym) == 0) { CS_DEBUG("dlsym redirect: " #sym); return (void*)sym; }
        REDIRECT(cudaMalloc)
        REDIRECT(cudaFree)
        REDIRECT(cuMemAlloc_v2)
        REDIRECT(cuMemAlloc)
        REDIRECT(cuMemFree_v2)
        REDIRECT(cuMemFree)
        REDIRECT(cuMemGetInfo_v2)
        REDIRECT(cuMemAllocAsync)
#undef REDIRECT
    }

    return real ? real(handle, symbol) : nullptr;
}

