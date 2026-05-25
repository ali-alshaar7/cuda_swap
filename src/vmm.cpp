#include "vmm.hpp"
#include "util.hpp"

#include <cuda.h>
#include <dlfcn.h>
#include <unordered_map>
#include <mutex>

// ---------------------------------------------------------------------------
// Driver function pointers — resolved via dlvsym to bypass our own dlsym hook.
// ---------------------------------------------------------------------------

using Fn_cuMemAlloc        = CUresult(*)(CUdeviceptr*, size_t);
using Fn_cuMemAllocManaged = CUresult(*)(CUdeviceptr*, size_t, unsigned int);
using Fn_cuMemFree         = CUresult(*)(CUdeviceptr);
using Fn_cuMemGetInfo      = CUresult(*)(size_t*, size_t*);
using Fn_cuCtxGetDevice    = CUresult(*)(CUdevice*);
using Fn_cuMemAdvise          = CUresult(*)(CUdeviceptr, size_t, CUmem_advise, CUdevice);
using Fn_cuMemPrefetchAsync   = CUresult(*)(CUdeviceptr, size_t, int, CUstream);

static Fn_cuMemAlloc          real_cuMemAlloc;
static Fn_cuMemAllocManaged   real_cuMemAllocManaged;
static Fn_cuMemFree           real_cuMemFree;
static Fn_cuMemGetInfo        real_cuMemGetInfo;
static Fn_cuCtxGetDevice      real_cuCtxGetDevice;
static Fn_cuMemAdvise         real_cuMemAdvise;
static Fn_cuMemPrefetchAsync  real_cuMemPrefetchAsync;

static std::once_flag g_init_flag;

static void* open_libcuda() {
    void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("libcuda.so",   RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("libcuda.so.1", RTLD_NOW);
    if (!h) CS_WARN("could not open libcuda.so: %s", dlerror());
    return h;
}

static void do_resolve() {
    using DlsymFn = void*(*)(void*, const char*);
    DlsymFn real_dlsym = (DlsymFn)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    if (!real_dlsym) real_dlsym = (DlsymFn)::dlsym(RTLD_NEXT, "dlsym");

    void* cuda = open_libcuda();
    auto R = [&](const char* sym) -> void* {
        void* p = cuda ? real_dlsym(cuda, sym) : nullptr;
        if (!p) p = real_dlsym(RTLD_NEXT, sym);
        CS_DEBUG("resolve %-40s -> %s", sym, p ? "OK" : "MISSING");
        return p;
    };

    real_cuMemAlloc        = (Fn_cuMemAlloc)        R("cuMemAlloc_v2");
    real_cuMemAllocManaged = (Fn_cuMemAllocManaged) R("cuMemAllocManaged");
    real_cuMemFree         = (Fn_cuMemFree)         R("cuMemFree_v2");
    real_cuMemGetInfo      = (Fn_cuMemGetInfo)      R("cuMemGetInfo_v2");
    real_cuCtxGetDevice    = (Fn_cuCtxGetDevice)    R("cuCtxGetDevice");
    real_cuMemAdvise         = (Fn_cuMemAdvise)        R("cuMemAdvise");
    real_cuMemPrefetchAsync  = (Fn_cuMemPrefetchAsync) R("cuMemPrefetchAsync");

    size_t ram_total = 0, ram_avail = 0;
    read_meminfo(&ram_total, &ram_avail);
    CS_INFO("cuda_swap active  threshold=%zu MB  max_host=%zu MB  "
            "(sys RAM: %zu MB total, %zu MB available)  log=%d",
            eviction_threshold_bytes()/(1024*1024),
            max_host_bytes()/(1024*1024),
            ram_total/(1024*1024),
            ram_avail/(1024*1024),
            log_level());
}

bool vmm_supported() { return real_cuMemAlloc && real_cuMemAllocManaged; }

// ---------------------------------------------------------------------------
// Registry — tracks whether each allocation is regular or managed.
// ---------------------------------------------------------------------------

struct AllocEntry {
    size_t size;
    bool   managed;
};

static std::mutex g_mutex;
static std::unordered_map<CUdeviceptr, AllocEntry> g_allocs;

static void ensure_init() {
    std::call_once(g_init_flag, do_resolve);
}


static size_t free_vram() {
    size_t f = 0, t = 0;
    if (real_cuMemGetInfo) real_cuMemGetInfo(&f, &t);
    return f;
}

CUresult vmm_memgetinfo(size_t* free, size_t* total) {
    ensure_init();
    if (!real_cuMemGetInfo) return CUDA_ERROR_NOT_SUPPORTED;
    return real_cuMemGetInfo(free, total);
}

// ---------------------------------------------------------------------------
// Allocate
// ---------------------------------------------------------------------------
CUresult vmm_alloc(CUdeviceptr* out, size_t size) {
    ensure_init();

    if (!vmm_supported()) return CUDA_ERROR_NOT_SUPPORTED;

    // Before falling back to managed, check current available RAM.
    // We read /proc/meminfo fresh each time so the check reflects actual system
    // pressure from page faults that have already occurred — the most accurate
    // signal we have without tracking per-page residency.
    if (free_vram() < eviction_threshold_bytes() + size) {
        size_t ram_total = 0, ram_avail = 0;
        read_meminfo(&ram_total, &ram_avail);
        const size_t safety = (size_t)2 * 1024 * 1024 * 1024;
        if (size + safety > ram_avail) {
            CS_WARN("host RAM too low: need %zu MB available=%zu MB safety=2GB — OOM",
                    size/(1024*1024), ram_avail/(1024*1024));
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
    }

    // Choose allocator based on current VRAM pressure.
    bool use_managed = free_vram() < eviction_threshold_bytes() + size;

    CUresult rc;
    if (!use_managed) {
        rc = real_cuMemAlloc(out, size);
        if (rc != CUDA_SUCCESS) {
            // VRAM full despite check — fall through to managed.
            CS_INFO("cuMemAlloc failed (rc=%d), falling back to managed", rc);
            use_managed = true;
        }
    }

    if (use_managed) {
        rc = real_cuMemAllocManaged(out, size, CU_MEM_ATTACH_GLOBAL);
        if (rc != CUDA_SUCCESS) {
            CS_WARN("cuMemAllocManaged(%zu MB) failed rc=%d", size/(1024*1024), rc);
            return rc;
        }
        // Prefer VRAM; only spill to host under genuine pressure.
        if (real_cuMemAdvise && real_cuCtxGetDevice) {
            CUdevice dev = 0;
            real_cuCtxGetDevice(&dev);
            real_cuMemAdvise(*out, size, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, dev);
            real_cuMemAdvise(*out, size, CU_MEM_ADVISE_SET_ACCESSED_BY, dev);
        }
        CS_INFO("managed alloc  %zu MB  free_vram=%zu MB",
                size/(1024*1024), free_vram()/(1024*1024));
    } else {
        CS_DEBUG("regular alloc  %zu MB  free_vram=%zu MB",
                 size/(1024*1024), free_vram()/(1024*1024));
    }

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_allocs[*out] = {size, use_managed};
    }

    return CUDA_SUCCESS;
}

// ---------------------------------------------------------------------------
// Free
// ---------------------------------------------------------------------------
CUresult vmm_free(CUdeviceptr ptr) {
    ensure_init();

    AllocEntry entry{};
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_allocs.find(ptr);
        if (it == g_allocs.end()) return CUDA_ERROR_INVALID_VALUE;
        entry = it->second;
        g_allocs.erase(it);
    }

    if (real_cuMemFree) real_cuMemFree(ptr);
    CS_DEBUG("vmm_free va=0x%llx  managed=%d", (unsigned long long)ptr, entry.managed);
    return CUDA_SUCCESS;
}
