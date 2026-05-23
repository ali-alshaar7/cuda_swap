#pragma once
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

// Log levels controlled by CUDA_SWAP_LOG env var (0=off, 1=warn, 2=info, 3=debug)
static inline int log_level() {
    static int lvl = -1;
    if (lvl < 0) {
        const char* e = getenv("CUDA_SWAP_LOG");
        lvl = e ? atoi(e) : 1;
    }
    return lvl;
}

#define CS_WARN(fmt, ...)  do { if (log_level() >= 1) fprintf(stderr, "[cuda_swap WARN] " fmt "\n", ##__VA_ARGS__); } while(0)
#define CS_INFO(fmt, ...)  do { if (log_level() >= 2) fprintf(stderr, "[cuda_swap INFO] " fmt "\n", ##__VA_ARGS__); } while(0)
#define CS_DEBUG(fmt, ...) do { if (log_level() >= 3) fprintf(stderr, "[cuda_swap DBG ] " fmt "\n", ##__VA_ARGS__); } while(0)

// Resolve a driver symbol once and cache it.
// Usage: auto real_fn = get_real<FnType>("cuMemAlloc_v2");
template<typename Fn>
static Fn get_real(const char* name) {
    void* sym = dlsym(RTLD_NEXT, name);
    if (!sym) {
        CS_WARN("dlsym failed for %s: %s", name, dlerror());
    }
    return reinterpret_cast<Fn>(sym);
}

// Free-VRAM floor that triggers proactive eviction to CPU.
// Override with CUDA_SWAP_THRESHOLD_MB (default 512 MB).
static inline size_t eviction_threshold_bytes() {
    static size_t v = 0;
    if (v == 0) {
        const char* e = getenv("CUDA_SWAP_THRESHOLD_MB");
        v = (size_t)(e ? atol(e) : 512) * 1024 * 1024;
    }
    return v;
}

// Read total and available system RAM from /proc/meminfo.
static inline void read_meminfo(size_t* total, size_t* available) {
    *total = *available = 0;
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return;
    char key[64];
    size_t kb;
    int found = 0;
    while (found < 2 && fscanf(f, "%63s %zu kB\n", key, &kb) == 2) {
        if (!strcmp(key, "MemTotal:"))     { *total     = kb * 1024; ++found; }
        if (!strcmp(key, "MemAvailable:")) { *available = kb * 1024; ++found; }
    }
    fclose(f);
}

// Hard cap on how much system RAM cuda_swap may use as managed-memory overflow.
// If CUDA_SWAP_MAX_HOST_MB is set, use that fixed value.
// Otherwise, dynamically compute as: available_ram - 2 GB safety margin.
// Reading dynamically (not cached) so it reflects current system pressure.
static inline size_t max_host_bytes() {
    const char* e = getenv("CUDA_SWAP_MAX_HOST_MB");
    if (e) return (size_t)atol(e) * 1024 * 1024;

    size_t total = 0, available = 0;
    read_meminfo(&total, &available);
    const size_t safety = (size_t)2 * 1024 * 1024 * 1024; // 2 GB for OS + page fault headroom
    return available > safety ? available - safety : (size_t)512 * 1024 * 1024;
}
