#include <cstdint>


struct GpuMemorySample {
    // memory size in kb
    int64_t local_used = -1;
    int64_t local_total = -1;
    int64_t nonlocal_used = -1;
    int64_t nonlocal_total = -1;
};


enum class InitGpuStatus {
    SUCCESS,
    SUBSYSTEM_UNAVAILABLE,
    SUBSYSTEM_UNSUPPORTED,
    GPU_NOT_FOUND,
};


InitGpuStatus initGpuSampling();

GpuMemorySample sampleGpuMemory();
