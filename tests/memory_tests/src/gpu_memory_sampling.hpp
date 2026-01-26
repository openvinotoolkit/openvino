#include <cstdint>


struct GpuMemorySample {
    // memory size in kb
    int64_t total = -1;
    int64_t free = -1;
    int64_t used = -1;
};


enum INIT_GPU_STATUS {
    INIT_GPU_STATUS_SUCCESS,
    INIT_GPU_STATUS_SUBSYSTEM_UNAVAILABLE,
    INIT_GPU_STATUS_SUBSYSTEM_UNSUPPORTED,
    INIT_GPU_STATUS_GPU_NOT_FOUND,
};


INIT_GPU_STATUS initGpuSampling();

GpuMemorySample sampleGpuMemory();
