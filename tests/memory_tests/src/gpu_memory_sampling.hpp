#include <cstdint>


struct GpuMemorySample {
    // memory size in kb
    int64_t total = -1;
    int64_t free = -1;
    int64_t used = -1;
};


int initGpuSampling();

GpuMemorySample sampleGpuMemory();
