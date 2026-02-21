#include <cstdint>


struct SystemMemorySample {
    // memory size in kb
    int64_t virtual_size = -1;
    int64_t virtual_peak = -1;
    int64_t resident_size = -1;
    int64_t resident_peak = -1;

    int32_t thread_count = -1;
};


SystemMemorySample sampleSystemMemory();
