// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include "ie_precision.hpp"

namespace ov {
namespace intel_cpu {
class CPUInfo {
public:
    CPUInfo();
    float getPeakGOPSImpl(InferenceEngine::Precision precision);
    void printDetails();

private:
    enum class ISA {
        sse,
        sse2,
        sse3,
        ssse3,
        sse4_1,
        sse4_2,
        avx,
        avx2,
        fma,
        avx512_common,
        avx512_core,
        avx512_mic,
        avx512_mic_4ops,
        avx512_vnni,
    };

    void init();
    bool checkIsaSupport(ISA cpu_isa);
    bool haveSSE();
    bool haveAVX();
    bool haveAVX512();

    float calcComputeBlockIPC(InferenceEngine::Precision precision);

    float getFrequency(const std::string path);
    float getMaxCPUFreq(size_t core_id);
    float getMinCPUFreq(size_t core_id);
    bool isFrequencyFixed();

#ifdef WIN32
    float currGHz = 1.0f;
#endif

    bool have_sse = false;
    bool have_sse2 = false;
    bool have_ssse3 = false;
    bool have_sse4_1 = false;
    bool have_sse4_2 = false;
    bool have_avx = false;
    bool have_avx2 = false;
    bool have_fma = false;
    bool have_avx512f = false;
    bool have_vnni = false;

    // Micro architecture level
    uint32_t simd_size = 1;
    float instructions_per_cycle = 1.0f;
    uint32_t operations_per_compute_block = 1;

    // Machine architecture level
    float freqGHz = 1.0f;
    uint32_t cores_per_socket = 1;
    uint32_t sockets_per_node = 1;
};
}  // namespace intel_cpu
}  // namespace ov