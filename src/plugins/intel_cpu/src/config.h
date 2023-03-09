// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <threading/ie_istreams_executor.hpp>
#include <ie_performance_hints.hpp>
#include <ie/ie_common.h>
#include <openvino/util/common_util.hpp>
#include "utils/debug_caps_config.h"

#include <bitset>
#include <string>
#include <map>
#include <mutex>

namespace ov {
namespace intel_cpu {

struct Config {
    Config();

    enum LPTransformsMode {
        Off,
        On,
    };

    enum DenormalsOptMode {
        DO_Keep,
        DO_Off,
        DO_On,
    };

    enum SnippetsMode {
        Enable,
        IgnoreCallback,
        Disable,
    };

    bool collectPerfCounters = false;
    bool exclusiveAsyncRequests = false;
    bool enableDynamicBatch = false;
    SnippetsMode snippetsMode = SnippetsMode::Enable;
    std::string dumpToDot = "";
    int batchLimit = 0;
    float fcSparseWeiDecompressionRate = 1.0f;
    size_t rtCacheCapacity = 5000ul;
    InferenceEngine::IStreamsExecutor::Config streamExecutorConfig;
    InferenceEngine::PerfHintsConfig  perfHintsConfig;
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
    LPTransformsMode lpTransformsMode = LPTransformsMode::On;
    bool enforceBF16 = true;
    bool manualEnforceBF16 = false;
#else
    // Currently INT8 mode is not optimized on ARM / RISCV or other non-x86 platforms, fallback to FP32 mode.
    LPTransformsMode lpTransformsMode = LPTransformsMode::Off;
    bool enforceBF16 = false;
    bool manualEnforceBF16 = false;
#endif

    DenormalsOptMode denormalsOptMode = DenormalsOptMode::DO_Keep;

    void readProperties(const std::map<std::string, std::string> &config);
    void updateProperties();

    std::map<std::string, std::string> _config;

    bool isNewApi = true;

#ifdef CPU_DEBUG_CAPS
    DebugCapsConfig debugCaps;
    void applyDebugCapsProperties();
#endif
};

}   // namespace intel_cpu
}   // namespace ov
