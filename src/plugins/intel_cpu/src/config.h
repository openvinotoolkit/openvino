// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie/ie_common.h>

#include <bitset>
#include <ie_performance_hints.hpp>
#include <map>
#include <mutex>
#include <openvino/util/common_util.hpp>
#include <string>
#include <threading/ie_istreams_executor.hpp>

#include "utils/debug_caps_config.h"

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

    /**
     * @brief Enum to define possible processor type hints for CPU inference
     * @ingroup ov_runtime_cpp_prop_api
     */
    enum class ProcessorType {
        UNDEFINED = -1,  //!<  Undefined value, default setting may vary by platform and performance hints
        ALL = 1,  //!<  All processors can be used. If hyper threading is enabled, both processors of oneperformance-core
                //!<  will be used.
        PHY_CORE_ONLY = 2,    //!<  Only one processor can be used per CPU core even with hyper threading enabled.
        P_CORE_ONLY = 3,      //!<  Only processors of performance-cores can be used. If hyper threading is enabled, both
                            //!<  processors of one performance-core will be used.
        E_CORE_ONLY = 4,      //!<  Only processors of efficient-cores can be used.
        PHY_P_CORE_ONLY = 5,  //!<  Only one processor can be used per performance-core even with hyper threading enabled.
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
    ProcessorType cpu_processor_type;
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

    std::string cache_dir{};

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
