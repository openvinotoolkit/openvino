// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie/ie_common.h>

#include <bitset>
#include <ie_performance_hints.hpp>
#include <map>
#include <mutex>
#include <openvino/runtime/system_conf.hpp>
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
    ProcessorType cpu_processor_type = ProcessorType::UNDEFINED;
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

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const ProcessorType& processor_type) {
    switch (processor_type) {
    case ProcessorType::UNDEFINED:
        return os << "UNDEFINED";
    case ProcessorType::ALL:
        return os << "ALL";
    case ProcessorType::PHY_CORE_ONLY:
        return os << "PHY_CORE_ONLY";
    case ProcessorType::P_CORE_ONLY:
        return os << "P_CORE_ONLY";
    case ProcessorType::E_CORE_ONLY:
        return os << "E_CORE_ONLY";
    case ProcessorType::PHY_P_CORE_ONLY:
        return os << "PHY_P_CORE_ONLY";
    default:
        throw ov::Exception{"Unsupported processor type!"};
    }
}

inline std::istream& operator>>(std::istream& is, ProcessorType& processor_type) {
    std::string str;
    is >> str;
    if (str =="UNDEFINED") {
        processor_type = ProcessorType::UNDEFINED;
    } else if (str =="ALL") {
        processor_type = ProcessorType::ALL;
    } else if (str =="PHY_CORE_ONLY") {
        processor_type = ProcessorType::PHY_CORE_ONLY;
    } else if (str =="P_CORE_ONLY") {
        processor_type = ProcessorType::P_CORE_ONLY;
    } else if (str =="E_CORE_ONLY") {
        processor_type = ProcessorType::E_CORE_ONLY;
    } else if (str =="PHY_P_CORE_ONLY") {
        processor_type = ProcessorType::PHY_P_CORE_ONLY;
    } else {
        throw ov::Exception{"Unsupported processor type: " + str};
    }
    return is;
}
/** @endcond */

}   // namespace intel_cpu
}   // namespace ov
