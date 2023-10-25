// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <threading/ie_istreams_executor.hpp>
#include <ie_performance_hints.hpp>
#include <ie/ie_common.h>
#include <openvino/runtime/properties.hpp>
#include <openvino/util/common_util.hpp>
#include "utils/debug_caps_config.h"
#include <openvino/core/type/element_type.hpp>

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

    enum class LatencyThreadingMode {
        PER_NUMA_NODE,
        PER_SOCKET,
        PER_PLATFORM,
    };

    enum class ModelType {
        CNN,
        Unknown
    };

    bool collectPerfCounters = false;
    bool exclusiveAsyncRequests = false;
    SnippetsMode snippetsMode = SnippetsMode::Enable;
    std::string dumpToDot = {};
    std::string device_id = {};
    float fcSparseWeiDecompressionRate = 1.0f;
#if defined(OPENVINO_ARCH_X86_64)
    size_t rtCacheCapacity = 5000ul;
#else
    // TODO: Executor cache may leads to incorrect behavior on oneDNN ACL primitives
    size_t rtCacheCapacity = 0ul;
#endif
    InferenceEngine::IStreamsExecutor::Config streamExecutorConfig;
    InferenceEngine::PerfHintsConfig  perfHintsConfig;
    bool enableCpuPinning = true;
    bool changedCpuPinning = false;
    ov::hint::SchedulingCoreType schedulingCoreType = ov::hint::SchedulingCoreType::ANY_CORE;
    bool enableHyperThreading = true;
    bool changedHyperThreading = false;
    Config::LatencyThreadingMode latencyThreadingMode = Config::LatencyThreadingMode::PER_SOCKET;
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
    LPTransformsMode lpTransformsMode = LPTransformsMode::On;
#else
    // Currently INT8 mode is not optimized on ARM / RISCV or other non-x86 platforms, fallback to FP32 mode.
    LPTransformsMode lpTransformsMode = LPTransformsMode::Off;
#endif
    // default inference precision
    ov::element::Type inferencePrecision = ov::element::f32;
    bool inferencePrecisionSetExplicitly = false;
    ov::hint::ExecutionMode executionMode = ov::hint::ExecutionMode::PERFORMANCE;

    DenormalsOptMode denormalsOptMode = DenormalsOptMode::DO_Keep;

    // The denormals-are-zeros flag was introduced in the Pentium 4 and Intel Xeon processor
    // In earlier IA-32 processors and in some models of the Pentium 4 processor, this flag (bit 6)
    // is reserved.
    bool DAZOn = false;

    void readProperties(const std::map<std::string, std::string> &config, const ModelType modelType = ModelType::Unknown);
    void updateProperties();

    std::map<std::string, std::string> _config;

    bool isLegacyApi = false;

    int modelPreferThreads = -1;
    ModelType modelType = ModelType::Unknown;

#ifdef CPU_DEBUG_CAPS
    DebugCapsConfig debugCaps;
    void applyDebugCapsProperties();
#endif
};

}  // namespace intel_cpu
}   // namespace ov
