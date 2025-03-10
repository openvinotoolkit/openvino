// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <bitset>
#include <map>
#include <mutex>

#include "internal_properties.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "openvino/util/common_util.hpp"
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

    enum CacheQuantMode {
        AUTO,
        BY_CHANNEL,
        BY_HIDDEN,
    };

    enum class ModelType { CNN, LLM, Unknown };

    bool collectPerfCounters = false;
    bool exclusiveAsyncRequests = false;
    SnippetsMode snippetsMode = SnippetsMode::Enable;
    std::string dumpToDot = {};
    std::string device_id = {};
    float fcSparseWeiDecompressionRate = 1.0f;
    uint64_t fcDynamicQuantizationGroupSize = 32;
    bool fcDynamicQuantizationGroupSizeSetExplicitly = false;
    bool kvCachePrecisionSetExplicitly = false;
    bool keyCachePrecisionSetExplicitly = false;
    bool valueCachePrecisionSetExplicitly = false;
    bool keyCacheGroupSizeSetExplicitly = false;
    bool valueCacheGroupSizeSetExplicitly = false;
#if defined(OV_CPU_WITH_ACL)
    bool aclFastMath = false;
#endif
#if defined(OPENVINO_ARCH_X86_64)
    ov::element::Type kvCachePrecision = ov::element::u8;
    ov::element::Type keyCachePrecision = ov::element::u8;
    ov::element::Type valueCachePrecision = ov::element::u8;
    size_t rtCacheCapacity = 5000ul;
#else
    ov::element::Type kvCachePrecision = ov::element::f16;
    ov::element::Type keyCachePrecision = ov::element::f16;
    ov::element::Type valueCachePrecision = ov::element::f16;
    // TODO: Executor cache may leads to incorrect behavior on oneDNN ACL primitives
    size_t rtCacheCapacity = 0ul;
#endif
    size_t keyCacheGroupSize = 0ul;
    size_t valueCacheGroupSize = 0ul;
    CacheQuantMode keyCacheQuantMode = CacheQuantMode::AUTO;
    ov::threading::IStreamsExecutor::Config streamExecutorConfig;
    int streams = 1;
    bool streamsChanged = false;
    int threads = 0;
    int threadsPerStream = 0;
    ov::hint::PerformanceMode hintPerfMode = ov::hint::PerformanceMode::LATENCY;
    std::vector<std::vector<int>> streamsRankTable;
    bool changedHintPerfMode = false;
    ov::log::Level logLevel = ov::log::Level::NO;
    uint32_t hintNumRequests = 0;
    bool enableCpuPinning = true;
    bool changedCpuPinning = false;
    bool enableCpuReservation = false;
    ov::hint::SchedulingCoreType schedulingCoreType = ov::hint::SchedulingCoreType::ANY_CORE;
    std::set<ov::hint::ModelDistributionPolicy> modelDistributionPolicy = {};
    int streamsRankLevel = 1;
    int numSubStreams = 0;
    bool enableNodeSplit = false;
    bool enableHyperThreading = true;
    bool changedHyperThreading = false;
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64) || defined(OPENVINO_ARCH_ARM64)
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

    void readProperties(const ov::AnyMap& config, const ModelType modelType = ModelType::Unknown);

    void updateProperties();

    void applyRtInfo(const std::shared_ptr<const ov::Model>& model);

    std::map<std::string, std::string> _config;

    int modelPreferThreads = -1;
    ModelType modelType = ModelType::Unknown;
    std::function<std::string(const std::string&)> cacheEncrypt;
    std::function<std::string(const std::string&)> cacheDecrypt;

#ifdef CPU_DEBUG_CAPS
    DebugCapsConfig debugCaps;
    void applyDebugCapsProperties();
#endif
};

}  // namespace intel_cpu
}  // namespace ov
