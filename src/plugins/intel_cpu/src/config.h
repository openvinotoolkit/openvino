// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "utils/debug_caps_config.h"

namespace ov::intel_cpu {
struct Config {
    Config();

    enum LPTransformsMode : uint8_t {
        Off,
        On,
    };

    enum DenormalsOptMode : uint8_t {
        DO_Keep,
        DO_Off,
        DO_On,
    };

    enum SnippetsMode : uint8_t {
        Enable,
        IgnoreCallback,
        Disable,
    };

    enum CacheQuantMode : uint8_t {
        AUTO,
        BY_CHANNEL,
        BY_TOKEN,
    };

    enum class ModelType : uint8_t { CNN, LLM, Unknown };

    bool collectPerfCounters = false;
    bool exclusiveAsyncRequests = false;
    SnippetsMode snippetsMode = SnippetsMode::Enable;
    std::string dumpToDot;
    std::string device_id;
    float fcSparseWeiDecompressionRate = 1.0F;
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
#if defined(OV_CPU_WITH_ACL) || defined(OV_CPU_WITH_SHL)
    // TODO: Executor cache may leads to incorrect behavior on oneDNN ACL primitives
    size_t rtCacheCapacity = 0UL;
#else
    size_t rtCacheCapacity = 5000UL;
#endif
    size_t snippetsCacheCapacity = 5000UL;
#if defined(OPENVINO_ARCH_X86_64)
    ov::element::Type kvCachePrecision = ov::element::u8;
    ov::element::Type keyCachePrecision = ov::element::u8;
    ov::element::Type valueCachePrecision = ov::element::u8;
#else
    ov::element::Type kvCachePrecision = ov::element::f16;
    ov::element::Type keyCachePrecision = ov::element::f16;
    ov::element::Type valueCachePrecision = ov::element::f16;
#endif
    size_t keyCacheGroupSize = 0UL;
    size_t valueCacheGroupSize = 0UL;
    CacheQuantMode keyCacheQuantMode = CacheQuantMode::AUTO;
    CacheQuantMode valueCacheQuantMode = CacheQuantMode::AUTO;
    bool enableSageAttn = false;
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
    std::set<ov::hint::ModelDistributionPolicy> modelDistributionPolicy;
    bool enableTensorParallel = false;
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

    void readProperties(const ov::AnyMap& prop, ModelType modelType = ModelType::Unknown);

    void updateProperties();

    void applyRtInfo(const std::shared_ptr<const ov::Model>& model);

    std::map<std::string, std::string> _config;

    int modelPreferThreads = -1;
    ModelType modelType = ModelType::Unknown;
    std::function<std::string(const std::string&)> cacheEncrypt;
    std::function<std::string(const std::string&)> cacheDecrypt;

    ov::CacheMode m_cache_mode = ov::CacheMode::OPTIMIZE_SPEED;

#ifdef CPU_DEBUG_CAPS
    DebugCapsConfig debugCaps;
    void applyDebugCapsProperties();
#endif
};

}  // namespace ov::intel_cpu
