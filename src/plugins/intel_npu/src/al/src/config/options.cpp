// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/config/options.hpp"

using namespace intel_npu;
using namespace ov::intel_npu;

//
// PERFORMANCE_HINT
//

std::string_view ov::hint::stringifyEnum(PerformanceMode val) {
    switch (val) {
    case PerformanceMode::LATENCY:
        return "LATENCY";
    case PerformanceMode::THROUGHPUT:
        return "THROUGHPUT";
    case PerformanceMode::CUMULATIVE_THROUGHPUT:
        return "CUMULATIVE_THROUGHPUT";
    default:
        return "<UNKNOWN>";
    }
}

// Heuristically obtained number. Varies depending on the values of PLATFORM and PERFORMANCE_HINT
// Note: this is the value provided by the plugin, application should query and consider it, but may supply its own
// preference for number of parallel requests via dedicated configuration
int64_t intel_npu::getOptimalNumberOfInferRequestsInParallel(const Config& config) {
    const std::string platform = ov::intel_npu::Platform::standardize(config.get<PLATFORM>());

    if (platform == ov::intel_npu::Platform::NPU3720) {
        if (config.get<PERFORMANCE_HINT>() == ov::hint::PerformanceMode::THROUGHPUT) {
            return 4;
        } else {
            return 1;
        }
    } else {
        if (config.get<PERFORMANCE_HINT>() == ov::hint::PerformanceMode::THROUGHPUT) {
            return 8;
        } else {
            return 1;
        }
    }
}

//
// PROFILING_TYPE
//

std::string_view ov::intel_npu::stringifyEnum(ov::intel_npu::ProfilingType val) {
    switch (val) {
    case ov::intel_npu::ProfilingType::MODEL:
        return "MODEL";
    case ov::intel_npu::ProfilingType::INFER:
        return "INFER";
    default:
        return "<UNKNOWN>";
    }
}

//
// COMPILER_TYPE
//

std::string_view ov::intel_npu::stringifyEnum(ov::intel_npu::CompilerType val) {
    switch (val) {
    case ov::intel_npu::CompilerType::MLIR:
        return "MLIR";
    case ov::intel_npu::CompilerType::DRIVER:
        return "DRIVER";
    default:
        return "<UNKNOWN>";
    }
}