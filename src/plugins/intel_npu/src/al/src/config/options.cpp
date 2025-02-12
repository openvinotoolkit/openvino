// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/config/options.hpp"

using namespace intel_npu;
using namespace ov::intel_npu;

//
// register
//

void intel_npu::registerOptions(OptionsDesc& desc, OptionMode mode) {
    // register
    desc.tryAdd<PERFORMANCE_HINT>(mode);
    desc.tryAdd<PERFORMANCE_HINT_NUM_REQUESTS>(mode);
    desc.tryAdd<INFERENCE_PRECISION_HINT>(mode);
    desc.tryAdd<PERF_COUNT>(mode);
    desc.tryAdd<LOG_LEVEL>(mode);
    desc.tryAdd<PLATFORM>(mode);
    desc.tryAdd<DEVICE_ID>(mode);
    desc.tryAdd<CACHE_DIR>(mode);
    desc.tryAdd<LOADED_FROM_CACHE>(mode);
    desc.tryAdd<BATCH_MODE>(mode);
    desc.tryAdd<COMPILER_TYPE>(mode);
    desc.tryAdd<COMPILATION_MODE>(mode);
    desc.tryAdd<COMPILATION_MODE_PARAMS>(mode);
    desc.tryAdd<BACKEND_COMPILATION_PARAMS>(mode);
    desc.tryAdd<COMPILATION_NUM_THREADS>(mode);
    desc.tryAdd<DPU_GROUPS>(mode);
    desc.tryAdd<TILES>(mode);
    desc.tryAdd<STEPPING>(mode);
    desc.tryAdd<MAX_TILES>(mode);
    desc.tryAdd<DMA_ENGINES>(mode);
    desc.tryAdd<DYNAMIC_SHAPE_TO_STATIC>(mode);
    desc.tryAdd<EXECUTION_MODE_HINT>(mode);
    desc.tryAdd<COMPILER_DYNAMIC_QUANTIZATION>(mode);
    desc.tryAdd<EXCLUSIVE_ASYNC_REQUESTS>(mode);
    desc.tryAdd<PROFILING_TYPE>(mode);
    desc.tryAdd<MODEL_PRIORITY>(mode);
    desc.tryAdd<CREATE_EXECUTOR>(mode);
    desc.tryAdd<NUM_STREAMS>(mode);
    desc.tryAdd<ENABLE_CPU_PINNING>(mode);
    desc.tryAdd<WORKLOAD_TYPE>(mode);
    desc.tryAdd<TURBO>(mode);
    desc.tryAdd<BYPASS_UMD_CACHING>(mode);
    desc.tryAdd<DEFER_WEIGHTS_LOAD>(mode);
    desc.tryAdd<WEIGHTS_PATH>(mode);
    desc.tryAdd<RUN_INFERENCES_SEQUENTIALLY>(mode);
    desc.tryAdd<DISABLE_VERSION_CHECK>(mode);
    desc.tryAdd<MODEL_PTR>(mode);
    desc.tryAdd<BATCH_COMPILER_MODE_SETTINGS>(mode);
}

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