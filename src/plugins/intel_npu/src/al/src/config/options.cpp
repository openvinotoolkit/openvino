// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/al/config/options.hpp"

using namespace intel_npu;
using namespace ov::intel_npu;

//
// register
//

void intel_npu::registerOptions(OptionsDesc& desc, compilerVersion compilerVer) {
    desc.add<PERFORMANCE_HINT>(compilerVer);
    desc.add<PERFORMANCE_HINT_NUM_REQUESTS>(compilerVer);
    desc.add<INFERENCE_PRECISION_HINT>(compilerVer);
    desc.add<PERF_COUNT>(compilerVer);
    desc.add<LOG_LEVEL>(compilerVer);
    desc.add<PLATFORM>(compilerVer);
    desc.add<DEVICE_ID>(compilerVer);
    desc.add<CACHE_DIR>(compilerVer);
    desc.add<LOADED_FROM_CACHE>(compilerVer);
    desc.add<BATCH_MODE>(compilerVer);
    desc.add<COMPILER_TYPE>(compilerVer);
    desc.add<COMPILATION_MODE>(compilerVer);
    desc.add<COMPILATION_MODE_PARAMS>(compilerVer);
    desc.add<BACKEND_COMPILATION_PARAMS>(compilerVer);
    desc.add<COMPILATION_NUM_THREADS>(compilerVer);
    desc.add<DPU_GROUPS>(compilerVer);
    desc.add<TILES>(compilerVer);
    desc.add<STEPPING>(compilerVer);
    desc.add<MAX_TILES>(compilerVer);
    desc.add<DMA_ENGINES>(compilerVer);
    desc.add<USE_ELF_COMPILER_BACKEND>(compilerVer);
    desc.add<DYNAMIC_SHAPE_TO_STATIC>(compilerVer);
    // desc.add<EXECUTION_MODE_HINT>(compilerVer);
    desc.add<EXCLUSIVE_ASYNC_REQUESTS>();
    desc.add<PROFILING_TYPE>(compilerVer);
    desc.add<MODEL_PRIORITY>(compilerVer);
    desc.add<CREATE_EXECUTOR>(compilerVer);
    desc.add<NUM_STREAMS>(compilerVer);
    desc.add<ENABLE_CPU_PINNING>(compilerVer);
    desc.add<WORKLOAD_TYPE>(compilerVer);
    desc.add<TURBO>(compilerVer);
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

//
// USE_ELF_COMPILER_BACKEND
//

std::string_view ov::intel_npu::stringifyEnum(ov::intel_npu::ElfCompilerBackend val) {
    switch (val) {
    case ov::intel_npu::ElfCompilerBackend::AUTO:
        return "AUTO";
    case ov::intel_npu::ElfCompilerBackend::NO:
        return "NO";
    case ov::intel_npu::ElfCompilerBackend::YES:
        return "YES";
    default:
        return "<UNKNOWN>";
    }
}
