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

ov::hint::PerformanceMode intel_npu::PERFORMANCE_HINT::parse(std::string_view val) {
    if (val.empty()) {
        return ov::hint::PerformanceMode::LATENCY;
    } else if (val == "LATENCY") {
        return ov::hint::PerformanceMode::LATENCY;
    } else if (val == "THROUGHPUT") {
        return ov::hint::PerformanceMode::THROUGHPUT;
    } else if (val == "CUMULATIVE_THROUGHPUT") {
        return ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT;
    }

    OPENVINO_THROW("Value '", val, "' is not a valid PERFORMANCE_HINT option");
}

//
// BATCH_MODE
//

ov::intel_npu::BatchMode intel_npu::BATCH_MODE::parse(std::string_view val) {
    if (val == "AUTO") {
        return ov::intel_npu::BatchMode::AUTO;
    } else if (val == "COMPILER") {
        return ov::intel_npu::BatchMode::COMPILER;
    } else if (val == "PLUGIN") {
        return ov::intel_npu::BatchMode::PLUGIN;
    }

    OPENVINO_THROW("Value '", val, "'is not a valid BATCH_MODE option");
}

std::string intel_npu::BATCH_MODE::toString(const ov::intel_npu::BatchMode& val) {
    std::stringstream strStream;

    strStream << val;

    return strStream.str();
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

ov::intel_npu::ProfilingType intel_npu::PROFILING_TYPE::parse(std::string_view val) {
    const auto extractProfilingString = [](ov::intel_npu::ProfilingType prof) -> std::string {
        return profiling_type(prof).second.as<std::string>();
    };

    if (val == extractProfilingString(ov::intel_npu::ProfilingType::MODEL)) {
        return ov::intel_npu::ProfilingType::MODEL;
    } else if (val == extractProfilingString(ov::intel_npu::ProfilingType::INFER)) {
        return ov::intel_npu::ProfilingType::INFER;
    }

    OPENVINO_THROW("Value '", val, "' is not a valid PROFILING_TYPE option");
}

std::string intel_npu::PROFILING_TYPE::toString(const ov::intel_npu::ProfilingType& val) {
    std::stringstream strStream;
    if (val == ov::intel_npu::ProfilingType::MODEL) {
        strStream << "MODEL";
    } else if (val == ov::intel_npu::ProfilingType::INFER) {
        strStream << "INFER";
    } else {
        OPENVINO_THROW("No valid string for current PROFILING_TYPE option");
    }

    return strStream.str();
}

//
// MODEL_PRIORITY
//

ov::hint::Priority intel_npu::MODEL_PRIORITY::parse(std::string_view val) {
    std::istringstream stringStream = std::istringstream(std::string(val));
    ov::hint::Priority priority;

    stringStream >> priority;

    return priority;
}

std::string intel_npu::MODEL_PRIORITY::toString(const ov::hint::Priority& val) {
    std::ostringstream stringStream;

    stringStream << val;

    return stringStream.str();
}

//
// NUM_STREAMS
//

const ov::streams::Num intel_npu::NUM_STREAMS::defVal = ov::streams::Num(1);

ov::streams::Num intel_npu::NUM_STREAMS::parse(std::string_view val) {
    std::istringstream stringStream = std::istringstream(std::string(val));
    ov::streams::Num numberOfStreams;

    stringStream >> numberOfStreams;

    return numberOfStreams;
}

std::string intel_npu::NUM_STREAMS::toString(const ov::streams::Num& val) {
    std::ostringstream stringStream;

    stringStream << val;

    return stringStream.str();
}

//
// WORKLOAD_TYPE
//

ov::WorkloadType intel_npu::WORKLOAD_TYPE::parse(std::string_view val) {
    std::istringstream ss = std::istringstream(std::string(val));
    ov::WorkloadType workloadType;

    ss >> workloadType;

    return workloadType;
}

std::string intel_npu::WORKLOAD_TYPE::toString(const ov::WorkloadType& val) {
    std::ostringstream ss;
    ss << val;
    return ss.str();
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

std::string_view intel_npu::COMPILER_TYPE::envVar() {
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    return "IE_NPU_COMPILER_TYPE";
#else
    return "";
#endif
}

ov::intel_npu::CompilerType intel_npu::COMPILER_TYPE::defaultValue() {
    return ov::intel_npu::CompilerType::DRIVER;
}

ov::intel_npu::CompilerType intel_npu::COMPILER_TYPE::parse(std::string_view val) {
    if (val == stringifyEnum(ov::intel_npu::CompilerType::MLIR)) {
        return ov::intel_npu::CompilerType::MLIR;
    } else if (val == stringifyEnum(ov::intel_npu::CompilerType::DRIVER)) {
        return ov::intel_npu::CompilerType::DRIVER;
    }

    OPENVINO_THROW("Value '", val, "' is not a valid COMPILER_TYPE option");
}

std::string intel_npu::COMPILER_TYPE::toString(const ov::intel_npu::CompilerType& val) {
    std::stringstream strStream;
    if (val == ov::intel_npu::CompilerType::MLIR) {
        strStream << "MLIR";
    } else if (val == ov::intel_npu::CompilerType::DRIVER) {
        strStream << "DRIVER";
    } else {
        OPENVINO_THROW("No valid string for current LOG_LEVEL option");
    }

    return strStream.str();
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

ov::intel_npu::ElfCompilerBackend intel_npu::USE_ELF_COMPILER_BACKEND::parse(std::string_view val) {
    if (val == stringifyEnum(ov::intel_npu::ElfCompilerBackend::AUTO)) {
        return ov::intel_npu::ElfCompilerBackend::AUTO;
    } else if (val == stringifyEnum(ov::intel_npu::ElfCompilerBackend::NO)) {
        return ov::intel_npu::ElfCompilerBackend::NO;
    } else if (val == stringifyEnum(ov::intel_npu::ElfCompilerBackend::YES)) {
        return ov::intel_npu::ElfCompilerBackend::YES;
    }

    OPENVINO_THROW("Value '", val, "' is not a valid USE_ELF_COMPILER_BACKEND option");
}

std::string intel_npu::USE_ELF_COMPILER_BACKEND::toString(const ov::intel_npu::ElfCompilerBackend& val) {
    std::stringstream strStream;
    if (val == ov::intel_npu::ElfCompilerBackend::AUTO) {
        strStream << "AUTO";
    } else if (val == ov::intel_npu::ElfCompilerBackend::NO) {
        strStream << "NO";
    } else if (val == ov::intel_npu::ElfCompilerBackend::YES) {
        strStream << "YES";
    } else {
        OPENVINO_THROW("No valid string for current USE_ELF_COMPILER_BACKEND option");
    }

    return strStream.str();
}
