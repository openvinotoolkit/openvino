// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/config/runtime.hpp"

#include <sstream>

#include "intel_npu/config/common.hpp"
#include "openvino/runtime/properties.hpp"

using namespace intel_npu;
using namespace ov::intel_npu;

//
// register
//

void intel_npu::registerRunTimeOptions(OptionsDesc& desc) {
    desc.add<EXCLUSIVE_ASYNC_REQUESTS>();
    desc.add<PROFILING_TYPE>();
    desc.add<MODEL_PRIORITY>();
    desc.add<CREATE_EXECUTOR>();
    desc.add<DEFER_WEIGHTS_LOAD>();
    desc.add<NUM_STREAMS>();
    desc.add<ENABLE_CPU_PINNING>();
    desc.add<WORKLOAD_TYPE>();
    desc.add<TURBO>();
    desc.add<BYPASS_UMD_CACHING>();
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
