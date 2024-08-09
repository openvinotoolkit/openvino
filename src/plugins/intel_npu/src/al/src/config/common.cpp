// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/al/config/common.hpp"

#include "npu_private_properties.hpp"

using namespace intel_npu;
using namespace ov::intel_npu;

//
// register
//

void intel_npu::registerCommonOptions(OptionsDesc& desc) {
    desc.add<PERFORMANCE_HINT>();
    desc.add<PERFORMANCE_HINT_NUM_REQUESTS>();
    desc.add<INFERENCE_PRECISION_HINT>();
    desc.add<PERF_COUNT>();
    desc.add<LOG_LEVEL>();
    desc.add<PLATFORM>();
    desc.add<DEVICE_ID>();
    desc.add<CACHE_DIR>();
    desc.add<LOADED_FROM_CACHE>();
    desc.add<BATCH_MODE>();
    desc.add<DRIVER_ELF_FORMAT_VERSION>();
    desc.add<DRIVER_MI_VERSION>();
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

//
// DRIVER_ELF_FORMAT_VERSION
//

ov::intel_npu::Version intel_npu::DRIVER_ELF_FORMAT_VERSION::parse(std::string_view val) {
    std::stringstream strStream;
    strStream << val;
    Version version;
    strStream >> version;
    return version;
}

std::string intel_npu::DRIVER_ELF_FORMAT_VERSION::toString(const ov::intel_npu::Version& val) {
    std::stringstream strStream;

    strStream << val;

    return strStream.str();
}

//
// DRIVER_MI_VERSION
//

ov::intel_npu::Version intel_npu::DRIVER_MI_VERSION::parse(std::string_view val) {
    std::stringstream strStream;
    strStream << val;
    Version version;
    strStream >> version;
    return version;
}

std::string intel_npu::DRIVER_MI_VERSION::toString(const ov::intel_npu::Version& val) {
    std::stringstream strStream;

    strStream << val;

    return strStream.str();
}
