// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/throughput_streams.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/utils/error.hpp"
#include <vpu/myriad_config.hpp>

namespace vpu {

void ThroughputStreamsOption::validate(const std::string& value) {
    if (value == defaultValue()) {
        return;
    }

    int intValue;
    try {
        intValue = std::stoi(value);
    } catch (const std::exception& e) {
        VPU_THROW_FORMAT(R"(unexpected {} option value "{}", must be a number)", key(), value);
    }

    VPU_THROW_UNLESS(intValue >= 0,
        R"(unexpected {} option value "{}", only not negative numbers are supported)", key(), value);
}

void ThroughputStreamsOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string ThroughputStreamsOption::key() {
    return InferenceEngine::MYRIAD_THROUGHPUT_STREAMS;
}

details::Access ThroughputStreamsOption::access() {
    return details::Access::Public;
}

details::Category ThroughputStreamsOption::category() {
    return details::Category::CompileTime;
}

std::string ThroughputStreamsOption::defaultValue() {
    return InferenceEngine::MYRIAD_THROUGHPUT_STREAMS_AUTO;
}

ThroughputStreamsOption::value_type ThroughputStreamsOption::parse(const std::string& value) {
    if (value == defaultValue()) {
        return ThroughputStreamsOption::value_type();
    }

    int intValue;
    try {
        intValue = std::stoi(value);
    } catch (const std::exception& e) {
        VPU_THROW_FORMAT(R"(unexpected {} option value "{}", must be a number)", key(), value);
    }

    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(intValue >= 0,
        R"(unexpected {} option value "{}", only not negative numbers are supported)", key(), value);
    return intValue;
}

}  // namespace vpu
