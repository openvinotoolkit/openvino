// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/ov_throughput_streams.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/utils/error.hpp"
#include <openvino/runtime/properties.hpp>
#include <sstream>
#include <vpu/myriad_config.hpp>
#include <openvino/util/common_util.hpp>

namespace vpu {

void OvThroughputStreamsOption::validate(const std::string& value) {
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

void OvThroughputStreamsOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string OvThroughputStreamsOption::key() {
    return ov::num_streams.name();
}

details::Access OvThroughputStreamsOption::access() {
    return details::Access::Public;
}

details::Category OvThroughputStreamsOption::category() {
    return details::Category::CompileTime;
}

std::string OvThroughputStreamsOption::defaultValue() {
    return ov::util::to_string(ov::streams::AUTO);
}

OvThroughputStreamsOption::value_type OvThroughputStreamsOption::parse(const std::string& value) {
    if (value == defaultValue()) {
        return OvThroughputStreamsOption::value_type();
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
