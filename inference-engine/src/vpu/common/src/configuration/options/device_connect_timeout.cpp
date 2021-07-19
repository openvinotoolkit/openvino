// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/configuration/options/device_connect_timeout.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/utils/error.hpp"

namespace vpu {

void DeviceConnectTimeoutOption::validate(const std::string& value) {
    int intValue;
    try {
        intValue = std::stoi(value);
    } catch (const std::exception& e) {
        VPU_THROW_FORMAT(R"(unexpected {} option value "{}", must be a number)", key(), value);
    }

    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(intValue >= 0,
        R"(unexpected {} option value "{}", only not negative numbers are supported)", key(), value);
    return;
}

void DeviceConnectTimeoutOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string DeviceConnectTimeoutOption::key() {
    return InferenceEngine::MYRIAD_DEVICE_CONNECT_TIMEOUT;
}

details::Access DeviceConnectTimeoutOption::access() {
    return details::Access::Private;
}

details::Category DeviceConnectTimeoutOption::category() {
    return details::Category::RunTime;
}

std::string DeviceConnectTimeoutOption::defaultValue() {
    return std::to_string(15);
}

DeviceConnectTimeoutOption::value_type DeviceConnectTimeoutOption::parse(const std::string& value) {
    int intValue;
    try {
        intValue = std::stoi(value);
    } catch (const std::exception& e) {
        VPU_THROW_FORMAT(R"(unexpected {} option value "{}", must be a number)", key(), value);
    }

    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(intValue >= 0,
        R"(unexpected {} option value "{}", only not negative numbers are supported)", key(), value);
    return DeviceConnectTimeoutOption::value_type(intValue);
}

}  // namespace vpu
