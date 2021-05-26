// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/configuration/options/device_id.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "ie_plugin_config.hpp"

namespace vpu {

void DeviceIDOption::validate(const std::string& value) {}

void DeviceIDOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string DeviceIDOption::key() {
    return CONFIG_KEY(DEVICE_ID);
}

details::Access DeviceIDOption::access() {
    return details::Access::Public;
}

details::Category DeviceIDOption::category() {
    return details::Category::RunTime;
}

std::string DeviceIDOption::defaultValue() {
    return std::string();
}

DeviceIDOption::value_type DeviceIDOption::parse(const std::string& value) {
    return value;
}

}  // namespace vpu
