// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/configuration/options/config_file.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

#include "ie_plugin_config.hpp"

namespace vpu {

void ConfigFileOption::validate(const std::string& value) {}

void ConfigFileOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string ConfigFileOption::key() {
    return CONFIG_KEY(CONFIG_FILE);
}

details::Access ConfigFileOption::access() {
    return details::Access::Public;
}

details::Category ConfigFileOption::category() {
    return details::Category::CompileTime;
}

std::string ConfigFileOption::defaultValue() {
    return std::string();
}

ConfigFileOption::value_type ConfigFileOption::parse(const std::string& value) {
    return value;
}

}  // namespace vpu
