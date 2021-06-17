// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "configuration/myriad_configuration.hpp"

namespace vpu {

MyriadConfiguration::MyriadConfiguration() {}

void MyriadConfiguration::from(const std::map<std::string, std::string>& configuration) {
    std::map<std::string, std::string> migratedOptions, notMigratedOptions;
    for (const auto& entry : configuration) {
        auto& destination = PluginConfiguration::supports(entry.first) ? migratedOptions : notMigratedOptions;
        destination.emplace(entry);
    }
    PluginConfiguration::from(migratedOptions);
    update(notMigratedOptions);
}

void MyriadConfiguration::fromAtRuntime(const std::map<std::string, std::string>& configuration) {
    std::map<std::string, std::string> migratedOptions, notMigratedOptions;
    for (const auto& entry : configuration) {
        auto& destination = PluginConfiguration::supports(entry.first) ? migratedOptions : notMigratedOptions;
        destination.emplace(entry);
    }
    PluginConfiguration::fromAtRuntime(migratedOptions);
    update(notMigratedOptions, ConfigMode::RunTime);
}

}  // namespace vpu
