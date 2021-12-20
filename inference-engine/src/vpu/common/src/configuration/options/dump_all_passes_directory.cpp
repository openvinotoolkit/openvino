// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/configuration/options/dump_all_passes_directory.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

namespace vpu {

void DumpAllPassesDirectoryOption::validate(const std::string& value) {}

void DumpAllPassesDirectoryOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string DumpAllPassesDirectoryOption::key() {
    return InferenceEngine::MYRIAD_DUMP_ALL_PASSES_DIRECTORY;
}

details::Access DumpAllPassesDirectoryOption::access() {
    return details::Access::Private;
}

details::Category DumpAllPassesDirectoryOption::category() {
    return details::Category::CompileTime;
}

std::string DumpAllPassesDirectoryOption::defaultValue() {
    return std::string();
}

DumpAllPassesDirectoryOption::value_type DumpAllPassesDirectoryOption::parse(const std::string& value) {
    return value;
}

}  // namespace vpu
