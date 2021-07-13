// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/configuration/options/ir_with_scales_directory.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

namespace vpu {

void IRWithScalesDirectoryOption::validate(const std::string& value) {}

void IRWithScalesDirectoryOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string IRWithScalesDirectoryOption::key() {
    return InferenceEngine::MYRIAD_IR_WITH_SCALES_DIRECTORY;
}

details::Access IRWithScalesDirectoryOption::access() {
    return details::Access::Private;
}

details::Category IRWithScalesDirectoryOption::category() {
    return details::Category::CompileTime;
}

std::string IRWithScalesDirectoryOption::defaultValue() {
    return std::string();
}

IRWithScalesDirectoryOption::value_type IRWithScalesDirectoryOption::parse(const std::string& value) {
    return value;
}

}  // namespace vpu
