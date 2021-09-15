// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/configuration/options/vpu_scales_option.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

namespace vpu {

void VPUScalesOption::validate(const std::string& value) {}

void VPUScalesOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string VPUScalesOption::key() {
    return InferenceEngine::MYRIAD_SCALES_PATTERN;
}

details::Access VPUScalesOption::access() {
    return details::Access::Private;
}

details::Category VPUScalesOption::category() {
    return details::Category::CompileTime;
}

std::string VPUScalesOption::defaultValue() {
    return std::string();
}

VPUScalesOption::value_type VPUScalesOption::parse(const std::string& value) {
    return value;
}

}  // namespace vpu
