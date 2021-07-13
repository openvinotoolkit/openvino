// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/hw_dilation.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

namespace vpu {

void HwDilationOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void HwDilationOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string HwDilationOption::key() {
    return InferenceEngine::MYRIAD_HW_DILATION;
}

details::Access HwDilationOption::access() {
    return details::Access::Private;
}

details::Category HwDilationOption::category() {
    return details::Category::CompileTime;
}

std::string HwDilationOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::NO;
}

HwDilationOption::value_type HwDilationOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
