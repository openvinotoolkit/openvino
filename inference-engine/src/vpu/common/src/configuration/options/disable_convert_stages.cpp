// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/disable_convert_stages.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

namespace vpu {

void DisableConvertStagesOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void DisableConvertStagesOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string DisableConvertStagesOption::key() {
    return InferenceEngine::MYRIAD_DISABLE_CONVERT_STAGES;
}

details::Access DisableConvertStagesOption::access() {
    return details::Access::Private;
}

details::Category DisableConvertStagesOption::category() {
    return details::Category::CompileTime;
}

std::string DisableConvertStagesOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::NO;
}

DisableConvertStagesOption::value_type DisableConvertStagesOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
