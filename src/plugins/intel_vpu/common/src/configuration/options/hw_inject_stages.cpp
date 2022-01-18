// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/hw_inject_stages.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

namespace vpu {

void HwInjectStagesOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(value == defaultValue() || converters.count(value) != 0,
        R"(unexpected {} option value "{}", only {} are supported)", key(), value, getKeys(converters));
}

void HwInjectStagesOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string HwInjectStagesOption::key() {
    return InferenceEngine::MYRIAD_HW_INJECT_STAGES;
}

details::Access HwInjectStagesOption::access() {
    return details::Access::Private;
}

details::Category HwInjectStagesOption::category() {
    return details::Category::CompileTime;
}

std::string HwInjectStagesOption::defaultValue() {
    return InferenceEngine::MYRIAD_HW_INJECT_STAGES_AUTO;
}

HwInjectStagesOption::value_type HwInjectStagesOption::parse(const std::string& value) {
    if (value == defaultValue()) {
        return HwInjectStagesOption::value_type();
    }

    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0,
        R"(unexpected {} option value "{}", only {} are supported)", key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
