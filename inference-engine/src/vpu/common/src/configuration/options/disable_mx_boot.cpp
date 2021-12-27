// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/disable_mx_boot.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/myriad_config.hpp"

namespace vpu {

void DisableMXBootOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void DisableMXBootOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string DisableMXBootOption::key() {
    return InferenceEngine::MYRIAD_DISABLE_MX_BOOT;
}

details::Access DisableMXBootOption::access() {
    return details::Access::Private;
}

details::Category DisableMXBootOption::category() {
    return details::Category::RunTime;
}

std::string DisableMXBootOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::NO;
}

DisableMXBootOption::value_type DisableMXBootOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
