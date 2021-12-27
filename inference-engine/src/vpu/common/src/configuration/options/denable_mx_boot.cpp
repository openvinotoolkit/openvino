// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/denable_mx_boot.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/myriad_config.hpp"

namespace vpu {

void DenableMXBootOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void DenableMXBootOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string DenableMXBootOption::key() {
    return InferenceEngine::MYRIAD_DENABLE_MX_BOOT;
}

details::Access DenableMXBootOption::access() {
    return details::Access::Private;
}

details::Category DenableMXBootOption::category() {
    return details::Category::RunTime;
}

std::string DenableMXBootOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::NO;
}

DenableMXBootOption::value_type DenableMXBootOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
