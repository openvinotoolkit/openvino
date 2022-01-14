// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/enable_force_reset.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/myriad_config.hpp"

namespace vpu {

void EnableForceResetOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void EnableForceResetOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string EnableForceResetOption::key() {
    return InferenceEngine::MYRIAD_ENABLE_FORCE_RESET;
}

details::Access EnableForceResetOption::access() {
    return details::Access::Public;
}

details::Category EnableForceResetOption::category() {
    return details::Category::RunTime;
}

std::string EnableForceResetOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::NO;
}

EnableForceResetOption::value_type EnableForceResetOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
