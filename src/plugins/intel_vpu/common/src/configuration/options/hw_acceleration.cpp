// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/hw_acceleration.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/vpu_config.hpp"

namespace vpu {

void HwAccelerationOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void HwAccelerationOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string HwAccelerationOption::key() {
    return InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION;
}

details::Access HwAccelerationOption::access() {
    return details::Access::Public;
}

details::Category HwAccelerationOption::category() {
    return details::Category::CompileTime;
}

std::string HwAccelerationOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::YES;
}

HwAccelerationOption::value_type HwAccelerationOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
