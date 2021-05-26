// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/configuration/options/power_config.hpp"
#include "vpu/utils/power_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

#include "ie_plugin_config.hpp"

#include <unordered_map>

namespace vpu {

namespace {

const std::unordered_map<std::string, PowerConfig>& string2power() {
    static const std::unordered_map<std::string, PowerConfig> converters = {
        {InferenceEngine::MYRIAD_POWER_FULL,         PowerConfig::FULL},
        {InferenceEngine::MYRIAD_POWER_INFER,        PowerConfig::INFER},
        {InferenceEngine::MYRIAD_POWER_STAGE,        PowerConfig::STAGE},
        {InferenceEngine::MYRIAD_POWER_STAGE_SHAVES, PowerConfig::STAGE_SHAVES},
        {InferenceEngine::MYRIAD_POWER_STAGE_NCES,   PowerConfig::STAGE_NCES},
    };
    return converters;
}

}  // namespace

void PowerConfigOption::validate(const std::string& value) {
    const auto& converters = string2power();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void PowerConfigOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string PowerConfigOption::key() {
    return InferenceEngine::MYRIAD_POWER_MANAGEMENT;
}

details::Access PowerConfigOption::access() {
    return details::Access::Private;
}

details::Category PowerConfigOption::category() {
    return details::Category::RunTime;
}

std::string PowerConfigOption::defaultValue() {
    return InferenceEngine::MYRIAD_POWER_FULL;
}

PowerConfigOption::value_type PowerConfigOption::parse(const std::string& value) {
    const auto& converters = string2power();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
