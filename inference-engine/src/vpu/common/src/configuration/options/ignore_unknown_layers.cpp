// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/ignore_unknown_layers.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

namespace vpu {

void IgnoreUnknownLayersOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void IgnoreUnknownLayersOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string IgnoreUnknownLayersOption::key() {
    return InferenceEngine::MYRIAD_IGNORE_UNKNOWN_LAYERS;
}

details::Access IgnoreUnknownLayersOption::access() {
    return details::Access::Private;
}

details::Category IgnoreUnknownLayersOption::category() {
    return details::Category::CompileTime;
}

std::string IgnoreUnknownLayersOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::NO;
}

IgnoreUnknownLayersOption::value_type IgnoreUnknownLayersOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
