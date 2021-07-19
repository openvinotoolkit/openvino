// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/configuration/options/custom_layers.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/vpu_config.hpp"

namespace vpu {

void CustomLayersOption::validate(const std::string& value) {}

void CustomLayersOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string CustomLayersOption::key() {
    return InferenceEngine::MYRIAD_CUSTOM_LAYERS;
}

details::Access CustomLayersOption::access() {
    return details::Access::Public;
}

details::Category CustomLayersOption::category() {
    return details::Category::CompileTime;
}

std::string CustomLayersOption::defaultValue() {
    return std::string();
}

CustomLayersOption::value_type CustomLayersOption::parse(const std::string& value) {
    return value;
}

}  // namespace vpu
