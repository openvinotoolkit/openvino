// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/copy_optimization.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

namespace vpu {

void CopyOptimizationOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected copy optimization option value "{}", only {} are supported)", value, getKeys(converters));
}

void CopyOptimizationOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string CopyOptimizationOption::key() {
    return InferenceEngine::MYRIAD_COPY_OPTIMIZATION;
}

details::Access CopyOptimizationOption::access() {
    return details::Access::Private;
}

details::Category CopyOptimizationOption::category() {
    return details::Category::CompileTime;
}

std::string CopyOptimizationOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::YES;
}

CopyOptimizationOption::value_type CopyOptimizationOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected copy optimization option value "{}", only {} are supported)",
        value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
