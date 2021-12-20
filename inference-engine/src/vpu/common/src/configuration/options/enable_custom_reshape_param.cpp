// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/enable_custom_reshape_param.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

namespace vpu {

void EnableCustomReshapeParamOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void EnableCustomReshapeParamOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string EnableCustomReshapeParamOption::key() {
    return InferenceEngine::MYRIAD_ENABLE_CUSTOM_RESHAPE_PARAM;
}

details::Access EnableCustomReshapeParamOption::access() {
    return details::Access::Private;
}

details::Category EnableCustomReshapeParamOption::category() {
    return details::Category::CompileTime;
}

std::string EnableCustomReshapeParamOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::NO;
}

EnableCustomReshapeParamOption::value_type EnableCustomReshapeParamOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
