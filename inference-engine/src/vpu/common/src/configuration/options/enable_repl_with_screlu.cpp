// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/enable_repl_with_screlu.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

namespace vpu {

void EnableReplWithSCReluOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void EnableReplWithSCReluOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string EnableReplWithSCReluOption::key() {
    return InferenceEngine::MYRIAD_ENABLE_REPL_WITH_SCRELU;
}

details::Access EnableReplWithSCReluOption::access() {
    return details::Access::Private;
}

details::Category EnableReplWithSCReluOption::category() {
    return details::Category::CompileTime;
}

std::string EnableReplWithSCReluOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::NO;
}

EnableReplWithSCReluOption::value_type EnableReplWithSCReluOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
