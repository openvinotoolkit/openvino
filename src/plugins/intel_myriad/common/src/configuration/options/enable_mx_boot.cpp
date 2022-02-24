// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/enable_mx_boot.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/private_plugin_config.hpp"
#include <vpu/utils/string.hpp>

namespace vpu {

void EnableMXBootOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void EnableMXBootOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string EnableMXBootOption::key() {
    return InferenceEngine::MYRIAD_ENABLE_MX_BOOT;
}

details::Access EnableMXBootOption::access() {
    return details::Access::Private;
}

details::Category EnableMXBootOption::category() {
    return details::Category::RunTime;
}

std::string EnableMXBootOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::YES;
}

EnableMXBootOption::value_type EnableMXBootOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
