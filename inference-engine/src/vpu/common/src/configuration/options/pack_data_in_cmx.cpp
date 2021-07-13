// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/pack_data_in_cmx.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

namespace vpu {

void PackDataInCMXOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void PackDataInCMXOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string PackDataInCMXOption::key() {
    return InferenceEngine::MYRIAD_PACK_DATA_IN_CMX;
}

details::Access PackDataInCMXOption::access() {
    return details::Access::Private;
}

details::Category PackDataInCMXOption::category() {
    return details::Category::CompileTime;
}

std::string PackDataInCMXOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::YES;
}

PackDataInCMXOption::value_type PackDataInCMXOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
