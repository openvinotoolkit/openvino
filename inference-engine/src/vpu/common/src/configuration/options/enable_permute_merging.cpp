// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/enable_permute_merging.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

namespace vpu {

void EnablePermuteMergingOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void EnablePermuteMergingOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string EnablePermuteMergingOption::key() {
    return InferenceEngine::MYRIAD_ENABLE_PERMUTE_MERGING;
}

details::Access EnablePermuteMergingOption::access() {
    return details::Access::Private;
}

details::Category EnablePermuteMergingOption::category() {
    return details::Category::CompileTime;
}

std::string EnablePermuteMergingOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::YES;
}

EnablePermuteMergingOption::value_type EnablePermuteMergingOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
