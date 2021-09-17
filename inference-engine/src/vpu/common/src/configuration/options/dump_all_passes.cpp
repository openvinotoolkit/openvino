// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/dump_all_passes.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/private_plugin_config.hpp"

#include <unordered_map>

namespace vpu {

void DumpAllPassesOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0,
        R"(unexpected {} option value "{}", only {} are supported)", key(), value, getKeys(converters));
}

void DumpAllPassesOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string DumpAllPassesOption::key() {
    return InferenceEngine::MYRIAD_DUMP_ALL_PASSES;
}

details::Access DumpAllPassesOption::access() {
    return details::Access::Private;
}

details::Category DumpAllPassesOption::category() {
    return details::Category::CompileTime;
}

std::string DumpAllPassesOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::NO;
}

DumpAllPassesOption::value_type DumpAllPassesOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0,
        R"(unexpected {} option value "{}", only {} are supported)", key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
