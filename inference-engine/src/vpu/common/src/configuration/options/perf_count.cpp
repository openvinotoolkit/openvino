// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/perf_count.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "ie_plugin_config.hpp"

namespace vpu {

void PerfCountOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void PerfCountOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string PerfCountOption::key() {
    return CONFIG_KEY(PERF_COUNT);
}

details::Access PerfCountOption::access() {
    return details::Access::Public;
}

details::Category PerfCountOption::category() {
    return details::Category::RunTime;
}

std::string PerfCountOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::NO;
}

PerfCountOption::value_type PerfCountOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
