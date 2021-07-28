// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/exclusive_async_requests.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "ie_plugin_config.hpp"

namespace vpu {

void ExclusiveAsyncRequestsOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void ExclusiveAsyncRequestsOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string ExclusiveAsyncRequestsOption::key() {
    return InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS;
}

details::Access ExclusiveAsyncRequestsOption::access() {
    return details::Access::Public;
}

details::Category ExclusiveAsyncRequestsOption::category() {
    return details::Category::RunTime;
}

std::string ExclusiveAsyncRequestsOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::NO;
}

ExclusiveAsyncRequestsOption::value_type ExclusiveAsyncRequestsOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
