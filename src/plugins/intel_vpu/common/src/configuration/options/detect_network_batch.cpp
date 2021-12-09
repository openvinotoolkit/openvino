// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/detect_network_batch.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

namespace vpu {

void DetectNetworkBatchOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void DetectNetworkBatchOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string DetectNetworkBatchOption::key() {
    return InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH;
}

details::Access DetectNetworkBatchOption::access() {
    return details::Access::Private;
}

details::Category DetectNetworkBatchOption::category() {
    return details::Category::CompileTime;
}

std::string DetectNetworkBatchOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::YES;
}

DetectNetworkBatchOption::value_type DetectNetworkBatchOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
