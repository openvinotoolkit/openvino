// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/enable_receiving_tensor_time.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/vpu_config.hpp"

namespace vpu {

void EnableReceivingTensorTimeOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void EnableReceivingTensorTimeOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string EnableReceivingTensorTimeOption::key() {
    return InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME;
}

details::Access EnableReceivingTensorTimeOption::access() {
    return details::Access::Public;
}

details::Category EnableReceivingTensorTimeOption::category() {
    return details::Category::RunTime;
}

std::string EnableReceivingTensorTimeOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::NO;
}

EnableReceivingTensorTimeOption::value_type EnableReceivingTensorTimeOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
