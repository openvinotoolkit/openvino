// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/enable_tensor_iterator_unrolling.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

namespace vpu {

void EnableTensorIteratorUnrollingOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void EnableTensorIteratorUnrollingOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string EnableTensorIteratorUnrollingOption::key() {
    return InferenceEngine::MYRIAD_ENABLE_TENSOR_ITERATOR_UNROLLING;
}

details::Access EnableTensorIteratorUnrollingOption::access() {
    return details::Access::Private;
}

details::Category EnableTensorIteratorUnrollingOption::category() {
    return details::Category::CompileTime;
}

std::string EnableTensorIteratorUnrollingOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::NO;
}

EnableTensorIteratorUnrollingOption::value_type EnableTensorIteratorUnrollingOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
