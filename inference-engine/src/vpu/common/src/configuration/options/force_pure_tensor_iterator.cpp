// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/force_pure_tensor_iterator.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

namespace vpu {

void ForcePureTensorIteratorOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void ForcePureTensorIteratorOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string ForcePureTensorIteratorOption::key() {
    return InferenceEngine::MYRIAD_FORCE_PURE_TENSOR_ITERATOR;
}

details::Access ForcePureTensorIteratorOption::access() {
    return details::Access::Private;
}

details::Category ForcePureTensorIteratorOption::category() {
    return details::Category::CompileTime;
}

std::string ForcePureTensorIteratorOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::NO;
}

ForcePureTensorIteratorOption::value_type ForcePureTensorIteratorOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
