// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/configuration/options/tensor_strides.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

#include <debug.h>

namespace vpu {

void TensorStridesOption::validate(const std::string& value) {
    if (value == defaultValue()) {
        return;
    }

    auto configStrides = value;
    configStrides.pop_back();

    const auto inputs = InferenceEngine::details::split(configStrides, "],");

    for (const auto& input : inputs) {
        const auto pair = InferenceEngine::details::split(input, "[");
        VPU_THROW_UNSUPPORTED_OPTION_UNLESS(pair.size() == 2,
            R"(unexpected {} option value "{}", value {} does not match the pattern: tensor_name[strides])",
            key(), value, input);
    }
}

void TensorStridesOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string TensorStridesOption::key() {
    return InferenceEngine::MYRIAD_TENSOR_STRIDES;
}

details::Access TensorStridesOption::access() {
    return details::Access::Private;
}

details::Category TensorStridesOption::category() {
    return details::Category::CompileTime;
}

std::string TensorStridesOption::defaultValue() {
    return std::string();
}

TensorStridesOption::value_type TensorStridesOption::parse(const std::string& value) {
    if (value == defaultValue()) {
        return TensorStridesOption::value_type();
    }

    auto configStrides = value;
    configStrides.pop_back();

    const auto inputs = InferenceEngine::details::split(configStrides, "],");

    std::map<std::string, std::vector<int>> stridesMap;

    for (const auto& input : inputs) {
        std::vector<int> strides;

        const auto pair = InferenceEngine::details::split(input, "[");
        VPU_THROW_UNSUPPORTED_OPTION_UNLESS(pair.size() == 2,
            R"(unexpected {} option value "{}", value {} does not match the pattern: tensor_name[strides])",
            key(), value, input);

        const auto strideValues = InferenceEngine::details::split(pair.at(1), ",");

        for (const auto& stride : strideValues) {
            strides.insert(strides.begin(), std::stoi(stride));
        }

        stridesMap.insert({pair.at(0), strides});
    }

    return stridesMap;
}

}  // namespace vpu
