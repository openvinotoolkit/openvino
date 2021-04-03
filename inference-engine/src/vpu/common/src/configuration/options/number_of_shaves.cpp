// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/number_of_shaves.hpp"
#include "vpu/configuration/parse_numeric.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

namespace vpu {

void NumberOfSHAVEsOption::validate(const std::string& value) {
    if (value == defaultValue()) {
        return;
    }

    int intValue;
    try {
        intValue = parseInt(value);
    } catch (const std::exception& e) {
        VPU_THROW_FORMAT(R"(unexpected {} option value "{}", must be a number)", key(), value);
    }

    VPU_THROW_UNLESS(!Negative(intValue),
        R"(unexpected {} option value "{}", only not negative numbers are supported)", key(), value);
}

void NumberOfSHAVEsOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
    VPU_THROW_UNLESS((configuration[key()] == defaultValue() && configuration.compileConfig().numCMXSlices < 0) ||
        (configuration[key()] != defaultValue() && configuration.compileConfig().numCMXSlices >= 0),
        R"(should set both options for resource management: {} and {})", key(), InferenceEngine::MYRIAD_NUMBER_OF_CMX_SLICES);
    if (configuration[key()] != defaultValue()) {
        VPU_THROW_UNLESS(parse(configuration[key()]).get() <= configuration.compileConfig().numCMXSlices,
            R"(Value of option {} must be not greater than value of option {}, but {} > {} are provided)",
            key(), InferenceEngine::MYRIAD_NUMBER_OF_CMX_SLICES,
            parse(configuration[key()]).get(), configuration.compileConfig().numCMXSlices);
    }
}

std::string NumberOfSHAVEsOption::key() {
    return InferenceEngine::MYRIAD_NUMBER_OF_SHAVES;
}

details::Access NumberOfSHAVEsOption::access() {
    return details::Access::Private;
}

details::Category NumberOfSHAVEsOption::category() {
    return details::Category::CompileTime;
}

std::string NumberOfSHAVEsOption::defaultValue() {
    return InferenceEngine::MYRIAD_NUMBER_OF_SHAVES_AUTO;
}

NumberOfSHAVEsOption::value_type NumberOfSHAVEsOption::parse(const std::string& value) {
    if (value == defaultValue()) {
        return NumberOfSHAVEsOption::value_type();
    }

    int intValue;
    try {
        intValue = parseInt(value);
    } catch (const std::exception& e) {
        VPU_THROW_FORMAT(R"(unexpected {} option value "{}", must be a number)", key(), value);
    }

    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(!Negative(intValue),
        R"(unexpected {} option value "{}", only not negative numbers are supported)", key(), value);
    return intValue;
}

}  // namespace vpu
