// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/number_of_shaves.hpp"
#include "vpu/configuration/options/number_of_cmx_slices.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/utils/error.hpp"

namespace vpu {

void NumberOfSHAVEsOption::validate(const std::string& value) {
    if (value == defaultValue()) {
        return;
    }

    int intValue;
    try {
        intValue = std::stoi(value);
    } catch (const std::exception& e) {
        VPU_THROW_FORMAT(R"(unexpected {} option value "{}", must be a number)", key(), value);
    }

    VPU_THROW_UNLESS(intValue >= 0,
        R"(unexpected {} option value "{}", only not negative numbers are supported)", key(), value);
}

void NumberOfSHAVEsOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
    VPU_THROW_UNLESS((configuration[key()] == defaultValue() &&
        configuration[NumberOfCMXSlicesOption::key()] == NumberOfCMXSlicesOption::defaultValue()) ||
        (configuration[key()] != defaultValue() &&
        configuration[NumberOfCMXSlicesOption::key()] != NumberOfCMXSlicesOption::defaultValue()),
        R"(should set both options for resource management: {} and {})", key(), NumberOfCMXSlicesOption::key());
    if (configuration[key()] != defaultValue()) {
        VPU_THROW_UNLESS(parse(configuration[key()]).get() <= configuration.get<NumberOfCMXSlicesOption>().get(),
            R"(Value of option {} must be not greater than value of option {}, but {} > {} are provided)",
            key(), NumberOfCMXSlicesOption::key(),
            parse(configuration[key()]).get(), configuration.get<NumberOfCMXSlicesOption>().get());
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
        intValue = std::stoi(value);
    } catch (const std::exception& e) {
        VPU_THROW_FORMAT(R"(unexpected {} option value "{}", must be a number)", key(), value);
    }

    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(intValue >= 0,
        R"(unexpected {} option value "{}", only not negative numbers are supported)", key(), value);
    return intValue;
}

}  // namespace vpu
