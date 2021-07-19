// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/number_of_cmx_slices.hpp"
#include "vpu/configuration/options/number_of_shaves.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/utils/error.hpp"

namespace vpu {

void NumberOfCMXSlicesOption::validate(const std::string& value) {
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

void NumberOfCMXSlicesOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
    VPU_THROW_UNLESS((configuration[key()] == defaultValue() &&
        configuration[NumberOfSHAVEsOption::key()] == NumberOfSHAVEsOption::defaultValue()) ||
        (configuration[key()] != defaultValue() &&
        configuration[NumberOfSHAVEsOption::key()] != NumberOfSHAVEsOption::defaultValue()),
        R"(should set both options for resource management: {} and {})", NumberOfSHAVEsOption::key(), key());
    if (configuration[key()] != defaultValue()) {
        VPU_THROW_UNLESS(configuration.get<NumberOfSHAVEsOption>().get() <= parse(configuration[key()]).get(),
            R"(Value of option {} must be not greater than value of option {}, but {} > {} are provided)",
            NumberOfSHAVEsOption::key(), key(),
            configuration.get<NumberOfSHAVEsOption>().get(), parse(configuration[key()]).get());
    }
}

std::string NumberOfCMXSlicesOption::key() {
    return InferenceEngine::MYRIAD_NUMBER_OF_CMX_SLICES;
}

details::Access NumberOfCMXSlicesOption::access() {
    return details::Access::Private;
}

details::Category NumberOfCMXSlicesOption::category() {
    return details::Category::CompileTime;
}

std::string NumberOfCMXSlicesOption::defaultValue() {
    return InferenceEngine::MYRIAD_NUMBER_OF_CMX_SLICES_AUTO;
}

NumberOfCMXSlicesOption::value_type NumberOfCMXSlicesOption::parse(const std::string& value) {
    if (value == defaultValue()) {
        return NumberOfCMXSlicesOption::value_type();
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
