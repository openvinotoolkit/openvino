// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/tiling_cmx_limit_kb.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/utils/error.hpp"

namespace vpu {

void TilingCMXLimitKBOption::validate(const std::string& value) {
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

void TilingCMXLimitKBOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string TilingCMXLimitKBOption::key() {
    return InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB;
}

details::Access TilingCMXLimitKBOption::access() {
    return details::Access::Private;
}

details::Category TilingCMXLimitKBOption::category() {
    return details::Category::CompileTime;
}

std::string TilingCMXLimitKBOption::defaultValue() {
    return InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB_AUTO;
}

TilingCMXLimitKBOption::value_type TilingCMXLimitKBOption::parse(const std::string& value) {
    if (value == defaultValue()) {
        return TilingCMXLimitKBOption::value_type();
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
