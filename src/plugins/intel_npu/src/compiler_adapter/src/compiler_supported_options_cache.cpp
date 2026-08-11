// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/compiler_supported_options_cache.hpp"

#include <algorithm>

namespace intel_npu {

bool CompilerSupportedOptionsCache::isOptionSupported(const ov::intel_npu::CompilerType& compilerType,
                                                      const std::string& optionName,
                                                      const std::optional<std::string>& optionValue) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto& compilerOptionsState = getStateForCompilerType(compilerType);
    const auto optionCacheKey = buildOptionCacheKey(optionName, optionValue);
    if (isOptionCachedInVector(compilerOptionsState.supportedOptions, optionCacheKey)) {
        return true;
    }

    return false;
}

void CompilerSupportedOptionsCache::addSupportedOption(const ov::intel_npu::CompilerType& compilerType,
                                                       const std::string& optionName,
                                                       const std::optional<std::string>& optionValue) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto& compilerOptionsState = getStateForCompilerType(compilerType);

    const auto optionCacheKey = buildOptionCacheKey(optionName, optionValue);
    if (!compilerOptionsState.supportedOptions.has_value()) {
        compilerOptionsState.supportedOptions = std::vector<std::string>{};
    }
    if (!isOptionCachedInVector(compilerOptionsState.supportedOptions, optionCacheKey)) {
        compilerOptionsState.supportedOptions->push_back(optionCacheKey);
    }
}

void CompilerSupportedOptionsCache::setSupportedOptions(const ov::intel_npu::CompilerType& compilerType,
                                                        const std::vector<std::string>& supportedOptions) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto& compilerOptionsState = getStateForCompilerType(compilerType);

    if (!compilerOptionsState.supportedOptions.has_value()) {
        compilerOptionsState.supportedOptions = supportedOptions;
        return;
    }

    for (const auto& option : supportedOptions) {
        if (!isOptionCachedInVector(compilerOptionsState.supportedOptions, option)) {
            compilerOptionsState.supportedOptions->push_back(option);
        }
    }
}

CompilerSupportedOptionsCache::CompilerTypeOptionsState& CompilerSupportedOptionsCache::getStateForCompilerType(
    const ov::intel_npu::CompilerType& compilerType) {
    if (compilerType == ov::intel_npu::CompilerType::DRIVER) {
        return _driverCompilerOptionsState;
    }

    if (compilerType == ov::intel_npu::CompilerType::PLUGIN) {
        return _pluginCompilerOptionsState;
    }

    OPENVINO_THROW("Unsupported compiler type in CompilerSupportedOptionsCache. Expected DRIVER or PLUGIN.");
}

std::string CompilerSupportedOptionsCache::buildOptionCacheKey(const std::string& optionName,
                                                               const std::optional<std::string>& optionValue) {
    if (!optionValue.has_value()) {
        return optionName;
    }

    std::string key = optionName;
    key += '=';
    key += optionValue.value();
    return key;
}

bool CompilerSupportedOptionsCache::isOptionCachedInVector(const std::optional<std::vector<std::string>>& options,
                                                           const std::string& optionCacheKey) {
    if (!options.has_value()) {
        return false;
    }

    const auto& values = options.value();
    return std::find(values.begin(), values.end(), optionCacheKey) != values.end();
}

}  // namespace intel_npu
