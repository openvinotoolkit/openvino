// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/compiler_options_cache.hpp"

#include <algorithm>

namespace intel_npu {

std::mutex CompilerOptionsCache::_mutex;
CompilerOptionsCache::CompilerTypeOptionsState CompilerOptionsCache::_driverCompilerOptionsState{};
CompilerOptionsCache::CompilerTypeOptionsState CompilerOptionsCache::_pluginCompilerOptionsState{};

bool CompilerOptionsCache::isOptionSupported(const ov::intel_npu::CompilerType& compilerType,
                                             const std::string& optionName,
                                             const std::optional<std::string>& optionValue,
                                             const std::optional<uint32_t>& compilerSupportVersion) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto& compilerOptionsState = getStateForCompilerType(compilerType);
    if (compilerOptionsState.legacy) {
        if (!compilerSupportVersion.has_value()) {
            OPENVINO_THROW("Cannot determine option support in legacy mode without compiler support version.");
        }
        return compilerOptionsState.compilerVersion >= compilerSupportVersion.value();
    }

    const auto optionCacheKey = buildOptionCacheKey(optionName, optionValue);
    if (isOptionCachedInVector(compilerOptionsState.supportedOptions, optionCacheKey) ||
        isOptionCachedInVector(compilerOptionsState.privateSupportedOptions, optionCacheKey)) {
        return true;
    }

    return false;
}

void CompilerOptionsCache::addSupportedOption(const ov::intel_npu::CompilerType& compilerType,
                                              const std::string& optionName,
                                              const std::optional<std::string>& optionValue) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto& compilerOptionsState = getStateForCompilerType(compilerType);
    if (compilerOptionsState.legacy) {
        OPENVINO_THROW("Cannot add private supported options in CompilerOptionsCache when legacy mode is enabled.");
    }

    const auto optionCacheKey = buildOptionCacheKey(optionName, optionValue);
    if (isOptionCachedInVector(compilerOptionsState.supportedOptions, optionCacheKey)) {
        return;
    }

    if (!compilerOptionsState.privateSupportedOptions.has_value()) {
        compilerOptionsState.privateSupportedOptions = std::vector<std::string>{};
    }
    if (!isOptionCachedInVector(compilerOptionsState.privateSupportedOptions, optionCacheKey)) {
        compilerOptionsState.privateSupportedOptions->push_back(optionCacheKey);
    }
}

void CompilerOptionsCache::setSupportedOptions(const ov::intel_npu::CompilerType& compilerType,
                                               const std::vector<std::string>& supportedOptions) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto& compilerOptionsState = getStateForCompilerType(compilerType);
    if (compilerOptionsState.legacy) {
        OPENVINO_THROW("Cannot set supported options in CompilerOptionsCache when legacy mode is enabled.");
    }

    compilerOptionsState.supportedOptions = supportedOptions;
    if (!compilerOptionsState.privateSupportedOptions.has_value()) {
        return;
    }

    auto& privateOptions = compilerOptionsState.privateSupportedOptions.value();
    privateOptions.erase(std::remove_if(privateOptions.begin(),
                                        privateOptions.end(),
                                        [&](const std::string& value) {
                                            return std::find(supportedOptions.begin(), supportedOptions.end(), value) !=
                                                   supportedOptions.end();
                                        }),
                         privateOptions.end());
}

std::optional<std::vector<std::string>> CompilerOptionsCache::getSupportedOptions(
    const ov::intel_npu::CompilerType& compilerType) {
    std::lock_guard<std::mutex> lock(_mutex);
    const auto& compilerOptionsState = getStateForCompilerType(compilerType);
    if (compilerOptionsState.legacy) {
        return std::nullopt;
    }
    return compilerOptionsState.supportedOptions;
}

std::optional<std::vector<std::string>> CompilerOptionsCache::getPrivateSupportedOptions(
    const ov::intel_npu::CompilerType& compilerType) {
    std::lock_guard<std::mutex> lock(_mutex);
    const auto& compilerOptionsState = getStateForCompilerType(compilerType);
    if (compilerOptionsState.legacy) {
        return std::nullopt;
    }
    return compilerOptionsState.privateSupportedOptions;
}

void CompilerOptionsCache::setLegacyCompilerVersion(const ov::intel_npu::CompilerType& compilerType,
                                                    uint32_t compilerVersion) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto& compilerOptionsState = getStateForCompilerType(compilerType);

    if (compilerOptionsState.supportedOptions.has_value()) {
        OPENVINO_THROW("Cannot switch CompilerOptionsCache to legacy mode after supported options were initialized.");
    }

    compilerOptionsState.legacy = true;
    compilerOptionsState.compilerVersion = compilerVersion;
}

CompilerOptionsCache::CompilerTypeOptionsState& CompilerOptionsCache::getStateForCompilerType(
    const ov::intel_npu::CompilerType& compilerType) {
    if (compilerType == ov::intel_npu::CompilerType::DRIVER) {
        return _driverCompilerOptionsState;
    }

    if (compilerType == ov::intel_npu::CompilerType::PLUGIN) {
        return _pluginCompilerOptionsState;
    }

    OPENVINO_THROW("Unsupported compiler type in CompilerOptionsCache. Expected DRIVER or PLUGIN.");
}

std::string CompilerOptionsCache::buildOptionCacheKey(const std::string& optionName,
                                                      const std::optional<std::string>& optionValue) {
    if (!optionValue.has_value()) {
        return optionName;
    }

    std::string key = optionName;
    key += '=';
    key += optionValue.value();
    return key;
}

bool CompilerOptionsCache::isOptionCachedInVector(const std::optional<std::vector<std::string>>& options,
                                                  const std::string& optionCacheKey) {
    if (!options.has_value()) {
        return false;
    }

    const auto& values = options.value();
    return std::find(values.begin(), values.end(), optionCacheKey) != values.end();
}

}  // namespace intel_npu
