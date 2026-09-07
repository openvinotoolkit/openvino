// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/option_support_cache.hpp"

#include <algorithm>

#include "openvino/core/except.hpp"

namespace {

std::optional<bool> getCachedOptionSupportFromVector(
    const std::vector<intel_npu::OptionSupportCache::OptionSupportState>& options,
    const std::string& optionCacheKey) {
    const auto it = std::find_if(options.begin(),
                                 options.end(),
                                 [&](const intel_npu::OptionSupportCache::OptionSupportState& value) {
                                     return value.optionCacheKey == optionCacheKey;
                                 });
    if (it == options.end()) {
        return std::nullopt;
    }

    return it->supported;
}

void addOptionSupportInVector(std::vector<intel_npu::OptionSupportCache::OptionSupportState>& options,
                              const std::string& optionCacheKey,
                              bool supported) {
    const auto it = std::find_if(options.begin(),
                                 options.end(),
                                 [&](const intel_npu::OptionSupportCache::OptionSupportState& value) {
                                     return value.optionCacheKey == optionCacheKey;
                                 });
    if (it != options.end()) {
        if (it->supported == supported) {
            return;
        }

        OPENVINO_THROW("Attempting to add an option support state for option '",
                       optionCacheKey,
                       "' that already exists with a different support state. Existing: ",
                       it->supported,
                       ", New: ",
                       supported);
    }

    options.push_back({optionCacheKey, supported});
}

}  // namespace

namespace intel_npu {

std::optional<bool> OptionSupportCache::isOptionSupported(const CacheKey key, const std::string& optionName) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto& optionSupportState = getStateForKey(key);
    return getCachedOptionSupportFromVector(optionSupportState.supportedOptions, optionName);
}

void OptionSupportCache::addSupportedOption(const CacheKey key, const std::string& optionName, bool supported) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto& optionSupportState = getStateForKey(key);
    addOptionSupportInVector(optionSupportState.supportedOptions, optionName, supported);
}

void OptionSupportCache::setSupportedOptions(const CacheKey key, const std::vector<std::string>& supportedOptions) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto& optionSupportState = getStateForKey(key);

    for (const auto& option : supportedOptions) {
        addOptionSupportInVector(optionSupportState.supportedOptions, option, true);
    }
}

OptionSupportCache::KeyOptionsState& OptionSupportCache::getStateForKey(CacheKey key) {
    auto it = std::find_if(_optionSupportStates.begin(),
                           _optionSupportStates.end(),
                           [&](const KeyOptionsState& optionSupportState) {
                               return optionSupportState.key == key;
                           });
    if (it != _optionSupportStates.end()) {
        return *it;
    }

    _optionSupportStates.push_back({key, {}});
    return _optionSupportStates.back();
}

}  // namespace intel_npu
