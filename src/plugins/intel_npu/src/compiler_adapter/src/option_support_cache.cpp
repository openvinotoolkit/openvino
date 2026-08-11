// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/option_support_cache.hpp"

#include <algorithm>

namespace intel_npu {

bool OptionSupportCache::isOptionSupported(const CacheKey key,
                                           const std::string& optionName,
                                           const std::optional<std::string>& optionValue) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto& optionSupportState = getStateForKey(key);
    const auto optionCacheKey = buildOptionCacheKey(optionName, optionValue);
    if (isOptionCachedInVector(optionSupportState.supportedOptions, optionCacheKey)) {
        return true;
    }

    return false;
}

void OptionSupportCache::addSupportedOption(const CacheKey key,
                                            const std::string& optionName,
                                            const std::optional<std::string>& optionValue) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto& optionSupportState = getStateForKey(key);

    const auto optionCacheKey = buildOptionCacheKey(optionName, optionValue);
    if (!optionSupportState.supportedOptions.has_value()) {
        optionSupportState.supportedOptions = std::vector<std::string>{};
    }
    if (!isOptionCachedInVector(optionSupportState.supportedOptions, optionCacheKey)) {
        optionSupportState.supportedOptions->push_back(optionCacheKey);
    }
}

void OptionSupportCache::setSupportedOptions(const CacheKey key,
                                             const std::vector<std::string>& supportedOptions) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto& optionSupportState = getStateForKey(key);

    if (!optionSupportState.supportedOptions.has_value()) {
        optionSupportState.supportedOptions = supportedOptions;
        return;
    }

    for (const auto& option : supportedOptions) {
        if (!isOptionCachedInVector(optionSupportState.supportedOptions, option)) {
            optionSupportState.supportedOptions->push_back(option);
        }
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

    _optionSupportStates.push_back({key, std::nullopt});
    return _optionSupportStates.back();
}

std::string OptionSupportCache::buildOptionCacheKey(const std::string& optionName,
                                                    const std::optional<std::string>& optionValue) {
    if (!optionValue.has_value()) {
        return optionName;
    }

    std::string key = optionName;
    key += '=';
    key += optionValue.value();
    return key;
}

bool OptionSupportCache::isOptionCachedInVector(const std::optional<std::vector<std::string>>& options,
                                                const std::string& optionCacheKey) {
    if (!options.has_value()) {
        return false;
    }

    const auto& values = options.value();
    return std::find(values.begin(), values.end(), optionCacheKey) != values.end();
}

}  // namespace intel_npu
