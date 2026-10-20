// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace intel_npu {

class OptionSupportCache final {
public:
    using CacheKey = uint32_t;

    struct OptionSupportState final {
        std::string optionCacheKey;
        bool supported;
    };

    std::optional<bool> isOptionSupported(CacheKey key, const std::string& optionName);
    void addSupportedOption(CacheKey key, const std::string& optionName, bool supported = true);
    void setSupportedOptions(CacheKey key, const std::vector<std::string>& supportedOptions);

private:
    struct KeyOptionsState final {
        CacheKey key;
        std::vector<OptionSupportState> supportedOptions;
    };

    KeyOptionsState& getStateForKey(CacheKey key);

    std::mutex _mutex;
    std::vector<KeyOptionsState> _optionSupportStates;
};

/**
 * @brief An option support cache bound to the cache key of a single compiler.
 * @note Copyable and default-constructible. A default-constructed (or null cache) instance is disabled: all
 *       queries answer "not cached" and all stores are dropped, so callers do not need to check for a cache.
 */
class ScopedOptionSupportCache final {
public:
    ScopedOptionSupportCache() = default;
    ScopedOptionSupportCache(std::shared_ptr<OptionSupportCache> cache, const OptionSupportCache::CacheKey key)
        : _cache(std::move(cache)),
          _key(key) {}

    /**
     * @brief Tells whether a cache is attached. Only needed by callers that want to skip work which is
     *        pointless without a cache; the accessors below are safe to call either way.
     */
    bool enabled() const {
        return _cache != nullptr;
    }

    std::optional<bool> isOptionSupported(const std::string& optionName) const {
        return _cache ? _cache->isOptionSupported(_key, optionName) : std::nullopt;
    }

    void addSupportedOption(const std::string& optionName, const bool supported = true) const {
        if (_cache) {
            _cache->addSupportedOption(_key, optionName, supported);
        }
    }

    void setSupportedOptions(const std::vector<std::string>& supportedOptions) const {
        if (_cache) {
            _cache->setSupportedOptions(_key, supportedOptions);
        }
    }

private:
    std::shared_ptr<OptionSupportCache> _cache;
    OptionSupportCache::CacheKey _key = 0;
};

}  // namespace intel_npu
