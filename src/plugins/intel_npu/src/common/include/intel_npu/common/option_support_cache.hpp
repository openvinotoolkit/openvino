// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace intel_npu {

class OptionSupportCache final {
public:
    using CacheKey = uint32_t;

    bool isOptionSupported(CacheKey key,
                           const std::string& optionName,
                           const std::optional<std::string>& optionValue = std::nullopt);

    void addSupportedOption(CacheKey key,
                            const std::string& optionName,
                            const std::optional<std::string>& optionValue = std::nullopt);

    void setSupportedOptions(CacheKey key,
                             const std::vector<std::string>& supportedOptions);

private:
    struct KeyOptionsState final {
        CacheKey key;
        std::optional<std::vector<std::string>> supportedOptions;
    };

    KeyOptionsState& getStateForKey(CacheKey key);

    static std::string buildOptionCacheKey(const std::string& optionName,
                                           const std::optional<std::string>& optionValue);

    static bool isOptionCachedInVector(const std::optional<std::vector<std::string>>& options,
                                       const std::string& optionCacheKey);

    std::mutex _mutex;
    std::vector<KeyOptionsState> _optionSupportStates;
};

}  // namespace intel_npu
