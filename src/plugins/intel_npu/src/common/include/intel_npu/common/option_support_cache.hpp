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

}  // namespace intel_npu
