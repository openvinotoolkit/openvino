// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {

class ICompilerAdapter;

class CompilerOptionsCache final {
public:
    static bool isOptionSupported(ov::intel_npu::CompilerType compilerType,
                                  const std::string& optionName,
                                  const std::optional<std::string>& optionValue = std::nullopt,
                                  const ICompilerAdapter* compiler = nullptr,
                                  const std::optional<uint32_t>& compilerSupportVersion = std::nullopt);

private:
    struct CompilerTypeOptionsState final {
        bool supportedOptionsQueried = false;
        std::optional<std::vector<std::string>> supportedOptions;
        bool legacy = false;
        uint32_t compilerVersion = 0;
    };

    static CompilerTypeOptionsState& getStateForCompilerType(ov::intel_npu::CompilerType compilerType);

    static std::string buildOptionCacheKey(const std::string& optionName,
                                           const std::optional<std::string>& optionValue);

    static std::string buildLegacyOptionCacheKey(const std::string& optionCacheKey, uint32_t compilerSupportVersion);

    static bool isOptionCached(const CompilerTypeOptionsState& compilerOptionsState, const std::string& optionCacheKey);

    static bool isLegacyOptionCached(const CompilerTypeOptionsState& compilerOptionsState,
                                     const std::string& optionCacheKey,
                                     const std::optional<uint32_t>& compilerSupportVersion);

    static void addSupportedOptionImpl(CompilerTypeOptionsState& compilerOptionsState, const std::string& optionName);

    static std::mutex _mutex;
    static CompilerTypeOptionsState _driverCompilerOptionsState;
    static CompilerTypeOptionsState _pluginCompilerOptionsState;
};

}  // namespace intel_npu
