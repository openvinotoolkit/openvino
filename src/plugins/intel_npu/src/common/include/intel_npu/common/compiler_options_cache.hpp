// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "intel_npu/common/npu.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {

class ICompilerAdapter;
class DriverCompilerAdapter;
class PluginCompilerAdapter;

class CompilerOptionsCache final {
public:
    static bool isOptionSupported(const ov::intel_npu::CompilerType& compilerType,
                                  const std::string& optionName,
                                  const std::optional<std::string>& optionValue = std::nullopt,
                                  const std::optional<uint32_t>& compilerSupportVersion = std::nullopt);

    static std::optional<std::vector<std::string>> getSupportedOptions(const ov::intel_npu::CompilerType& compilerType);

    static std::optional<std::vector<std::string>> getPrivateSupportedOptions(
        const ov::intel_npu::CompilerType& compilerType);

private:
    friend class ICompilerAdapter;
    friend class DriverCompilerAdapter;
    friend class PluginCompilerAdapter;

    static void addSupportedOption(const ov::intel_npu::CompilerType& compilerType,
                                   const std::string& optionName,
                                   const std::optional<std::string>& optionValue = std::nullopt);

    static void setSupportedOptions(const ov::intel_npu::CompilerType& compilerType,
                                    const std::vector<std::string>& supportedOptions);

    static void setLegacyCompilerVersion(const ov::intel_npu::CompilerType& compilerType, uint32_t compilerVersion);

    struct CompilerTypeOptionsState final {
        std::optional<std::vector<std::string>> supportedOptions;
        std::optional<std::vector<std::string>> privateSupportedOptions;
        bool legacy = false;
        uint32_t compilerVersion = 0;
    };

    static CompilerTypeOptionsState& getStateForCompilerType(const ov::intel_npu::CompilerType& compilerType);

    static std::string buildOptionCacheKey(const std::string& optionName,
                                           const std::optional<std::string>& optionValue);

    static bool isOptionCachedImpl(const CompilerTypeOptionsState& compilerOptionsState,
                                   const std::string& optionCacheKey);

    static bool isOptionCachedInVector(const std::optional<std::vector<std::string>>& options,
                                       const std::string& optionCacheKey);

    static std::mutex _mutex;
    static CompilerTypeOptionsState _driverCompilerOptionsState;
    static CompilerTypeOptionsState _pluginCompilerOptionsState;
};

}  // namespace intel_npu
