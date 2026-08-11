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

class CompilerSupportedOptionsCache final {
public:
    bool isOptionSupported(const ov::intel_npu::CompilerType& compilerType,
                           const std::string& optionName,
                           const std::optional<std::string>& optionValue = std::nullopt);

    void addSupportedOption(const ov::intel_npu::CompilerType& compilerType,
                            const std::string& optionName,
                            const std::optional<std::string>& optionValue = std::nullopt);

    void setSupportedOptions(const ov::intel_npu::CompilerType& compilerType,
                             const std::vector<std::string>& supportedOptions);

private:
    struct CompilerTypeOptionsState final {
        std::optional<std::vector<std::string>> supportedOptions;
    };

    CompilerTypeOptionsState& getStateForCompilerType(const ov::intel_npu::CompilerType& compilerType);

    static std::string buildOptionCacheKey(const std::string& optionName,
                                           const std::optional<std::string>& optionValue);

    static bool isOptionCachedInVector(const std::optional<std::vector<std::string>>& options,
                                       const std::string& optionCacheKey);

    std::mutex _mutex;
    CompilerTypeOptionsState _driverCompilerOptionsState;
    CompilerTypeOptionsState _pluginCompilerOptionsState;
};

}  // namespace intel_npu
