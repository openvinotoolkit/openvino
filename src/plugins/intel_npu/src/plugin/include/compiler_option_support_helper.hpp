// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "intel_npu/common/compiler_supported_options_cache.hpp"
#include "intel_npu/common/npu.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {

class CompilerOptionSupportHelper final {
public:
    explicit CompilerOptionSupportHelper(const ov::SoPtr<IEngineBackend>& backend);

    const std::shared_ptr<CompilerSupportedOptionsCache>& getCompilerSupportedOptionsCache() const;

    bool isOptionSupported(ov::intel_npu::CompilerType compilerType,
                           const std::string& optionName,
                           const std::optional<std::string>& optionValue = std::nullopt);

private:
    const ov::SoPtr<IEngineBackend> _backend;
    std::shared_ptr<CompilerSupportedOptionsCache> _supportedOptionsCache;

    std::atomic<bool> _driverSupportedOptionsLoaded{false};
    std::atomic<bool> _pluginSupportedOptionsLoaded{false};
};

}  // namespace intel_npu
