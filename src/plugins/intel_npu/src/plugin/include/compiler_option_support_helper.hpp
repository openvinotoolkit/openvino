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
#include "intel_npu/common/option_support_cache.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {

class CompilerOptionSupportHelper final {
public:
    explicit CompilerOptionSupportHelper(const ov::SoPtr<IEngineBackend>& backend);

    const std::shared_ptr<OptionSupportCache>& getOptionSupportCache() const;

    bool isOptionSupported(ov::intel_npu::CompilerType compilerType,
                           const std::string& optionName,
                           const std::optional<std::string>& optionValue = std::nullopt);

private:
    const ov::SoPtr<IEngineBackend> _backend;
    std::shared_ptr<OptionSupportCache> _optionSupportCache;

    std::once_flag _driverSupportedOptionsLoaded;
    std::once_flag _pluginSupportedOptionsLoaded;
};

}  // namespace intel_npu
