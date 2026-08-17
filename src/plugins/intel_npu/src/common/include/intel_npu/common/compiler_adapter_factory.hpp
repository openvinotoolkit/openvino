// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <memory>
#include <string_view>
#include <vector>

#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/common/option_support_cache.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {

class CompilerAdapterFactory final {
public:
    ov::intel_npu::CompilerType determineAppropriateCompilerTypeBasedOnPlatform(std::string_view platform) const;

    std::unique_ptr<ICompilerAdapter> getCompiler(
        const ov::SoPtr<IEngineBackend>& engineBackend,
        ov::intel_npu::CompilerType& compilerType,
        std::string_view platform,
        const std::shared_ptr<OptionSupportCache>& optionSupportCache = nullptr) const;

    static const std::vector<ov::intel_npu::CompilerType>& getSupportedCompilerTypes();

private:
    inline static std::atomic<bool> _pluginCompilerIsPresent{true};
};

}  // namespace intel_npu
