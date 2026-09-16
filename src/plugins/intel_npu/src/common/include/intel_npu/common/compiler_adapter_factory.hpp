// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/common/option_support_cache.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {

class CompilerAdapterFactory final {
public:
    std::unique_ptr<ICompilerAdapter> getCompiler(
        const ov::SoPtr<IEngineBackend>& engineBackend,
        ov::intel_npu::CompilerType& compilerType,
        std::string_view platform,
        const std::shared_ptr<OptionSupportCache>& optionSupportCache = nullptr) const;

    void decideCompilerType(ov::intel_npu::CompilerType& compilerType,
                            const std::shared_ptr<intel_npu::IDevice>& device,
                            std::string_view platform);

    static const std::vector<ov::intel_npu::CompilerType>& getKnownCompilerTypes();

private:
    ov::intel_npu::CompilerType determineAppropriateCompilerTypeBasedOnPlatform(std::string_view platform) const;

    std::pair<std::unique_ptr<ICompilerAdapter>, ov::intel_npu::CompilerType> resolvePreferPluginCompiler(
        const ov::SoPtr<IEngineBackend>& engineBackend,
        const std::shared_ptr<OptionSupportCache>& optionSupportCache,
        const std::shared_ptr<intel_npu::IDevice>& device,
        std::string_view platform) const;

    enum class PluginCompilerPresence : std::uint8_t {
        UNKNOWN = 0,
        PRESENT = 1,
        ABSENT = 2,
    };

    inline static std::atomic<PluginCompilerPresence> _pluginCompilerPresence{PluginCompilerPresence::UNKNOWN};
};

}  // namespace intel_npu
