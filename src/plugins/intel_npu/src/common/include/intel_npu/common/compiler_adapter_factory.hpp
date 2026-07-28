// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <string_view>

#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/common/npu.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"

namespace intel_npu {

class CompilerAdapterFactory final {
public:
    ov::intel_npu::CompilerType determineAppropriateCompilerTypeBasedOnPlatform(std::string_view platform) const;

    /**
     * @brief Creates a compiler adapter appropriate for the given backend, compiler type and platform.
     * @param engineBackend The backend used to determine device availability and properties.
     * @param compilerType The requested compiler type; may be updated in place if PREFER_PLUGIN falls back to
     * DRIVER.
     * @param platform The target compilation platform.
     * @param compilerLogLevel Verbosity of the compiler's own logging at creation time. Only honored by the
     * in-process PLUGIN compiler; the DRIVER compiler receives its log level exclusively through the per-compile
     * build flags.
     * @return A compiler adapter for the resolved compiler type.
     */
    std::unique_ptr<ICompilerAdapter> getCompiler(
        const ov::SoPtr<IEngineBackend>& engineBackend,
        ov::intel_npu::CompilerType& compilerType,
        std::string_view platform,
        const std::optional<ov::log::Level>& compilerLogLevel = std::nullopt) const;

private:
    inline static std::atomic<bool> _pluginCompilerIsPresent{true};
};

}  // namespace intel_npu
