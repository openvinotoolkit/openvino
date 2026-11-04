// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/common/option_support_cache.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {

class CompilerAdapterFactory final {
public:
    /**
     * @param vclFunctions Supplies the VCL entry points. This factory calls it only to build or probe a
     *        compiler-in-plugin. Driver compilation never calls it. The factory keeps no availability state:
     *        the provider owns it and should remember a failed load (VCLLoaderHolder does).
     */
    explicit CompilerAdapterFactory(const VCLFunctionTableProvider& vclFunctions);

    /**
     * @brief Creates the compiler adapter and writes the resolved type back to `compilerType`.
     *
     * A PREFER_PLUGIN request that the driver compiler can serve (online compilation) falls back to it when the
     * plugin compiler cannot be loaded or created. An explicit PLUGIN request and offline compilation throw.
     */
    std::unique_ptr<ICompilerAdapter> getCompiler(
        const ov::SoPtr<IEngineBackend>& engineBackend,
        ov::intel_npu::CompilerType& compilerType,
        std::string_view platform,
        const std::shared_ptr<OptionSupportCache>& optionSupportCache = nullptr) const;

    /**
     * @brief Resolves PREFER_PLUGIN to PLUGIN or DRIVER without creating a compiler.
     *
     * Unlike the previous compiler construction probe, this decision checks only whether the provider can
     * supply VCL entry points. Obtaining the entry points does not guarantee that a VCL compiler can be created.
     * getCompiler() performs construction and falls back to the driver compiler for an eligible PREFER_PLUGIN
     * request if construction fails.
     */
    void decideCompilerType(ov::intel_npu::CompilerType& compilerType,
                            const std::shared_ptr<intel_npu::IDevice>& device,
                            std::string_view platform);

    static const std::vector<ov::intel_npu::CompilerType>& getKnownCompilerTypes();

private:
    ov::intel_npu::CompilerType determineAppropriateCompilerTypeBasedOnPlatform(std::string_view platform) const;

    ov::intel_npu::CompilerType resolvePreferPluginCompiler(const std::shared_ptr<intel_npu::IDevice>& device,
                                                            std::string_view platform) const;

    VCLFunctionTableProvider _vclFunctions;
};

}  // namespace intel_npu
