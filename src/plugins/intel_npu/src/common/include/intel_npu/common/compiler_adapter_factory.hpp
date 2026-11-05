// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
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
     * @brief Creates a factory that obtains the VCL entry points from the given provider.
     *
     * The factory calls the provider only to resolve PREFER_PLUGIN or to create a compiler-in-plugin; driver
     * compilation never calls it. The factory keeps no availability state: the provider owns it and should remember
     * a failed load (VCLLoaderHolder does).
     *
     * @param vclFunctions Supplies the VCL entry points. Must not be empty.
     * @throws ov::Exception if the provider is empty.
     */
    explicit CompilerAdapterFactory(const VCLFunctionTableProvider& vclFunctions);

    /**
     * @brief Creates the compiler adapter and writes the resolved type back to `compilerType`.
     *
     * A PREFER_PLUGIN request that the driver compiler can serve (online compilation) falls back to it when the
     * plugin compiler cannot be loaded or created. An explicit PLUGIN request and offline compilation do not fall
     * back.
     *
     * @param engineBackend The backend that provides the device and the Level Zero structures. May be empty for
     *        offline compilation with the compiler-in-plugin.
     * @param[in,out] compilerType The requested compiler type on input; the type of the returned adapter on output
     *        (never PREFER_PLUGIN).
     * @param platform The compilation target platform. Empty means no explicit target, which is treated as online
     *        compilation on the backend device.
     * @param optionSupportCache Where the compiler records which options it supports. May be null.
     * @return The compiler adapter. Never null.
     * @throws ov::Exception if the requested compiler cannot be created and no fallback applies, if the driver
     *         compiler is selected without a device matching `platform`, or if `compilerType` is invalid.
     */
    std::unique_ptr<ICompilerAdapter> getCompiler(
        const ov::SoPtr<IEngineBackend>& engineBackend,
        ov::intel_npu::CompilerType& compilerType,
        std::string_view platform,
        const std::shared_ptr<OptionSupportCache>& optionSupportCache = nullptr) const;

    /**
     * @brief Resolves PREFER_PLUGIN to PLUGIN or DRIVER without creating a compiler.
     *
     * PLUGIN means that the provider can supply the VCL entry points, not that a compiler can be created from them.
     * getCompiler() creates the compiler and, for an eligible PREFER_PLUGIN request, falls back to the driver
     * compiler if the creation fails. The two can therefore disagree. Any other `compilerType` is left unchanged.
     *
     * @param[in,out] compilerType The requested compiler type on input; the resolved type on output.
     * @param device The device that runs the compiled model. May be null for offline compilation.
     * @param platform The compilation target platform. Empty means no explicit target, which is treated as online
     *        compilation on `device`.
     * @throws ov::Exception if the compilation is offline (or cross compilation) and the provider cannot supply the
     *         VCL entry points, since only the compiler-in-plugin can serve it.
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
