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
#include "intel_npu/utils/vcl/vcl_function_table_provider.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {

/**
 * @brief Creates an adapter for the selected NPU compiler.
 *
 * A factory is bound to one backend and one VCL provider. A plugin has one VCL loader holder, which loads the library
 * at most once and holds one set of entry points, and one factory for its backend and provider.
 *
 * Share one instance, for example through a `std::shared_ptr<const CompilerAdapterFactory>`, among requests that use
 * the same backend and provider.
 */
class CompilerAdapterFactory final {
public:
    /**
     * @brief Creates a factory for the given backend and VCL provider.
     *
     * Driver-only operations do not require a provider; a missing provider is rejected only if a VCL compiler is
     * actually needed.
     * @param engineBackend The backend that provides the device and Level Zero structures. May be empty for offline
     *        compilation with the compiler-in-plugin.
     * @param vclFunctionsProvider Supplies VCL entry points when a compiler-in-plugin is needed.
     */
    CompilerAdapterFactory(const ov::SoPtr<IEngineBackend>& engineBackend,
                           const VCLFunctionTableProvider& vclFunctionsProvider);

    CompilerAdapterFactory(const CompilerAdapterFactory&) = delete;
    CompilerAdapterFactory& operator=(const CompilerAdapterFactory&) = delete;

    /**
     * @brief Creates the compiler adapter and writes the resolved type back to `compilerType`.
     *
     * PREFER_PLUGIN is resolved like in decideCompilerType(): resolution only loads the function table. If loading
     * fails during online compilation, the request falls back to DRIVER. A later compiler creation failure, or an
     * explicit PLUGIN request failure, throws without fallback. A provider that loads the library once, like the
     * loader holder of the plugin, rethrows its first load failure.
     *
     * @param[in,out] compilerType The requested compiler type on input; the type of the returned adapter on output
     *        (never PREFER_PLUGIN).
     * @param platform The compilation target platform. Empty means no explicit target, which is treated as online
     *        compilation on the backend device.
     * @param optionSupportCache Where the compiler records which options it supports. May be null.
     * @return The compiler adapter. Never null.
     * @throws ov::Exception if the requested compiler cannot be created, if the driver compiler is selected without a
     *         device matching `platform`, or if `compilerType` is invalid.
     */
    std::unique_ptr<ICompilerAdapter> getCompiler(
        ov::intel_npu::CompilerType& compilerType,
        std::string_view platform,
        const std::shared_ptr<OptionSupportCache>& optionSupportCache = nullptr) const;

    /**
     * @brief Resolves PREFER_PLUGIN to PLUGIN or DRIVER. Any other `compilerType` is left unchanged.
     *
     * Meant for validating properties against the selected compiler type. Resolving PREFER_PLUGIN only loads the
     * function table; it does not verify that a compiler can be created. An online function-table load failure falls
     * back to DRIVER, while an offline or cross-compilation load failure throws. A platform that only the driver
     * compiler serves resolves to DRIVER without loading VCL when compilation is online.
     *
     * @param[in,out] compilerType The requested compiler type on input; the resolved type on output.
     * @param device The device that runs the compiled model. May be null for offline compilation.
     * @param platform The compilation target platform. Empty means no explicit target, which is treated as online
     *        compilation on `device`.
     * @throws ov::Exception if loading the function table fails during offline (or cross) compilation, since only the
     *         compiler-in-plugin can serve such a compilation.
     */
    void decideCompilerType(ov::intel_npu::CompilerType& compilerType,
                            const std::shared_ptr<intel_npu::IDevice>& device,
                            std::string_view platform) const;

    static const std::vector<ov::intel_npu::CompilerType>& getKnownCompilerTypes();

private:
    ov::intel_npu::CompilerType determineAppropriateCompilerTypeBasedOnPlatform(std::string_view platform) const;

    ov::intel_npu::CompilerType resolvePreferPluginCompiler(const std::shared_ptr<intel_npu::IDevice>& device,
                                                            std::string_view platform) const;

    ov::SoPtr<IEngineBackend> _engineBackend;
    VCLFunctionTableProvider _vclFunctionsProvider;
};

}  // namespace intel_npu
