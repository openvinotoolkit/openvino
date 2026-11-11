// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
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
 * A factory knows how to build a compiler-in-plugin only from the entry points of the provider it was given. A plugin
 * has one VCL loader holder, which loads the library at most once and holds one set of entry points, and one factory.
 * Entry points from another provider need another factory.
 *
 * The factory is not copyable: share one instance, for example through a
 * `std::shared_ptr<const CompilerAdapterFactory>`, among everything that works with the same provider, so that all of
 * it sees the same outcome and a failed attempt is not repeated. Never share an instance across providers.
 */
class CompilerAdapterFactory final {
public:
    /**
     * @brief Creates a factory that obtains VCL entry points from the given provider.
     *
     * Driver-only operations do not require a provider; a missing provider is rejected only if a VCL compiler is
     * actually needed.
     */
    explicit CompilerAdapterFactory(const VCLFunctionTableProvider& vclFunctionsProvider);

    CompilerAdapterFactory(const CompilerAdapterFactory&) = delete;
    CompilerAdapterFactory& operator=(const CompilerAdapterFactory&) = delete;

    /**
     * @brief Creates the compiler adapter and writes the resolved type back to `compilerType`.
     *
     * PREFER_PLUGIN is resolved like in decideCompilerType(), and a compiler that the probe creates is returned as is.
     * A request that resolved to PLUGIN without probing, or that asked for PLUGIN explicitly, creates the
     * compiler-in-plugin and throws if that fails; there is no fallback to the driver compiler at that point. An
     * explicit PLUGIN request is not remembered: it asks the provider again. A provider that loads the library once,
     * like the loader holder of the plugin, rethrows its first load failure instead of loading again.
     *
     * @param engineBackend The backend that provides the device and the Level Zero structures. May be empty for
     *        offline compilation with the compiler-in-plugin.
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
        const ov::SoPtr<IEngineBackend>& engineBackend,
        ov::intel_npu::CompilerType& compilerType,
        std::string_view platform,
        const std::shared_ptr<OptionSupportCache>& optionSupportCache = nullptr) const;

    /**
     * @brief Resolves PREFER_PLUGIN to PLUGIN or DRIVER. Any other `compilerType` is left unchanged.
     *
     * Meant for validating properties against the compiler that will be used, so PLUGIN means that a
     * compiler-in-plugin could really be created. The first resolution that needs to know creates one and discards
     * it. The factory remembers the outcome, so later resolutions, including the ones of getCompiler(), do not probe
     * again. A platform that only the driver compiler serves resolves to DRIVER without probing, when the
     * compilation is online. Threads that race on the first resolution may each probe, with the same outcome.
     *
     * @param[in,out] compilerType The requested compiler type on input; the resolved type on output.
     * @param device The device that runs the compiled model. May be null for offline compilation.
     * @param platform The compilation target platform. Empty means no explicit target, which is treated as online
     *        compilation on `device`.
     * @throws ov::Exception if the compilation is offline (or cross compilation) and the compiler-in-plugin cannot
     *         be created, since only it can serve such a compilation.
     */
    void decideCompilerType(ov::intel_npu::CompilerType& compilerType,
                            const std::shared_ptr<intel_npu::IDevice>& device,
                            std::string_view platform) const;

    static const std::vector<ov::intel_npu::CompilerType>& getKnownCompilerTypes();

private:
    ov::intel_npu::CompilerType determineAppropriateCompilerTypeBasedOnPlatform(std::string_view platform) const;

    std::pair<std::unique_ptr<ICompilerAdapter>, ov::intel_npu::CompilerType> resolvePreferPluginCompiler(
        const ov::SoPtr<IEngineBackend>& engineBackend,
        const std::shared_ptr<OptionSupportCache>& optionSupportCache,
        const std::shared_ptr<intel_npu::IDevice>& device,
        std::string_view platform) const;

    enum class PluginCompilerPresence : std::uint8_t {
        UNKNOWN = 0,  // Not probed yet.
        PRESENT = 1,  // A compiler-in-plugin was created from the provider.
        ABSENT = 2,   // Creating a compiler-in-plugin from the provider failed.
    };

    // Outcome of the compiler-in-plugin probe for this factory's provider. The factory is not copyable, so everything
    // sharing one instance sees the same outcome, and a factory for another provider probes on its own.
    mutable std::atomic<PluginCompilerPresence> _pluginCompilerPresence{PluginCompilerPresence::UNKNOWN};
    VCLFunctionTableProvider _vclFunctionsProvider;
};

}  // namespace intel_npu
