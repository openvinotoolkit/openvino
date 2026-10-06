// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "compiler_option_support_helper.hpp"
#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "property_registration.hpp"

namespace intel_npu {

enum class ConfigMergeMode { Compile, Import, Query };

struct MergedConfig {
    // Runtime config, compile-time-only and internal compiler options are removed from it.
    Config runtimeConfig;
    // Compile-time, both-mode and internal compiler options supported by the resolved compiler, serialized as strings.
    // Includes the values set through set_property and environment variables, overridden by the merged properties.
    std::map<std::string, std::string> compilerProperties;
    // Properties unknown to the plugin, forwarded as they are to the compiled model.
    ov::AnyMap unknownProperties;
};

class PluginPropertyManager final : private PropertyRegistrationBase {
public:
    PluginPropertyManager(const std::shared_ptr<OptionsDesc>& options,
                          const ov::SoPtr<IEngineBackend>& backend,
                          const std::shared_ptr<CompilerOptionSupportHelper>& optionSupportHelper,
                          Logger& logger);

    PluginPropertyManager& operator=(const PluginPropertyManager& other) = delete;

    void setProperty(const ov::AnyMap& properties);
    ov::Any getProperty(const std::string& name, const ov::AnyMap& arguments = {}) const;
    bool isPropertySupported(const std::string& name, const ov::AnyMap& arguments = {}) const;

    /**
     * @brief Merges the given properties into a copy of the plugin config for the compile or query path.
     * @param mergeMode Either ConfigMergeMode::Compile or ConfigMergeMode::Query.
     * @return The merged config, the properties to be sent to the compiler and the unknown properties.
     */
    MergedConfig getMergedConfigForCompilation(const ov::AnyMap& properties, ConfigMergeMode mergeMode);

    /**
     * @brief Merges the given properties into a copy of the plugin config for the import path. Compile-time-only
     * options are skipped, so no compiler properties are produced.
     * @return The merged runtime config and the unknown properties.
     */
    std::pair<Config, ov::AnyMap> getMergedConfigForImport(const ov::AnyMap& properties);

    std::string determinePlatform(const ov::AnyMap& properties) const;
    std::string determineDeviceId(const ov::AnyMap& properties) const;
    ov::intel_npu::CompilerType determineCompilerType(const ov::AnyMap& properties) const;

private:
    void registerProperties();

    // Merges the given properties into a copy of the stored config. The returned compiler properties hold only the
    // internal compiler options (stored ones overridden by the passed ones), the caller is responsible for adding the
    // compile-time options and for cleaning up the runtime config. Doesn't lock _mutex, callers must hold it.
    MergedConfig mergeConfig(const ov::AnyMap& properties, ConfigMergeMode mergeMode);

    // The helpers below read the value from the arguments and fall back to the stored config when missing.
    // They don't lock _mutex, callers must hold it.
    std::string getDeviceIdOrDefault(const ov::AnyMap& arguments) const;
    std::string getPlatformOrDefault(const ov::AnyMap& arguments) const;
    std::optional<ov::intel_npu::CompilerType> getCompilerTypeOrDefault(const ov::AnyMap& arguments) const;
    std::optional<ov::intel_npu::CompilerType> resolveCompilerType(const ov::AnyMap& arguments) const;

    void warnCompilerOnlyOptionSkipped(const std::string& key) const;

    Config _config;
    // Internal compiler options set through set_property. They are unknown to the plugin, only the compiler supports
    // them, so they are kept here and sent to the compiler through the compiler properties instead of the config.
    std::map<std::string, std::string> _internalCompilerProperties;

    ov::SoPtr<IEngineBackend> _backend;
    std::shared_ptr<CompilerOptionSupportHelper> _compilerOptionSupportHelper;
    Logger& _logger;

    mutable std::mutex _mutex;

    const std::vector<ov::PropertyName> _cachingProperties = [] {
        std::vector<ov::PropertyName> properties = {
            ov::cache_mode.name(),
            ov::enable_profiling.name(),
            ov::intel_npu::profiling_type.name(),
            ov::device::architecture.name(),
            ov::hint::execution_mode.name(),
            ov::hint::inference_precision.name(),
            ov::hint::performance_mode.name(),
            ov::intel_npu::batch_compiler_mode_settings.name(),
            ov::intel_npu::batch_mode.name(),
            ov::intel_npu::compilation_mode.name(),
            ov::intel_npu::compilation_mode_params.name(),
            ov::intel_npu::compiler_dynamic_quantization.name(),
            ov::intel_npu::compiler_type.name(),
            ov::intel_npu::dma_engines.name(),
            ov::intel_npu::driver_version.name(),
            ov::intel_npu::dynamic_shape_to_static.name(),
            ov::intel_npu::enable_strides_for.name(),
            ov::intel_npu::max_tiles.name(),
            ov::intel_npu::stepping.name(),
            ov::intel_npu::tiles.name(),
            ov::intel_npu::turbo.name(),
            ov::intel_npu::qdq_optimization.name(),
            ov::intel_npu::qdq_optimization_aggressive.name(),
        };
        for_each_cached_npuw_option([&](auto tag) {
            using Opt = typename decltype(tag)::type;
            properties.emplace_back(std::string{Opt::key()});
        });
        return properties;
    }();
};

}  // namespace intel_npu
