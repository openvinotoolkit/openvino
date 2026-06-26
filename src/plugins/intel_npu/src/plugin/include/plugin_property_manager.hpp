// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/config/npuw.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "metrics.hpp"
#include "property_registration.hpp"

namespace intel_npu {

class PluginPropertyManager final {
public:
    PluginPropertyManager(const FilteredConfig& config,
                          const std::shared_ptr<Metrics>& metrics,
                          const ov::SoPtr<IEngineBackend>& backend,
                          Logger& logger);

    PluginPropertyManager& operator=(const PluginPropertyManager& other) = delete;

    void setProperty(const ov::AnyMap& properties);
    ov::Any getProperty(const std::string& name, const ov::AnyMap& arguments = {}) const;
    bool isPropertySupported(const std::string& name, const ov::AnyMap& arguments = {}) const;

    const FilteredConfig& getConfig() const {
        return _config;
    }

    FilteredConfig getConfigWithCompilerPropertiesDisabled(const ov::AnyMap& properties) const;
    FilteredConfig getConfigForSpecificCompiler(const ov::AnyMap& properties, const ICompilerAdapter* compiler) const;

    std::string determinePlatform(const ov::AnyMap& properties) const;
    std::string determineDeviceId(const ov::AnyMap& properties) const;
    ov::intel_npu::CompilerType determineCompilerType(const ov::AnyMap& properties) const;

private:
    PluginPropertyManager(const PluginPropertyManager& other);
    struct CopyState {
        FilteredConfig config;
        std::shared_ptr<Metrics> metrics;
        ov::SoPtr<IEngineBackend> backend;
        Logger& logger;
        ov::intel_npu::CompilerType currentlyUsedCompiler;
        ov::intel_npu::CompilerType _compilerForCompatibilityCheck;
        bool compatibilityCheckSupported;
        std::string currentlyUsedPlatform;
        bool compilerConfigsFilteredByCompiler;
        bool compatibilityCheckFiltered;
    };

    explicit PluginPropertyManager(CopyState&& state);

    void registerProperties() const;
    void initializeCompatibilityCheckSupportIfNeeded() const;
    bool isPropertyRegistered(const std::string& propertyName) const;

    mutable FilteredConfig _config;

    std::shared_ptr<Metrics> _metrics;
    ov::SoPtr<IEngineBackend> _backend;
    Logger& _logger;

    mutable ov::intel_npu::CompilerType _currentlyUsedCompiler = ov::intel_npu::CompilerType::PREFER_PLUGIN;
    mutable ov::intel_npu::CompilerType _compilerForCompatibilityCheck = ov::intel_npu::CompilerType::DRIVER;
    mutable bool _compatibilityCheckSupported = false;
    mutable std::string _currentlyUsedPlatform;
    mutable bool _compilerConfigsFilteredByCompiler = false;
    mutable bool _compatibilityCheckFiltered = false;

    mutable std::map<std::string, PropertyDescriptor> _properties;
    mutable std::vector<ov::PropertyName> _supportedProperties;

    const std::vector<ov::PropertyName> _cachingProperties = [] {
        std::vector<ov::PropertyName> properties = {
            ov::cache_mode.name(),
            ov::enable_profiling.name(),
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

    const std::vector<ov::PropertyName> _internalSupportedProperties = {ov::internal::caching_properties.name(),
                                                                        ov::internal::caching_with_mmap.name(),
                                                                        ov::internal::cache_header_alignment.name()};

    mutable std::mutex _mutex;
};

}  // namespace intel_npu
