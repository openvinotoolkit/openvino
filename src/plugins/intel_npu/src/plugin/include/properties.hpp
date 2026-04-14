// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/config/npuw.hpp"
#include "metrics.hpp"

namespace intel_npu {

enum class PropertiesType { PLUGIN, COMPILED_MODEL };

class Properties final {
public:
    /**
     * @brief Properties handler constructor
     * @param pType - type of object this handler gets attached to: PLUGIN or COMPILED_MODEL
     * @param config - reference to the global configuration table of the parent object
     * @param metrics - reference ptr to the metrics object of the parent object (PLUGIN only)
     */
    Properties(const PropertiesType pType,
               const FilteredConfig& config,
               const std::shared_ptr<Metrics>& metrics = nullptr,
               const ov::SoPtr<IEngineBackend>& backend = {nullptr});

    Properties(const Properties& other);
    Properties& operator=(const Properties& other) = delete;

    /**
     * @brief Get the values of a property in a map
     */
    ov::Any getProperty(const std::string& name);

    /**
     * @brief Set the values of a subset of properties, provided as a map
     * @details
     * - checks if the property exists, will report if unsupported
     * - checks if the property is Read-only, will report error if so
     */
    void setProperty(const ov::AnyMap& properties);

    /**
     * @brief Checks if a property is supported by the plugin.
     */
    bool isPropertySupported(const std::string& name);

    /**
     * @brief Get a const reference to the stored config
     */
    const FilteredConfig& getConfig() const {
        return _config;
    }

    /**
     * @brief Updates a copy of the config list based on the provided properties, and returns it.
     * @details
     * - Updates the config with the provided arguments and returns it.
     */
    FilteredConfig getConfigWithCompilerPropertiesDisabled(const ov::AnyMap& properties);

    /**
     * @brief Updates a copy of the config list based on the provided properties and compiler, and returns it.
     * @details
     * - Checks if the compiler has changed; if so, re-filters configs.
     * - Filters compiler options based on the current compiler.
     * - Updates the config with the provided arguments and returns it.
     */
    FilteredConfig getConfigForSpecificCompiler(const ov::AnyMap& properties, const ICompilerAdapter* compiler);

    std::string determinePlatform(const ov::AnyMap& properties) const;
    std::string determineDeviceId(const ov::AnyMap& properties) const;
    ov::intel_npu::CompilerType determineCompilerType(const ov::AnyMap& properties) const;
    std::vector<uint8_t> getCompiledModelCompatibilityDescriptor(const std::shared_ptr<IGraph>& graph) const;
    bool checkCompiledModelCompatibilityDescriptor(const std::string& compatibilityString) const;

private:
    struct CopyState {
        PropertiesType pType;
        FilteredConfig config;
        std::shared_ptr<Metrics> metrics;
        ov::SoPtr<IEngineBackend> backend;
        Logger logger;
        ov::intel_npu::CompilerType currentlyUsedCompiler;
        std::string currentlyUsedPlatform;
        bool compilerConfigsFilteredByCompiler;
        std::map<std::string, std::tuple<bool, ov::PropertyMutability, std::function<ov::Any(const Config&)>>>
            properties;
        std::vector<ov::PropertyName> supportedProperties;
    };

    explicit Properties(CopyState&& state);

    PropertiesType _pType;
    FilteredConfig _config;
    std::shared_ptr<Metrics> _metrics;
    ov::SoPtr<IEngineBackend> _backend;
    Logger _logger;

    ov::intel_npu::CompilerType _currentlyUsedCompiler = ov::intel_npu::CompilerType::PREFER_PLUGIN;
    std::string _currentlyUsedPlatform;

    bool _compilerConfigsFilteredByCompiler =
        false;  ///< Boolean to check whether properties was filtered with compiler supported properties

    // properties map: {name -> [supported, mutable, eval function]}
    std::map<std::string, std::tuple<bool, ov::PropertyMutability, std::function<ov::Any(const Config&)>>> _properties;
    std::vector<ov::PropertyName> _supportedProperties;

    /**
     * @brief Checks whether a property was registered by its name
     */
    bool isPropertyRegistered(const std::string& propertyName) const;

    // internal registration functions based on client object
    /**
     * @brief Initialize the properties map and try registering the properties for npu-plugin and compiled-model
     * Can be used for both plugin and compiled-model properties maps, based on the provided pType param to the
     * constructor of this object
     * @details
     * - it will reset the properties map
     * - it will try registering config-backed option-based properties, with data from global configuration (supported,
     * visibilty, mutability, value)
     * - if an option is not present in the global config, it assumes it is not supported and will skip it
     * - it will register metric-based properties, with data from the metrics interface
     * - at the end it populates supported_properties with the now dynamically registered public properties
     */
    void registerProperties();
    void registerPluginProperties();
    void registerCompiledModelProperties();

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
