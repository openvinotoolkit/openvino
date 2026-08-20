// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin_property_manager.hpp"

#include <algorithm>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "compiler_option_support_helper.hpp"
#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/device_helpers.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/utils.hpp"
#include "metadata.hpp"

namespace {

const std::vector<ov::PropertyName> cachingProperties = [] {
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
    intel_npu::for_each_cached_npuw_option([&](auto tag) {
        using Opt = typename decltype(tag)::type;
        properties.emplace_back(std::string{Opt::key()});
    });
    return properties;
}();

const std::vector<ov::PropertyName> internalSupportedProperties = {ov::internal::caching_properties.name(),
                                                                   ov::internal::caching_with_mmap.name(),
                                                                   ov::internal::cache_header_alignment.name()};

constexpr uint32_t maxNumOfOptimalInferRequests = 8u;

const std::vector<std::string> optimizationCapabilities = {
    ov::device::capability::FP16,
    ov::device::capability::INT8,
    ov::device::capability::EXPORT_IMPORT,
};

// Provides a hint for a range for number of async infer requests (bottom bound, upper bound, step)
const std::tuple<uint32_t, uint32_t, uint32_t> rangeForAsyncInferRequests{1u, maxNumOfOptimalInferRequests, 1u};

// Provides information about a range for streams (bottom bound, upper bound)
const std::tuple<uint32_t, uint32_t> rangeForStreams{0u, maxNumOfOptimalInferRequests};

std::map<std::string, std::string> any_copy(const ov::AnyMap& params) {
    std::map<std::string, std::string> result;
    for (auto&& value : params) {
        result.emplace(value.first, value.second.as<std::string>());
    }
    return result;
}

// These properties are special because support is not guaranteed by either the compiler or the plugin.
bool isSpecialBothProperty(const std::string& key) {
    return key == ov::intel_npu::turbo.name();
}

void logCpuPinningDeprecationWarning(intel_npu::Logger& logger) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    logger.warning(intel_npu::ENABLE_CPU_PINNING::deprecationMessage());
    OPENVINO_SUPPRESS_DEPRECATED_END
}

void exclude_model_ptr_from_map(ov::AnyMap& properties) {
    if (properties.count(ov::hint::model.name())) {
        properties.erase(ov::hint::model.name());
    }
}

struct ResolvedRequestContext {
    ov::intel_npu::CompilerType compilerType;
    std::string deviceId;
    std::string platform;
};

ResolvedRequestContext resolveRequestContext(const ov::AnyMap& arguments,
                                             ov::intel_npu::CompilerType defaultCompilerType,
                                             std::string defaultDeviceId,
                                             std::string defaultPlatform) {
    const auto compilerTypeIt = arguments.find(ov::intel_npu::compiler_type.name());
    const auto compilerType = compilerTypeIt == arguments.end()
                                  ? defaultCompilerType
                                  : ::intel_npu::COMPILER_TYPE::parse(compilerTypeIt->second.as<std::string>());

    const auto deviceIdIt = arguments.find(std::string(ov::device::id.name()));
    auto deviceId = deviceIdIt == arguments.end() ? std::move(defaultDeviceId) : deviceIdIt->second.as<std::string>();

    const auto platformIt = arguments.find(ov::intel_npu::platform.name());
    auto platform = platformIt == arguments.end() ? std::move(defaultPlatform) : platformIt->second.as<std::string>();

    return {compilerType, std::move(deviceId), std::move(platform)};
}

bool isCompatibilityCheckSupported(const ov::SoPtr<intel_npu::IEngineBackend>& backend,
                                   intel_npu::CompilerOptionSupportHelper& optionSupportHelper) {
    using namespace intel_npu;

    if (!backend || !backend->getDevice()) {
        return false;
    }

    const auto initStructs = backend->getInitStructs();
    if (initStructs != nullptr && initStructs->getZeDrvApiVersion() >= ZE_MAKE_VERSION(1, 16)) {
        return true;
    }

    // Fallback to plugin compiler support check routed through the option support helper.
    try {
        return optionSupportHelper.isOptionSupported(ov::intel_npu::CompilerType::PLUGIN,
                                                     ov::compatibility_check.name());
    } catch (...) {
        return false;
    }
}

ov::CompatibilityCheck validateCompatibilityDescriptor(const ov::SoPtr<intel_npu::IEngineBackend>& backend,
                                                       const ov::AnyMap& arguments,
                                                       intel_npu::CompilerOptionSupportHelper& optionSupportHelper) {
    using namespace intel_npu;

    if (arguments.empty() || arguments.find(ov::runtime_requirements.name()) == arguments.end()) {
        return ov::CompatibilityCheck::NOT_APPLICABLE;
    }

    const auto& runtimeRequirements = arguments.at(ov::runtime_requirements.name()).as<const std::string&>();

    if (runtimeRequirements.empty()) {
        return ov::CompatibilityCheck::NOT_APPLICABLE;
    }

    std::unique_ptr<MetadataBase> metadata = nullptr;
    try {
        metadata = read_as_text(runtimeRequirements);
    } catch (...) {
        return ov::CompatibilityCheck::UNSUPPORTED;
    }

    const auto descriptorView = metadata->get_compatibility_descriptor();
    std::string compatibilityDescriptor = descriptorView.has_value() ? std::string(descriptorView.value()) : "";

    if (compatibilityDescriptor.empty()) {
        return ov::CompatibilityCheck::NOT_APPLICABLE;
    }

    OPENVINO_ASSERT(backend && backend->getDevice(), "Device is not available for compatibility descriptor validation");

    const auto device = backend->getDevice();
    const auto initStructs = backend->getInitStructs();

    if (device != nullptr && initStructs != nullptr && initStructs->getZeDrvApiVersion() >= ZE_MAKE_VERSION(1, 16)) {
        auto result = device->validateCompatibilityDescriptor(compatibilityDescriptor);
        return result ? ov::CompatibilityCheck::SUPPORTED : ov::CompatibilityCheck::UNSUPPORTED;
    }

    // Fallback routed through the option support helper.
    try {
        const bool supported = optionSupportHelper.isOptionSupported(ov::intel_npu::CompilerType::PLUGIN,
                                                                     ov::compatibility_check.name(),
                                                                     std::make_optional(compatibilityDescriptor));
        return supported ? ov::CompatibilityCheck::SUPPORTED : ov::CompatibilityCheck::UNSUPPORTED;
    } catch (...) {
        return ov::CompatibilityCheck::NOT_APPLICABLE;
    }
}

void register_options(const ov::SoPtr<intel_npu::IEngineBackend>& backend, intel_npu::OptionsDesc& options) {
    using namespace intel_npu;
#define REGISTER_OPTION(OPT_TYPE) \
    do {                          \
        options.add<OPT_TYPE>();  \
    } while (0)

    REGISTER_OPTION(LOG_LEVEL);
    REGISTER_OPTION(COMPILE_LOG_LEVEL);
    REGISTER_OPTION(CACHE_DIR);
    REGISTER_OPTION(CACHE_MODE);
    REGISTER_OPTION(COMPILED_BLOB);
    REGISTER_OPTION(DEVICE_ID);
    REGISTER_OPTION(NUM_STREAMS);
    REGISTER_OPTION(PERF_COUNT);
    REGISTER_OPTION(LOADED_FROM_CACHE);
    REGISTER_OPTION(COMPILATION_NUM_THREADS);
    REGISTER_OPTION(PERFORMANCE_HINT);
    REGISTER_OPTION(EXECUTION_MODE_HINT);
    REGISTER_OPTION(PERFORMANCE_HINT_NUM_REQUESTS);
    REGISTER_OPTION(INFERENCE_PRECISION_HINT);
    REGISTER_OPTION(MODEL_PRIORITY);
    REGISTER_OPTION(COMPILATION_MODE_PARAMS);
    REGISTER_OPTION(DMA_ENGINES);
    REGISTER_OPTION(TILES);
    REGISTER_OPTION(COMPILATION_MODE);
    REGISTER_OPTION(COMPILER_TYPE);
    REGISTER_OPTION(COMPILER_VERSION);
    REGISTER_OPTION(PLATFORM);
    REGISTER_OPTION(CREATE_EXECUTOR);
    REGISTER_OPTION(DYNAMIC_SHAPE_TO_STATIC);
    REGISTER_OPTION(PROFILING_TYPE);
    REGISTER_OPTION(BACKEND_COMPILATION_PARAMS);
    REGISTER_OPTION(BATCH_MODE);
    REGISTER_OPTION(BYPASS_UMD_CACHING);
    REGISTER_OPTION(DEFER_WEIGHTS_LOAD);
    REGISTER_OPTION(WEIGHTS_PATH);
    REGISTER_OPTION(RUN_INFERENCES_SEQUENTIALLY);
    REGISTER_OPTION(COMPILER_DYNAMIC_QUANTIZATION);
    REGISTER_OPTION(QDQ_OPTIMIZATION);
    REGISTER_OPTION(QDQ_OPTIMIZATION_AGGRESSIVE);
    REGISTER_OPTION(STEPPING);
    REGISTER_OPTION(DISABLE_VERSION_CHECK);
    REGISTER_OPTION(EXPORT_RAW_BLOB);
    REGISTER_OPTION(IMPORT_RAW_BLOB);
    REGISTER_OPTION(BATCH_COMPILER_MODE_SETTINGS);
    REGISTER_OPTION(TURBO);
    REGISTER_OPTION(ENABLE_WEIGHTLESS);
    REGISTER_OPTION(SEPARATE_WEIGHTS_VERSION);
    REGISTER_OPTION(WS_COMPILE_CALL_NUMBER);
    REGISTER_OPTION(MODEL_SERIALIZER_VERSION);
    REGISTER_OPTION(ENABLE_STRIDES_FOR);
    REGISTER_OPTION(SHARED_COMMON_QUEUE);
    REGISTER_OPTION(CACHE_ENCRYPTION_CALLBACKS);
    REGISTER_OPTION(RUNTIME_REQUIREMENTS);
    REGISTER_OPTION(COMPATIBILITY_CHECK);
    REGISTER_OPTION(MAX_TILES);

    if (backend) {
        if (backend->isCommandQueueExtSupported()) {
            REGISTER_OPTION(WORKLOAD_TYPE);
        }
        if (backend->isContextExtSupported()) {
            REGISTER_OPTION(DISABLE_IDLE_MEMORY_PRUNING);
        }
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    REGISTER_OPTION(ENABLE_CPU_PINNING);
    OPENVINO_SUPPRESS_DEPRECATED_END

    // NPUW properties are requested by OV Core during caching and
    // have no effect on the NPU plugin. But we still need to enable
    // those for OV Core to query. Note: do this last to not filter
    // them out. register npuw caching properties
    for_each_exposed_npuw_option([&](auto tag) {
        using Opt = typename decltype(tag)::type;
        REGISTER_OPTION(Opt);
    });
}

intel_npu::FilteredConfig create_config(const ov::SoPtr<intel_npu::IEngineBackend>& backend) {
    auto options = std::make_shared<intel_npu::OptionsDesc>();
    register_options(backend, *options);

    if (backend) {
        backend->registerOptions(*options);
    }

    return intel_npu::FilteredConfig(options);
}

}  // namespace

namespace intel_npu {

PluginPropertyManager::PluginPropertyManager(const ov::SoPtr<IEngineBackend>& backend,
                                             const std::shared_ptr<CompilerOptionSupportHelper>& optionSupportHelper,
                                             Logger& logger)
    : _config(create_config(backend)),
      _backend(backend),
      _compilerOptionSupportHelper(optionSupportHelper),
      _logger(logger) {
    if (_backend == nullptr) {
        _logger.info("No backend is available. Backend/device-dependent properties will be unavailable.");
    }

    // parse again env_variables to update registered configs which have env vars set
    _config.parseEnvVars();

    if (_config.get<COMPILER_TYPE>() == ov::intel_npu::CompilerType::PREFER_PLUGIN && _backend != nullptr) {
        auto device = _backend->getDevice();
        if (device) {
            auto platformName = device->getName();
            CompilerAdapterFactory compilerFactory;
            auto compileType = compilerFactory.determineAppropriateCompilerTypeBasedOnPlatform(platformName);
            if (compileType == ov::intel_npu::CompilerType::DRIVER) {
                _config.update({{ov::intel_npu::compiler_type.name(), COMPILER_TYPE::toString(compileType)}});
            }
        }
    }

    registerProperties();
}

void PluginPropertyManager::setProperty(const ov::AnyMap& properties) {
    std::lock_guard<std::mutex> lock(_mutex);
    if (properties.find(ov::hint::enable_cpu_pinning.name()) != properties.end()) {
        logCpuPinningDeprecationWarning(_logger);
    }

    auto normalizedArguments = resolveRequestContext(properties,
                                                     _config.get<COMPILER_TYPE>(),
                                                     _config.get<DEVICE_ID>(),
                                                     _config.get<PLATFORM>());
    ov::AnyMap supportCheckArguments = {
        {ov::intel_npu::compiler_type.name(), normalizedArguments.compilerType},
        {ov::device::id.name(), normalizedArguments.deviceId},
        {ov::intel_npu::platform.name(), normalizedArguments.platform},
    };

    bool propertyIsCompilerConfig = false;
    bool propertyIsRegistered = true;
    for (const auto& property : properties) {
        if (!isPropertyRegistered(property.first)) {
            propertyIsRegistered = false;
            break;
        }

        if (_config.hasOpt(property.first)) {
            auto opt = _config.getOpt(property.first);
            if (opt.mode() == OptionMode::CompileTime || isSpecialBothProperty(property.first)) {
                propertyIsCompilerConfig = true;
                break;
            }
        }
    }
    // Create a compiler to get the type in case it is set to PreferPlugin
    if ((propertyIsCompilerConfig || !propertyIsRegistered) &&
        normalizedArguments.compilerType == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
        auto device = utils::getDeviceById(_backend, normalizedArguments.deviceId);

        auto compilationPlatform = utils::getCompilationPlatform(
            normalizedArguments.platform,
            device == nullptr ? normalizedArguments.deviceId : device->getName(),
            _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());

        CompilerAdapterFactory factory;
        (void)factory.getCompiler(_backend, normalizedArguments.compilerType, compilationPlatform);
        supportCheckArguments[ov::intel_npu::compiler_type.name()] = normalizedArguments.compilerType;
    }

    std::map<std::string, std::string> cfgs_to_set;
    ov::AnyMap special_cfgs_to_set;
    for (auto&& value : properties) {
        const auto propertyDescriptorIt = _properties.find(value.first);
        if (propertyDescriptorIt == _properties.end()) {
            // property doesn't exist - checking as internal now
            bool isSupported = false;
            try {
                isSupported =
                    _compilerOptionSupportHelper->isOptionSupported(normalizedArguments.compilerType, value.first);
            } catch (...) {
                // ignore any exceptions from the compiler and treat the property as unsupported
                isSupported = false;
            }
            if (!isSupported) {
                OPENVINO_THROW("Unsupported configuration key: ", value.first);
            }

            // if compiler reports it supported > registering as internal
            _config.addOrUpdateInternal(value.first, value.second.as<std::string>());
        } else {
            const auto& descriptor = propertyDescriptorIt->second;
            if (!descriptor.isSupported(supportCheckArguments)) {
                OPENVINO_THROW("Unsupported configuration key: ", value.first);
            }
            if (descriptor.mutability == ov::PropertyMutability::RO) {
                OPENVINO_THROW("READ-ONLY configuration key: ", value.first);
            }
            if (value.first == ov::cache_encryption_callbacks.name()) {
                special_cfgs_to_set.emplace(value.first, value.second);
            } else {
                cfgs_to_set.emplace(value.first, value.second.as<std::string>());
            }
        }
    }

    if (!cfgs_to_set.empty()) {
        _config.update(cfgs_to_set);
    }
    if (!special_cfgs_to_set.empty()) {
        _config.updateAny(special_cfgs_to_set);
    }
}

ov::Any PluginPropertyManager::getProperty(const std::string& name, const ov::AnyMap& arguments) {
    std::lock_guard<std::mutex> lock(_mutex);
    if (name == ov::hint::enable_cpu_pinning.name()) {
        logCpuPinningDeprecationWarning(_logger);
    }

    auto normalizedArguments = resolveRequestContext(arguments,
                                                     _config.get<COMPILER_TYPE>(),
                                                     _config.get<DEVICE_ID>(),
                                                     _config.get<PLATFORM>());
    auto propertyArguments = arguments;
    propertyArguments[ov::intel_npu::compiler_type.name()] = normalizedArguments.compilerType;
    propertyArguments[std::string(ov::device::id.name())] = normalizedArguments.deviceId;
    propertyArguments[ov::intel_npu::platform.name()] = normalizedArguments.platform;

    bool propertyIsCompilerConfig = false;
    bool propertyIsRegistered = true;
    // If the property is not registered, there is no point of checking the config.
    if (!isPropertyRegistered(name)) {
        propertyIsRegistered = false;
    } else if (_config.hasOpt(name)) {
        // Property is already registered but need to re-check if the CompilerTime config is still supported by the
        // current compiler.
        auto opt = _config.getOpt(name);
        if (opt.mode() == OptionMode::CompileTime || isSpecialBothProperty(name)) {
            propertyIsCompilerConfig = true;
        }
    }

    // Special case for Supported Properties and Caching Properties as they are compiler dependent. So we need to
    // check compiler support for those properties on each getProperty call as well.
    if ((propertyIsCompilerConfig || !propertyIsRegistered || name == ov::supported_properties.name() ||
         name == ov::internal::caching_properties.name()) &&
        normalizedArguments.compilerType == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
        auto device = utils::getDeviceById(_backend, normalizedArguments.deviceId);

        auto compilationPlatform = utils::getCompilationPlatform(
            normalizedArguments.platform,
            device == nullptr ? normalizedArguments.deviceId : device->getName(),
            _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());

        // Create a compiler to get the type and fetch version and supported options if needed
        try {
            CompilerAdapterFactory factory;
            (void)factory.getCompiler(_backend, normalizedArguments.compilerType, compilationPlatform);
            propertyArguments[ov::intel_npu::compiler_type.name()] = normalizedArguments.compilerType;
        } catch (const std::exception& ex) {
            if (_config.hasOpt(name) && _config.getOpt(name).mode() == OptionMode::CompileTime) {
                OPENVINO_THROW("Failed to create compiler for getting property ", name, " with error: ", ex.what());
            }
            _logger.warning("Failed to create compiler for getting property %s with error: %s. Returning only "
                            "properties that do not require compiler support.",
                            name.c_str(),
                            ex.what());
            propertyArguments.erase(ov::intel_npu::compiler_type.name());
        }
    }

    auto&& configIterator = _properties.find(name);
    if (configIterator != _properties.cend()) {
        if (!configIterator->second.isSupported(propertyArguments)) {
            OPENVINO_THROW("Unsupported configuration key: ", name);
        }
        if (configIterator->second.mutability == ov::PropertyMutability::WO) {
            _logger.warning("Trying to get WRITE-ONLY property: %s. Returning empty `ov::Any` object", name.c_str());
            return ov::Any();
        }
        return configIterator->second.get(propertyArguments);
    }
    if (_config.hasInternal(name)) {
        const auto compilerTypeIt = propertyArguments.find(ov::intel_npu::compiler_type.name());
        if (compilerTypeIt != propertyArguments.end()) {
            try {
                const auto compilerType = compilerTypeIt->second.as<ov::intel_npu::CompilerType>();
                if (_compilerOptionSupportHelper->isOptionSupported(compilerType, name)) {
                    return _config.getInternal(name);
                }
            } catch (...) {
            }
        }
    }
    OPENVINO_THROW("Unsupported configuration key: ", name);
}

bool PluginPropertyManager::isPropertySupported(const std::string& name, const ov::AnyMap& arguments) {
    std::lock_guard<std::mutex> lock(_mutex);
    if (name == ov::hint::enable_cpu_pinning.name()) {
        logCpuPinningDeprecationWarning(_logger);
    }
    if (!isPropertyRegistered(name)) {
        return false;
    }

    auto normalizedArguments = resolveRequestContext(arguments,
                                                     _config.get<COMPILER_TYPE>(),
                                                     _config.get<DEVICE_ID>(),
                                                     _config.get<PLATFORM>());
    auto propertyArguments = arguments;
    propertyArguments[ov::intel_npu::compiler_type.name()] = normalizedArguments.compilerType;
    propertyArguments[std::string(ov::device::id.name())] = normalizedArguments.deviceId;
    propertyArguments[ov::intel_npu::platform.name()] = normalizedArguments.platform;
    auto deviceIdForRequest = normalizedArguments.deviceId;
    const auto& platformForRequest = normalizedArguments.platform;

    if (!_config.hasOpt(name)) {
        const auto it = _properties.find(name);
        return it->second.isPublic && it->second.isSupported(propertyArguments);
    }

    auto opt = _config.getOpt(name);
    if (opt.mode() != OptionMode::CompileTime && !isSpecialBothProperty(name)) {
        const auto it = _properties.find(name);
        return it->second.isPublic && it->second.isSupported(propertyArguments);
    }

    if (isSpecialBothProperty(name)) {
        // Fast path: Remove compiler type for special both properties and check if supported.
        propertyArguments.erase(ov::intel_npu::compiler_type.name());
        const auto it = _properties.find(name);
        if (it != _properties.end() && it->second.isPublic && it->second.isSupported(propertyArguments)) {
            return true;
        }
    }

    // Property is compiler config, need to check compiler support
    auto device = utils::getDeviceById(_backend, deviceIdForRequest);

    auto compilationPlatform =
        utils::getCompilationPlatform(platformForRequest,
                                      device == nullptr ? std::move(deviceIdForRequest) : device->getName(),
                                      _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());

    // Create a compiler to get the type and fetch version and supported options if needed
    try {
        CompilerAdapterFactory factory;
        (void)factory.getCompiler(_backend,
                                  normalizedArguments.compilerType,
                                  compilationPlatform,
                                  _compilerOptionSupportHelper->getOptionSupportCache());
        propertyArguments[ov::intel_npu::compiler_type.name()] = normalizedArguments.compilerType;
    } catch (const std::exception& ex) {
        if (_config.hasOpt(name) && _config.getOpt(name).mode() == OptionMode::CompileTime) {
            return false;
        }
        _logger.info("Failed to create compiler to query property %s with error: %s.", name.c_str(), ex.what());
        propertyArguments.erase(ov::intel_npu::compiler_type.name());
    }

    const auto it = _properties.find(name);
    return it != _properties.end() && it->second.isPublic && it->second.isSupported(propertyArguments);
}

FilteredConfig PluginPropertyManager::deriveConfigForProperties(const ov::AnyMap& properties) {
    auto [updatedConfig, logger] = [&]() {
        std::lock_guard<std::mutex> lock(_mutex);
        return std::make_tuple(_config, _logger);
    }();
    if (properties.find(ov::hint::enable_cpu_pinning.name()) != properties.end()) {
        logCpuPinningDeprecationWarning(logger);
    }
    auto pluginProperties = properties;
    exclude_model_ptr_from_map(pluginProperties);

    const std::map<std::string, std::string> rawConfig = any_copy(pluginProperties);
    std::map<std::string, std::string> cfgsToSet;
    ov::AnyMap specialCfgsToSet;
    for (const auto& [key, value] : rawConfig) {
        if (!updatedConfig.hasOpt(key)) {
            // not a known config key
            bool isSupported = false;
            try {
                const auto compilerTypeIt = pluginProperties.find(ov::intel_npu::compiler_type.name());
                if (compilerTypeIt != pluginProperties.end()) {
                    isSupported = _compilerOptionSupportHelper->isOptionSupported(
                        COMPILER_TYPE::parse(compilerTypeIt->second.as<std::string>()),
                        key);
                }
            } catch (...) {
                // ignore any exceptions from the compiler and treat the property as unsupported
                isSupported = false;
            }
            if (!isSupported) {
                OPENVINO_THROW("[ NOT_FOUND ] Option '", key, "' is not supported for current configuration");
            }
            updatedConfig.addOrUpdateInternal(key, value);
        } else if (key == ov::cache_encryption_callbacks.name()) {
            specialCfgsToSet.emplace(key, pluginProperties.at(key));
        } else {
            cfgsToSet.emplace(key, value);
        }
    }

    updatedConfig.update(cfgsToSet);
    updatedConfig.updateAny(specialCfgsToSet);

    return std::move(updatedConfig);
}

std::string PluginPropertyManager::determinePlatform(const ov::AnyMap& properties) const {
    auto platform = properties.find(ov::intel_npu::platform.name());
    if (platform != properties.end()) {
        return platform->second.as<std::string>();
    }
    std::lock_guard<std::mutex> lock(_mutex);
    return _config.get<PLATFORM>();
}

std::string PluginPropertyManager::determineDeviceId(const ov::AnyMap& properties) const {
    auto device_id = properties.find(std::string(ov::device::id.name()));
    if (device_id != properties.end()) {
        return device_id->second.as<std::string>();
    }
    std::lock_guard<std::mutex> lock(_mutex);
    return _config.get<DEVICE_ID>();
}

ov::intel_npu::CompilerType PluginPropertyManager::determineCompilerType(const ov::AnyMap& properties) const {
    auto it = properties.find(ov::intel_npu::compiler_type.name());
    if (it != properties.end()) {
        return COMPILER_TYPE::parse(it->second.as<std::string>());
    }
    std::lock_guard<std::mutex> lock(_mutex);
    return _config.get<COMPILER_TYPE>();
}

bool PluginPropertyManager::isPropertyRegistered(const std::string& propertyName) const {
    return _properties.find(propertyName) != _properties.end();
}

void PluginPropertyManager::registerProperties() {
    _properties.clear();

    const bool hasBackend = _backend != nullptr;
    const auto hasBackendPredicate = [hasBackend](const ov::AnyMap&) {
        return hasBackend;
    };

    using DeviceValidationCache = std::optional<std::pair<std::string, bool>>;
    auto hasBackendAndValidDeviceCache = std::make_shared<DeviceValidationCache>();

    const auto getDeviceId = [this](const ov::AnyMap& arguments) -> std::string {
        const auto deviceIdIt = arguments.find(ov::device::id.name());
        if (deviceIdIt != arguments.end()) {
            return deviceIdIt->second.as<std::string>();
        }
        return std::string();
    };

    const auto hasBackendAndValidDevice =
        [this, hasBackend, hasBackendAndValidDeviceCache, getDeviceId](const ov::AnyMap& arguments) {
            if (!hasBackend) {
                return false;
            }

            try {
                const auto specifiedDeviceName = getDeviceId(arguments);

                if (hasBackendAndValidDeviceCache->has_value() &&
                    hasBackendAndValidDeviceCache->value().first == specifiedDeviceName) {
                    return hasBackendAndValidDeviceCache->value().second;
                }

                const bool isValidDevice = utils::getDeviceById(_backend, specifiedDeviceName) != nullptr;
                *hasBackendAndValidDeviceCache = std::make_pair(specifiedDeviceName, isValidDevice);
                return isValidDevice;
            } catch (...) {
                _logger.debug("Property is not supported for current configuration due to unavailable device.");
            }

            return false;
        };

    const auto isCompilerOptionSupported = [this](std::string_view propertyName,
                                                  ov::intel_npu::CompilerType compilerType) {
        try {
            return _compilerOptionSupportHelper->isOptionSupported(compilerType, std::string(propertyName));
        } catch (...) {
            return false;
        }
    };

    // clang-format off
    register_property<BYPASS_UMD_CACHING>(_config, _properties, true, ov::PropertyMutability::RW);
    register_property<CACHE_DIR>(_config, _properties, true, ov::PropertyMutability::RW);
    register_property<COMPILER_TYPE>(_config, _properties, true, ov::PropertyMutability::RW);
    register_property<DEFER_WEIGHTS_LOAD>(_config, _properties, true, ov::PropertyMutability::RW);
    register_property<DEVICE_ID>(_config, _properties, true, ov::PropertyMutability::RW);
    register_property<DISABLE_IDLE_MEMORY_PRUNING>(_config, _properties, true, ov::PropertyMutability::RW);
    register_property<LOG_LEVEL>(_config, _properties, true, ov::PropertyMutability::RW);
    register_property<MODEL_PRIORITY>(_config, _properties, true, ov::PropertyMutability::RW);
    register_property<NUM_STREAMS>(_config, _properties, true, ov::PropertyMutability::RW);
    register_property<PERFORMANCE_HINT>(_config, _properties, true, ov::PropertyMutability::RW);
    register_property<PERFORMANCE_HINT_NUM_REQUESTS>(_config, _properties, true, ov::PropertyMutability::RW);
    register_property<WEIGHTS_PATH>(_config, _properties, true, ov::PropertyMutability::RW);
    register_property<WORKLOAD_TYPE>(_config, _properties, true, ov::PropertyMutability::RW); //TODO

    register_property<COMPILED_BLOB>(_config, _properties, false, ov::PropertyMutability::RW);
    register_property<CREATE_EXECUTOR>(_config, _properties, false, ov::PropertyMutability::RW);
    register_property<DISABLE_VERSION_CHECK>(_config, _properties, false, ov::PropertyMutability::RW);
    register_property<EXPORT_RAW_BLOB>(_config, _properties, false, ov::PropertyMutability::RW);
    register_property<IMPORT_RAW_BLOB>(_config, _properties, false, ov::PropertyMutability::RW);
    register_property<PERF_COUNT>(_config, _properties, false, ov::PropertyMutability::RW);
    register_property<PROFILING_TYPE>(_config, _properties, false, ov::PropertyMutability::RW);
    register_property<SHARED_COMMON_QUEUE>(_config, _properties, false, ov::PropertyMutability::RW);

    OPENVINO_SUPPRESS_DEPRECATED_START
    register_property<ENABLE_CPU_PINNING>(_config, _properties, false, ov::PropertyMutability::RW);
    OPENVINO_SUPPRESS_DEPRECATED_END

     auto getCompilerTypeOrDefault = [](const ov::AnyMap& arguments) -> std::optional<ov::intel_npu::CompilerType> {
        auto compilerTypeIt = arguments.find(ov::intel_npu::compiler_type.name());
        if (compilerTypeIt == arguments.end()) {
            return std::nullopt;
        }

        try {
            return compilerTypeIt->second.as<ov::intel_npu::CompilerType>();
        } catch (...) {
            return std::nullopt;
        }
    };

    register_property_with_support<CACHE_MODE>(_config, _properties, true, ov::PropertyMutability::RW, [isCompilerOptionSupported, getCompilerTypeOrDefault](const ov::AnyMap& arguments) {
        const auto compilerType = getCompilerTypeOrDefault(arguments);
        return compilerType.has_value() && isCompilerOptionSupported(ov::cache_mode.name(), compilerType.value());
    });
    register_property_with_support<COMPILATION_MODE_PARAMS>(_config, _properties, true, ov::PropertyMutability::RW, [isCompilerOptionSupported, getCompilerTypeOrDefault](const ov::AnyMap& arguments) {
        const auto compilerType = getCompilerTypeOrDefault(arguments);
        return compilerType.has_value() && isCompilerOptionSupported(ov::intel_npu::compilation_mode_params.name(), compilerType.value());
    });
    register_property_with_support<COMPILATION_NUM_THREADS>(_config, _properties, true, ov::PropertyMutability::RW, [isCompilerOptionSupported, getCompilerTypeOrDefault](const ov::AnyMap& arguments) {
        const auto compilerType = getCompilerTypeOrDefault(arguments);
        return compilerType.has_value() && isCompilerOptionSupported(ov::compilation_num_threads.name(), compilerType.value());
    });
    register_property_with_support<COMPILER_DYNAMIC_QUANTIZATION>(_config, _properties, true, ov::PropertyMutability::RW, [isCompilerOptionSupported, getCompilerTypeOrDefault](const ov::AnyMap& arguments) {
        const auto compilerType = getCompilerTypeOrDefault(arguments);
        return compilerType.has_value() && isCompilerOptionSupported(ov::intel_npu::compiler_dynamic_quantization.name(), compilerType.value());
    });
    register_property_with_support<EXECUTION_MODE_HINT>(_config, _properties, true, ov::PropertyMutability::RW, [isCompilerOptionSupported, getCompilerTypeOrDefault](const ov::AnyMap& arguments) {
        const auto compilerType = getCompilerTypeOrDefault(arguments);
        return compilerType.has_value() && isCompilerOptionSupported(ov::hint::execution_mode.name(), compilerType.value());
    });
    register_property_with_support<INFERENCE_PRECISION_HINT>(_config, _properties, true, ov::PropertyMutability::RW, [isCompilerOptionSupported, getCompilerTypeOrDefault](const ov::AnyMap& arguments) {
        const auto compilerType = getCompilerTypeOrDefault(arguments);
        return compilerType.has_value() && isCompilerOptionSupported(ov::hint::inference_precision.name(), compilerType.value());
    });
    register_property_with_support<PLATFORM>(_config, _properties, true, ov::PropertyMutability::RW, [isCompilerOptionSupported, getCompilerTypeOrDefault](const ov::AnyMap& arguments) {
        const auto compilerType = getCompilerTypeOrDefault(arguments);
        return compilerType.has_value() && isCompilerOptionSupported(ov::intel_npu::platform.name(), compilerType.value());
    });
    register_property_with_support<QDQ_OPTIMIZATION>(_config, _properties, true, ov::PropertyMutability::RW, [isCompilerOptionSupported, getCompilerTypeOrDefault](const ov::AnyMap& arguments) {
        const auto compilerType = getCompilerTypeOrDefault(arguments);
        return compilerType.has_value() && isCompilerOptionSupported(ov::intel_npu::qdq_optimization.name(), compilerType.value());
    });
    register_property_with_support<QDQ_OPTIMIZATION_AGGRESSIVE>(_config, _properties, true, ov::PropertyMutability::RW, [isCompilerOptionSupported, getCompilerTypeOrDefault](const ov::AnyMap& arguments) {
        const auto compilerType = getCompilerTypeOrDefault(arguments);
        return compilerType.has_value() && isCompilerOptionSupported(ov::intel_npu::qdq_optimization_aggressive.name(), compilerType.value());
    });
    register_property_with_support<TILES>(_config, _properties, true, ov::PropertyMutability::RW, [isCompilerOptionSupported, getCompilerTypeOrDefault](const ov::AnyMap& arguments) {
        const auto compilerType = getCompilerTypeOrDefault(arguments);
        return compilerType.has_value() && isCompilerOptionSupported(ov::intel_npu::tiles.name(), compilerType.value());
    });

    register_property_with_support<BACKEND_COMPILATION_PARAMS>(_config, _properties, false, ov::PropertyMutability::RW, [isCompilerOptionSupported, getCompilerTypeOrDefault](const ov::AnyMap& arguments) {
        const auto compilerType = getCompilerTypeOrDefault(arguments);
        return compilerType.has_value() && isCompilerOptionSupported(ov::intel_npu::backend_compilation_params.name(), compilerType.value());
    });
    register_property_with_support<BATCH_COMPILER_MODE_SETTINGS>(_config, _properties, false, ov::PropertyMutability::RW, [isCompilerOptionSupported, getCompilerTypeOrDefault](const ov::AnyMap& arguments) {
        const auto compilerType = getCompilerTypeOrDefault(arguments);
        return compilerType.has_value() && isCompilerOptionSupported(ov::intel_npu::batch_compiler_mode_settings.name(), compilerType.value());
    });
    register_property_with_support<BATCH_MODE>(_config, _properties, false, ov::PropertyMutability::RW, [isCompilerOptionSupported, getCompilerTypeOrDefault](const ov::AnyMap& arguments) {
        const auto compilerType = getCompilerTypeOrDefault(arguments);
        return compilerType.has_value() && isCompilerOptionSupported(ov::intel_npu::batch_mode.name(), compilerType.value());
    });
    register_property_with_support<COMPILATION_MODE>(_config, _properties, false, ov::PropertyMutability::RW, [isCompilerOptionSupported, getCompilerTypeOrDefault](const ov::AnyMap& arguments) {
        const auto compilerType = getCompilerTypeOrDefault(arguments);
        return compilerType.has_value() && isCompilerOptionSupported(ov::intel_npu::compilation_mode.name(), compilerType.value());
    });
    register_property_with_support<DMA_ENGINES>(_config, _properties, false, ov::PropertyMutability::RW, [isCompilerOptionSupported, getCompilerTypeOrDefault](const ov::AnyMap& arguments) {
        const auto compilerType = getCompilerTypeOrDefault(arguments);
        return compilerType.has_value() && isCompilerOptionSupported(ov::intel_npu::dma_engines.name(), compilerType.value());
    });
    register_property_with_support<DYNAMIC_SHAPE_TO_STATIC>(_config, _properties, false, ov::PropertyMutability::RW, [isCompilerOptionSupported, getCompilerTypeOrDefault](const ov::AnyMap& arguments) {
        const auto compilerType = getCompilerTypeOrDefault(arguments);
        return compilerType.has_value() && isCompilerOptionSupported(ov::intel_npu::dynamic_shape_to_static.name(), compilerType.value());
    });
    register_property_with_support<ENABLE_WEIGHTLESS>(_config, _properties, false, ov::PropertyMutability::RW, [isCompilerOptionSupported, getCompilerTypeOrDefault](const ov::AnyMap& arguments) {
        const auto compilerType = getCompilerTypeOrDefault(arguments);
        return compilerType.has_value() && isCompilerOptionSupported(ov::enable_weightless.name(), compilerType.value());
    });
    register_property_with_support<MODEL_SERIALIZER_VERSION>(_config, _properties, false, ov::PropertyMutability::RW, [isCompilerOptionSupported, getCompilerTypeOrDefault](const ov::AnyMap& arguments) {
        const auto compilerType = getCompilerTypeOrDefault(arguments);
        return compilerType.has_value() && isCompilerOptionSupported(ov::intel_npu::model_serializer_version.name(), compilerType.value());
    });
    register_property_with_support<SEPARATE_WEIGHTS_VERSION>(_config, _properties, false, ov::PropertyMutability::RW, [isCompilerOptionSupported, getCompilerTypeOrDefault](const ov::AnyMap& arguments) {
        const auto compilerType = getCompilerTypeOrDefault(arguments);
        return compilerType.has_value() && isCompilerOptionSupported(ov::intel_npu::separate_weights_version.name(), compilerType.value());
    });

    register_property_with_custom_function<COMPILE_LOG_LEVEL>(_config, _properties, false, ov::PropertyMutability::RW, [this](const ov::AnyMap&) -> ov::Any {
        return COMPILE_LOG_LEVEL::resolve(_config);
    });
    register_property_with_custom_function<CACHE_ENCRYPTION_CALLBACKS>(_config, _properties, true, ov::PropertyMutability::WO, [](const ov::AnyMap&) {
        return ov::EncryptionCallbacks{nullptr, nullptr};
    });

    register_property_with_custom_function(_properties, ov::execution_devices.name(), true, ov::PropertyMutability::RO, [](const ov::AnyMap&) {
        return std::vector<std::string>{"NPU"};
    });
    register_property_with_custom_function(_properties, ov::device::capabilities.name(), true, ov::PropertyMutability::RO, [](const ov::AnyMap&) {
        return optimizationCapabilities;
    });
    register_property_with_custom_function(_properties, ov::range_for_async_infer_requests.name(), true, ov::PropertyMutability::RO, [](const ov::AnyMap&) {
        return rangeForAsyncInferRequests;
    });
    register_property_with_custom_function(_properties, ov::range_for_streams.name(), true, ov::PropertyMutability::RO, [](const ov::AnyMap&) {
        return rangeForStreams;
    });
    register_property_with_custom_function(_properties, ov::available_devices.name(), true, ov::PropertyMutability::RO, [this](const ov::AnyMap&) {
        return _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames();
    });
    register_property_with_custom_function(_properties, ov::hint::model.name(), true, ov::PropertyMutability::RO, [](const ov::AnyMap&) {
        return std::shared_ptr<const ov::Model>(nullptr);
    });
    register_property_with_custom_function(_properties, ov::supported_properties.name(), true, ov::PropertyMutability::RO, [this](const ov::AnyMap&) {
        std::vector<ov::PropertyName> supportedProperties;
        for (auto& property : _properties) {
            if (property.second.isPublic && property.second.isSupported(ov::AnyMap{})) {
                supportedProperties.emplace_back(ov::PropertyName(property.first, property.second.mutability));
            }
        }
        return supportedProperties;
    });

    register_property_with_custom_function(_properties, ov::internal::supported_properties.name(), false, ov::PropertyMutability::RO, [](const ov::AnyMap&) {
        return internalSupportedProperties;
    });
    register_property_with_custom_function(_properties, ov::internal::cache_header_alignment.name(), false, ov::PropertyMutability::RO, [](const ov::AnyMap&) {
        return utils::STANDARD_PAGE_SIZE;
    });
    register_property_with_custom_function(_properties, ov::internal::caching_properties.name(), false, ov::PropertyMutability::RO, [this](const ov::AnyMap&) {
        std::vector<ov::PropertyName> caching_props{};
        for (auto prop : cachingProperties) {
            const auto propertyIt = _properties.find(prop);
            if (propertyIt != _properties.end() && propertyIt->second.isSupported(ov::AnyMap{})) {
                caching_props.emplace_back(prop);
            }
        }
        return caching_props;
    });


    // Special case: this property is always registered because it's supported by the implementation,
    // but it's not visible in supported_properties if the driver doesn't support it.
    register_property<RUN_INFERENCES_SEQUENTIALLY>(_config, _properties, [this] {
        if (_backend && _backend->getInitStructs()) {
            if (_backend->getInitStructs()->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 1)) {
                return true;
            }
        }
        return false;
    }(), ov::PropertyMutability::RW);

    register_property_with_support_and_custom_function(_properties, ov::intel_npu::driver_version.name(), true, ov::PropertyMutability::RO, hasBackendPredicate, [this](const ov::AnyMap&) {
        if (_backend == nullptr) {
            OPENVINO_THROW("No available backend");
        }
        return _backend->getDriverVersion();
    });
    register_property_with_support_and_custom_function(_properties, ov::device::pci_info.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        return utils::getPciInfo(_backend, getDeviceId(arguments));
    });
    register_property_with_support_and_custom_function(_properties, ov::device::gops.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        return utils::getGops(_backend, getDeviceId(arguments));
    });
    register_property_with_support_and_custom_function(_properties, ov::device::type.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        return utils::getDeviceType(_backend, getDeviceId(arguments));
    });
    register_property_with_support_and_custom_function(_properties, ov::intel_npu::device_alloc_mem_size.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        return utils::getDeviceAllocMemSize(_backend, getDeviceId(arguments));
    });
    register_property_with_support_and_custom_function(_properties, ov::intel_npu::device_total_mem_size.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        return utils::getDeviceTotalMemSize(_backend, getDeviceId(arguments));
    });
    register_property_with_support_and_custom_function(_properties, ov::device::uuid.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        auto devUuid = utils::getDeviceUuid(_backend, getDeviceId(arguments));
        return decltype(ov::device::uuid)::value_type{devUuid};
    });
    register_property_with_support_and_custom_function(_properties, ov::device::architecture.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        const auto devName = utils::getDeviceName(_backend, getDeviceId(arguments));
        return utils::getPlatformByDeviceName(devName);
    });
    register_property_with_support_and_custom_function(_properties, ov::device::full_name.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        return utils::getFullDeviceName(_backend, getDeviceId(arguments));
    });
    register_property_with_support_and_custom_function(_properties, ov::device::luid.name(), _backend != nullptr && _backend->isLUIDExtSupported(), ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        return utils::getDeviceLUID(_backend, getDeviceId(arguments));
    });
    register_property_with_support_and_custom_function(_properties, ov::intel_npu::stepping.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        if (!_config.has<STEPPING>()) {
            try {
                const auto specifiedDeviceName = getDeviceId(arguments);
                return static_cast<int64_t>(utils::getSteppingNumber(_backend, specifiedDeviceName));
            } catch (...) {
                _logger.warning("GetSteppingNumber failed to get value from device.");
            }
        }
        return _config.get<STEPPING>();
    });
    register_property_with_support_and_custom_function(_properties, ov::intel_npu::max_tiles.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        if (!_config.has<MAX_TILES>()) {
            try {
                const auto specifiedDeviceName = getDeviceId(arguments);
                return static_cast<int64_t>(utils::getMaxTiles(_backend, specifiedDeviceName));
            } catch (...) {
                _logger.warning("GetMaxTiles failed to get value from device.");
            }
        }
        return _config.get<MAX_TILES>();
    });

    register_property_with_support_and_custom_function(_properties, ov::intel_npu::backend_name.name(), false, ov::PropertyMutability::RO, hasBackendPredicate, [this](const ov::AnyMap&) {
        if (_backend == nullptr) {
            OPENVINO_THROW("No available backend");
        }
        return _backend->getName();
    });

    register_property_with_support_and_custom_function<ENABLE_STRIDES_FOR>(_config, _properties, true, ov::PropertyMutability::RW,
        [this, isCompilerOptionSupported, getCompilerTypeOrDefault](const ov::AnyMap& arguments) {  // support predicate
            const auto compilerType = getCompilerTypeOrDefault(arguments);
            if (!compilerType.has_value() || !isCompilerOptionSupported(ov::intel_npu::enable_strides_for.name(), compilerType.value())) {
                return false;
            }
            // Return true if the backend is not available, in case of offline compilation.
            if (_backend == nullptr) {
                return true;
            }
            // If a backend is present, check if the driver supports this property. If not, return false.
            if (_backend->getGraphExtVersion() < ZE_MAKE_VERSION(1, 16)) {
                _logger.info("Config option %s not supported by the driver! Requirements not met.", ov::intel_npu::enable_strides_for.name());
                return false;
            }
            return true;
        },
        [this](const ov::AnyMap&) { // value getter
            return _config.get<ENABLE_STRIDES_FOR>();
        });
    register_property_with_support_and_custom_function<TURBO>(_config, _properties, true, ov::PropertyMutability::RW,
        [this, isCompilerOptionSupported, getCompilerTypeOrDefault](const ov::AnyMap& arguments) {  // support predicate
            const auto compilerType = getCompilerTypeOrDefault(arguments);
            if (compilerType.has_value() && isCompilerOptionSupported(ov::intel_npu::turbo.name(), compilerType.value())) {
                return true;
            }
            return _backend != nullptr && _backend->isCommandQueueExtSupported();
        },
        [this](const ov::AnyMap&) { // value getter
            return _config.get<TURBO>();
        });
    // clang-format on

    register_property_with_support_and_custom_function(
        _properties,
        ov::intel_npu::compiler_version.name(),
        true,
        ov::PropertyMutability::RO,
        [this,
         getCompilerTypeOrDefault,
         getDeviceId,
         compilerVersionSupportCache = std::optional<std::tuple<ov::intel_npu::CompilerType, bool>>{}](
            const ov::AnyMap& arguments) mutable {  // support predicate
            auto compilerType = getCompilerTypeOrDefault(arguments);
            if (!compilerType.has_value()) {
                return false;
            }
            std::string platform;
            if (compilerType.value() == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
                const auto platformIt = arguments.find(ov::intel_npu::platform.name());
                if (platformIt == arguments.end()) {
                    OPENVINO_THROW("Compilation platform is not specified in properties.");
                }
                platform = platformIt->second.as<std::string>();
            }

            try {
                if (compilerVersionSupportCache.has_value() &&
                    std::get<0>(compilerVersionSupportCache.value()) == compilerType.value()) {
                    return std::get<1>(compilerVersionSupportCache.value());
                }

                CompilerAdapterFactory factory;
                const bool isSupported = factory.getCompiler(_backend, compilerType.value(), platform) != nullptr;
                compilerVersionSupportCache = std::make_tuple(compilerType.value(), isSupported);
                return isSupported;
            } catch (...) {
                return false;
            }
        },
        [this, getCompilerTypeOrDefault](const ov::AnyMap& arguments) {  // value getter
            auto compilerType = getCompilerTypeOrDefault(arguments);
            if (!compilerType.has_value()) {
                OPENVINO_THROW("Compiler type is not specified in properties.");
            }
            std::string platform;
            if (compilerType.value() == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
                const auto platformIt = arguments.find(ov::intel_npu::platform.name());
                if (platformIt == arguments.end()) {
                    OPENVINO_THROW("Compilation platform is not specified in properties.");
                }
                platform = platformIt->second.as<std::string>();
            }

            CompilerAdapterFactory factory;
            auto dummyCompiler = factory.getCompiler(_backend, compilerType.value(), platform);
            return dummyCompiler->get_version();
        });

    register_property_with_support_custom_function_and_args(
        _properties,
        ov::compatibility_check.name(),
        true,
        ov::PropertyMutability::RO,
        [this, compatibilityCheckSupported = std::optional<bool>{}](const ov::AnyMap&) mutable {  // support predicate
            if (!compatibilityCheckSupported.has_value()) {
                compatibilityCheckSupported = isCompatibilityCheckSupported(_backend, *_compilerOptionSupportHelper);
            }
            return compatibilityCheckSupported.value();
        },
        [this](const ov::AnyMap& arguments) {  // value getter
            return validateCompatibilityDescriptor(_backend, arguments, *_compilerOptionSupportHelper);
        });

    for_each_exposed_npuw_option([this](auto tag) {
        using Opt = typename decltype(tag)::type;
        register_npuw_property<Opt>(_config, _properties);
    });
}

}  // namespace intel_npu
