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

}  // namespace

namespace intel_npu {

PluginPropertyManager::PluginPropertyManager(const FilteredConfig& config,
                                             const ov::SoPtr<IEngineBackend>& backend,
                                             const std::shared_ptr<CompilerOptionSupportHelper>& optionSupportHelper,
                                             Logger& logger)
    : _config(config),
      _backend(backend),
      _compilerOptionSupportHelper(optionSupportHelper),
      _logger(logger) {
    if (_backend == nullptr) {
        _logger.info("No backend is available. Backend/device-dependent properties will be unavailable.");
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

    std::map<std::string, std::string> cfgs_to_set;
    ov::AnyMap special_cfgs_to_set;
    for (auto&& value : properties) {
        const auto propertyDescriptorIt = _properties.find(value.first);
        if (propertyDescriptorIt == _properties.end()) {
            // property doesn't exist - checking as internal now
            ov::intel_npu::CompilerType compilerType = normalizedArguments.compilerType;
            if (compilerType == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
                const auto resolvedCompilerType =
                    resolveCompilerType(compilerType, normalizedArguments.deviceId, normalizedArguments.platform);
                if (!resolvedCompilerType.has_value()) {
                    OPENVINO_THROW("Unsupported configuration key: ", value.first);
                }
                compilerType = resolvedCompilerType.value();
            }

            bool isSupported = false;
            try {
                isSupported = _compilerOptionSupportHelper->isOptionSupported(compilerType, value.first);
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
            if (descriptor.mutability == ov::PropertyMutability::RO) {
                OPENVINO_THROW("READ-ONLY configuration key: ", value.first);
            }
            if (!descriptor.isSupported(supportCheckArguments)) {
                OPENVINO_THROW("Unsupported configuration key: ", value.first);
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
        ov::intel_npu::CompilerType compilerType = normalizedArguments.compilerType;
        if (compilerType == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
            const auto resolvedCompilerType =
                resolveCompilerType(compilerType, normalizedArguments.deviceId, normalizedArguments.platform);
            if (!resolvedCompilerType.has_value()) {
                OPENVINO_THROW("Unsupported configuration key: ", name);
            }
            compilerType = resolvedCompilerType.value();
        }

        try {
            if (_compilerOptionSupportHelper->isOptionSupported(compilerType, name)) {
                return _config.getInternal(name);
            }
        } catch (...) {
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

    if (isSpecialBothProperty(name)) {
        // Fast path: Remove compiler type for special both properties and check if supported.
        propertyArguments.erase(ov::intel_npu::compiler_type.name());
        const auto it = _properties.find(name);
        if (it != _properties.end() && it->second.isPublic && it->second.isSupported(propertyArguments)) {
            return true;
        }
    }

    const auto it = _properties.find(name);
    return it != _properties.end() && it->second.isPublic && it->second.isSupported(propertyArguments);
}

FilteredConfig PluginPropertyManager::deriveConfigForPropertiesForCompiler(const ov::AnyMap& properties) {
    auto [updatedConfig, logger] = [&]() {
        std::lock_guard<std::mutex> lock(_mutex);
        return std::make_tuple(_config, _logger);
    }();
    if (properties.find(ov::hint::enable_cpu_pinning.name()) != properties.end()) {
        logCpuPinningDeprecationWarning(logger);
    }
    auto localProperties = properties;
    exclude_model_ptr_from_map(localProperties);

    if (localProperties.find(ov::intel_npu::compiler_type.name()) == localProperties.end()) {
        updatedConfig.remove(ov::intel_npu::compiler_type.name());
    }

    const std::map<std::string, std::string> rawConfig = any_copy(localProperties);
    std::map<std::string, std::string> cfgsToSet;
    ov::AnyMap specialCfgsToSet;
    for (const auto& [key, value] : rawConfig) {
        if (!updatedConfig.hasOpt(key)) {
            // not a known config key
            bool isSupported = false;
            try {
                const auto compilerTypeIt = localProperties.find(ov::intel_npu::compiler_type.name());
                if (compilerTypeIt != localProperties.end()) {
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
        } else {
            const auto descriptorIt = _properties.find(key);
            if (descriptorIt != _properties.end() && !descriptorIt->second.isSupported(localProperties)) {
                const bool isCompileTimeProperty = _config.getOpt(key).mode() == OptionMode::CompileTime;
                const bool hasCompilerArgument =
                    localProperties.find(ov::intel_npu::compiler_type.name()) != localProperties.end();
                if (isCompileTimeProperty && !hasCompilerArgument) {
                    _logger.warning(
                        "Property '%s' is recognized as a compiler option, will not be used for current configuration.",
                        key.c_str());
                    continue;
                } else {
                    OPENVINO_THROW("[ NOT_FOUND ] Option '", key, "' is not supported for current configuration");
                }
            }

            if (key == ov::cache_encryption_callbacks.name()) {
                specialCfgsToSet.emplace(key, localProperties.at(key));
            } else {
                cfgsToSet.emplace(key, value);
            }
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

std::optional<ov::intel_npu::CompilerType> PluginPropertyManager::resolveCompilerType(
    ov::intel_npu::CompilerType compilerType,
    const std::string& deviceId,
    const std::string& platform) const {
    try {
        auto device = utils::getDeviceById(_backend, deviceId);
        auto compilationPlatform = utils::getCompilationPlatform(
            platform,
            device == nullptr ? deviceId : device->getName(),
            _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());

        CompilerAdapterFactory factory;
        (void)factory.getCompiler(_backend, compilerType, compilationPlatform);
        return compilerType;
    } catch (const std::exception& ex) {
        _logger.warning("Failed to resolve compiler type: %s. Compiler-dependent properties will be unsupported.",
                        ex.what());
        return std::nullopt;
    }
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

    const auto isCompilerOptionSupported = [this,
                                            getCompilerTypeOrDefault,
                                            getDeviceId](const std::string& propertyName, const ov::AnyMap& arguments) {
        const auto compilerType = getCompilerTypeOrDefault(arguments);
        if (!compilerType.has_value()) {
            return false;
        }

        ov::intel_npu::CompilerType resolvedCompilerType;
        if (compilerType.value() == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
            const auto platformIt = arguments.find(ov::intel_npu::platform.name());
            const auto platform = platformIt != arguments.end() ? platformIt->second.as<std::string>() : std::string{};
            const auto resolvedCompilerTypeOpt =
                resolveCompilerType(compilerType.value(), getDeviceId(arguments), platform);
            if (!resolvedCompilerTypeOpt.has_value()) {
                return false;
            }
            resolvedCompilerType = resolvedCompilerTypeOpt.value();
        } else {
            resolvedCompilerType = compilerType.value();
        }

        try {
            return _compilerOptionSupportHelper->isOptionSupported(resolvedCompilerType, propertyName);
        } catch (...) {
            return false;
        }
    };

    // clang-format off
    register_property<BYPASS_UMD_CACHING>(_config, true, ov::PropertyMutability::RW);
    register_property<CACHE_DIR>(_config, true, ov::PropertyMutability::RW);
    register_property<DEFER_WEIGHTS_LOAD>(_config, true, ov::PropertyMutability::RW);
    register_property<DISABLE_IDLE_MEMORY_PRUNING>(_config, true, ov::PropertyMutability::RW);
    register_property<LOG_LEVEL>(_config, true, ov::PropertyMutability::RW);
    register_property<MODEL_PRIORITY>(_config, true, ov::PropertyMutability::RW);
    register_property<NUM_STREAMS>(_config, true, ov::PropertyMutability::RW);
    register_property<PERF_COUNT>(_config, true, ov::PropertyMutability::RW);
    register_property<PERFORMANCE_HINT>(_config, true, ov::PropertyMutability::RW);
    register_property<PERFORMANCE_HINT_NUM_REQUESTS>(_config, true, ov::PropertyMutability::RW);
    register_property<WEIGHTS_PATH>(_config, true, ov::PropertyMutability::RW);
    register_property<WORKLOAD_TYPE>(_config, true, ov::PropertyMutability::RW); //TODO

    register_property<COMPILED_BLOB>(_config, false, ov::PropertyMutability::RW);
    register_property<CREATE_EXECUTOR>(_config, false, ov::PropertyMutability::RW);
    register_property<DISABLE_VERSION_CHECK>(_config, false, ov::PropertyMutability::RW);
    register_property<EXPORT_RAW_BLOB>(_config, false, ov::PropertyMutability::RW);
    register_property<IMPORT_RAW_BLOB>(_config, false, ov::PropertyMutability::RW);
    register_property<PROFILING_TYPE>(_config, false, ov::PropertyMutability::RW);
    register_property<SHARED_COMMON_QUEUE>(_config, false, ov::PropertyMutability::RW);

    OPENVINO_SUPPRESS_DEPRECATED_START
    register_property<ENABLE_CPU_PINNING>(_config, false, ov::PropertyMutability::RW);
    OPENVINO_SUPPRESS_DEPRECATED_END

    register_property_with_support<CACHE_MODE>(_config, true, ov::PropertyMutability::RW, [isCompilerOptionSupported](const ov::AnyMap& arguments) {
        return isCompilerOptionSupported(ov::cache_mode.name(), arguments);
    });
    register_property_with_support<COMPILATION_MODE_PARAMS>(_config, true, ov::PropertyMutability::RW, [isCompilerOptionSupported](const ov::AnyMap& arguments) {
        return isCompilerOptionSupported(ov::intel_npu::compilation_mode_params.name(), arguments);
    });
    register_property_with_support<COMPILATION_NUM_THREADS>(_config, true, ov::PropertyMutability::RW, [isCompilerOptionSupported](const ov::AnyMap& arguments) {
        return isCompilerOptionSupported(ov::compilation_num_threads.name(), arguments);
    });
    register_property_with_support<COMPILER_DYNAMIC_QUANTIZATION>(_config, true, ov::PropertyMutability::RW, [isCompilerOptionSupported](const ov::AnyMap& arguments) {
        return isCompilerOptionSupported(ov::intel_npu::compiler_dynamic_quantization.name(), arguments);
    });
    register_property_with_support<EXECUTION_MODE_HINT>(_config, true, ov::PropertyMutability::RW, [isCompilerOptionSupported](const ov::AnyMap& arguments) {
        return isCompilerOptionSupported(ov::hint::execution_mode.name(), arguments);
    });
    register_property_with_support<INFERENCE_PRECISION_HINT>(_config, true, ov::PropertyMutability::RW, [isCompilerOptionSupported](const ov::AnyMap& arguments) {
        return isCompilerOptionSupported(ov::hint::inference_precision.name(), arguments);
    });
    register_property_with_support<QDQ_OPTIMIZATION>(_config, true, ov::PropertyMutability::RW, [isCompilerOptionSupported](const ov::AnyMap& arguments) {
        return isCompilerOptionSupported(ov::intel_npu::qdq_optimization.name(), arguments);
    });
    register_property_with_support<QDQ_OPTIMIZATION_AGGRESSIVE>(_config, true, ov::PropertyMutability::RW, [isCompilerOptionSupported](const ov::AnyMap& arguments) {
        return isCompilerOptionSupported(ov::intel_npu::qdq_optimization_aggressive.name(), arguments);
    });
    register_property_with_support<TILES>(_config, true, ov::PropertyMutability::RW, [isCompilerOptionSupported](const ov::AnyMap& arguments) {
        return isCompilerOptionSupported(ov::intel_npu::tiles.name(), arguments);
    });

    register_property_with_support<BACKEND_COMPILATION_PARAMS>(_config, false, ov::PropertyMutability::RW, [isCompilerOptionSupported](const ov::AnyMap& arguments) {
        return isCompilerOptionSupported(ov::intel_npu::backend_compilation_params.name(), arguments);
    });
    register_property_with_support<BATCH_COMPILER_MODE_SETTINGS>(_config, false, ov::PropertyMutability::RW, [isCompilerOptionSupported](const ov::AnyMap& arguments) {
        return isCompilerOptionSupported(ov::intel_npu::batch_compiler_mode_settings.name(), arguments);
    });
    register_property_with_support<BATCH_MODE>(_config, false, ov::PropertyMutability::RW, [isCompilerOptionSupported](const ov::AnyMap& arguments) {
        return isCompilerOptionSupported(ov::intel_npu::batch_mode.name(), arguments);
    });
    register_property_with_support<COMPILATION_MODE>(_config, false, ov::PropertyMutability::RW, [isCompilerOptionSupported](const ov::AnyMap& arguments) {
        return isCompilerOptionSupported(ov::intel_npu::compilation_mode.name(), arguments);
    });
    register_property_with_support<DMA_ENGINES>(_config, false, ov::PropertyMutability::RW, [isCompilerOptionSupported](const ov::AnyMap& arguments) {
        return isCompilerOptionSupported(ov::intel_npu::dma_engines.name(), arguments);
    });
    register_property_with_support<DYNAMIC_SHAPE_TO_STATIC>(_config, false, ov::PropertyMutability::RW, [isCompilerOptionSupported](const ov::AnyMap& arguments) {
        return isCompilerOptionSupported(ov::intel_npu::dynamic_shape_to_static.name(), arguments);
    });
    register_property_with_support<ENABLE_WEIGHTLESS>(_config, false, ov::PropertyMutability::RW, [isCompilerOptionSupported](const ov::AnyMap& arguments) {
        return isCompilerOptionSupported(ov::enable_weightless.name(), arguments);
    });
    register_property_with_support<MODEL_SERIALIZER_VERSION>(_config, false, ov::PropertyMutability::RW, [isCompilerOptionSupported](const ov::AnyMap& arguments) {
        return isCompilerOptionSupported(ov::intel_npu::model_serializer_version.name(), arguments);
    });
    register_property_with_support<SEPARATE_WEIGHTS_VERSION>(_config, false, ov::PropertyMutability::RW, [isCompilerOptionSupported](const ov::AnyMap& arguments) {
        return isCompilerOptionSupported(ov::intel_npu::separate_weights_version.name(), arguments);
    });

    register_property_with_custom_function<DEVICE_ID>(_config, true, ov::PropertyMutability::RW, [this, getDeviceId](const ov::AnyMap& arguments) -> ov::Any {
        const auto deviceId = getDeviceId(arguments);
        if (!deviceId.empty()) {
            return deviceId;
        }
        return _config.get<DEVICE_ID>();
    });
    register_property_with_custom_function<COMPILER_TYPE>(_config, true, ov::PropertyMutability::RW, [this, getCompilerTypeOrDefault](const ov::AnyMap& arguments) -> ov::Any {
        const auto compilerType = getCompilerTypeOrDefault(arguments);
        if (compilerType.has_value()) {
            return compilerType.value();
        }
        return _config.get<COMPILER_TYPE>();
    });

    register_property_with_custom_function<COMPILE_LOG_LEVEL>(_config, false, ov::PropertyMutability::RW, [this](const ov::AnyMap&) -> ov::Any {
        return COMPILE_LOG_LEVEL::resolve(_config);
    });
    register_property_with_custom_function<CACHE_ENCRYPTION_CALLBACKS>(_config, true, ov::PropertyMutability::WO, [](const ov::AnyMap&) {
        return ov::EncryptionCallbacks{nullptr, nullptr};
    });

    register_property_with_custom_function(ov::execution_devices.name(), true, ov::PropertyMutability::RO, [](const ov::AnyMap&) {
        return std::vector<std::string>{"NPU"};
    });
    register_property_with_custom_function(ov::device::capabilities.name(), true, ov::PropertyMutability::RO, [](const ov::AnyMap&) {
        return std::vector<std::string>{ov::device::capability::FP16,
                                        ov::device::capability::INT8,
                                        ov::device::capability::EXPORT_IMPORT};
    });
    register_property_with_custom_function(ov::range_for_async_infer_requests.name(), true, ov::PropertyMutability::RO, [](const ov::AnyMap&) {
        return std::tuple<uint32_t, uint32_t, uint32_t>{1u, 8u, 1u};
    });
    register_property_with_custom_function(ov::range_for_streams.name(), true, ov::PropertyMutability::RO, [](const ov::AnyMap&) {
        return std::tuple<uint32_t, uint32_t>{0u, 8u};
    });
    register_property_with_custom_function(ov::available_devices.name(), true, ov::PropertyMutability::RO, [this](const ov::AnyMap&) {
        return _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames();
    });
    register_property_with_custom_function(ov::hint::model.name(), true, ov::PropertyMutability::RO, [](const ov::AnyMap&) {
        return std::shared_ptr<const ov::Model>(nullptr);
    });
    register_property_with_custom_function(ov::supported_properties.name(), true, ov::PropertyMutability::RO, [this, getCompilerTypeOrDefault, getDeviceId](const ov::AnyMap& arguments) {
        auto resolvedArguments = arguments;
        if (const auto compilerType = getCompilerTypeOrDefault(arguments);
            compilerType.has_value() && compilerType.value() == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
            const auto platformIt = arguments.find(ov::intel_npu::platform.name());
            const auto platform = platformIt != arguments.end() ? platformIt->second.as<std::string>() : std::string{};
            const auto resolvedCompilerType = resolveCompilerType(compilerType.value(), getDeviceId(arguments), platform);
            if (resolvedCompilerType) {
                resolvedArguments[ov::intel_npu::compiler_type.name()] = *resolvedCompilerType;
            } else {
                resolvedArguments.erase(ov::intel_npu::compiler_type.name());
            }
        }

        std::vector<ov::PropertyName> supportedProperties;
        for (auto& property : _properties) {
            if (property.second.isPublic && property.second.isSupported(resolvedArguments)) {
                supportedProperties.emplace_back(ov::PropertyName(property.first, property.second.mutability));
            }
        }
        return supportedProperties;
    });

    register_property_with_custom_function(ov::internal::supported_properties.name(), false, ov::PropertyMutability::RO, [](const ov::AnyMap&) {
        return std::vector<ov::PropertyName>{ov::internal::caching_properties.name(),
                                             ov::internal::caching_with_mmap.name(),
                                             ov::internal::cache_header_alignment.name()};
    });
    register_property_with_custom_function(ov::internal::cache_header_alignment.name(), false, ov::PropertyMutability::RO, [](const ov::AnyMap&) {
        return utils::STANDARD_PAGE_SIZE;
    });
    register_property_with_custom_function(ov::internal::caching_properties.name(), false, ov::PropertyMutability::RO, [this, getCompilerTypeOrDefault, getDeviceId](const ov::AnyMap& arguments) {
        auto resolvedArguments = arguments;
        if (const auto compilerType = getCompilerTypeOrDefault(arguments);
            compilerType.has_value() && compilerType.value() == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
            const auto platformIt = arguments.find(ov::intel_npu::platform.name());
            const auto platform = platformIt != arguments.end() ? platformIt->second.as<std::string>() : std::string{};
            const auto resolvedCompilerType = resolveCompilerType(compilerType.value(), getDeviceId(arguments), platform);
            if (resolvedCompilerType) {
                resolvedArguments[ov::intel_npu::compiler_type.name()] = *resolvedCompilerType;
            } else {
                resolvedArguments.erase(ov::intel_npu::compiler_type.name());
            }
        }

        std::vector<ov::PropertyName> caching_props{};
        for (auto prop : _cachingProperties) {
            const auto propertyIt = _properties.find(prop);
            if (propertyIt != _properties.end() && propertyIt->second.isSupported(resolvedArguments)) {
                caching_props.emplace_back(prop);
            }
        }
        return caching_props;
    });
    register_property_with_custom_function<STEPPING>(_config, false, ov::PropertyMutability::RW, [this, getDeviceId](const ov::AnyMap& arguments) {
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


    // Special case: this property is always registered because it's supported by the implementation,
    // but it's not visible in supported_properties if the driver doesn't support it.
    register_property<RUN_INFERENCES_SEQUENTIALLY>(_config, [this] {
        if (_backend && _backend->getInitStructs()) {
            if (_backend->getInitStructs()->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 1)) {
                return true;
            }
        }
        return false;
    }(), ov::PropertyMutability::RW);

    register_property_with_support_and_custom_function(ov::intel_npu::driver_version.name(), true, ov::PropertyMutability::RO, hasBackendPredicate, [this](const ov::AnyMap&) {
        if (_backend == nullptr) {
            OPENVINO_THROW("No available backend");
        }
        return _backend->getDriverVersion();
    });
    register_property_with_support_and_custom_function(ov::device::pci_info.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        return utils::getPciInfo(_backend, getDeviceId(arguments));
    });
    register_property_with_support_and_custom_function(ov::device::gops.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        return utils::getGops(_backend, getDeviceId(arguments));
    });
    register_property_with_support_and_custom_function(ov::device::type.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        return utils::getDeviceType(_backend, getDeviceId(arguments));
    });
    register_property_with_support_and_custom_function(ov::intel_npu::device_alloc_mem_size.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        return utils::getDeviceAllocMemSize(_backend, getDeviceId(arguments));
    });
    register_property_with_support_and_custom_function(ov::intel_npu::device_total_mem_size.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        return utils::getDeviceTotalMemSize(_backend, getDeviceId(arguments));
    });
    register_property_with_support_and_custom_function(ov::device::uuid.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        auto devUuid = utils::getDeviceUuid(_backend, getDeviceId(arguments));
        return decltype(ov::device::uuid)::value_type{devUuid};
    });
    register_property_with_support_and_custom_function(ov::device::architecture.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        const auto devName = utils::getDeviceName(_backend, getDeviceId(arguments));
        return utils::getPlatformByDeviceName(devName);
    });
    register_property_with_support_and_custom_function(ov::device::full_name.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        return utils::getFullDeviceName(_backend, getDeviceId(arguments));
    });
    register_property_with_support_and_custom_function(ov::device::luid.name(), _backend != nullptr && _backend->isLUIDExtSupported(), ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        return utils::getDeviceLUID(_backend, getDeviceId(arguments));
    });

    register_property_with_support_and_custom_function(ov::intel_npu::backend_name.name(), false, ov::PropertyMutability::RO, hasBackendPredicate, [this](const ov::AnyMap&) {
        if (_backend == nullptr) {
            OPENVINO_THROW("No available backend");
        }
        return _backend->getName();
    });

    register_property_with_support_and_custom_function<MAX_TILES>(_config, true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
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
    register_property_with_support_and_custom_function<PLATFORM>(_config, true, ov::PropertyMutability::RW,
        [isCompilerOptionSupported](const ov::AnyMap& arguments) { // support predicate
            return isCompilerOptionSupported(ov::intel_npu::platform.name(), arguments);
        },
        [this](const ov::AnyMap& arguments) -> ov::Any { // custom getter
            const auto platformIt = arguments.find(ov::intel_npu::platform.name());
            if (platformIt != arguments.end()) {
                return platformIt->second;
            }
            return _config.get<PLATFORM>();
        });
    register_property_with_support_and_custom_function<TURBO>(_config, true, ov::PropertyMutability::RW,
        [this, isCompilerOptionSupported](const ov::AnyMap& arguments) {  // support predicate
            if (isCompilerOptionSupported(ov::intel_npu::turbo.name(), arguments)) {
                return true;
            }
            return _backend != nullptr && _backend->isCommandQueueExtSupported();
        },
        [this](const ov::AnyMap&) { // value getter
            return _config.get<TURBO>();
        });
    register_property_with_support_and_custom_function<ENABLE_STRIDES_FOR>(_config, true, ov::PropertyMutability::RW,
        [this, isCompilerOptionSupported](const ov::AnyMap& arguments) {  // support predicate
            if (!isCompilerOptionSupported(ov::intel_npu::enable_strides_for.name(), arguments)) {
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
    // clang-format on

    register_property_with_support_and_custom_function<COMPILER_VERSION>(
        _config,
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
                auto deviceId = getDeviceId(arguments);
                auto device = utils::getDeviceById(_backend, deviceId);
                platform = utils::getCompilationPlatform(
                    platformIt->second.as<std::string>(),
                    device == nullptr ? deviceId : device->getName(),
                    _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());
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
        [this, getCompilerTypeOrDefault, getDeviceId](const ov::AnyMap& arguments) {  // value getter
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
                auto deviceId = getDeviceId(arguments);
                auto device = utils::getDeviceById(_backend, deviceId);
                platform = utils::getCompilationPlatform(
                    platformIt->second.as<std::string>(),
                    device == nullptr ? deviceId : device->getName(),
                    _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());
            }

            CompilerAdapterFactory factory;
            auto dummyCompiler = factory.getCompiler(_backend, compilerType.value(), platform);
            return dummyCompiler->get_version();
        });

    register_property_with_support_and_custom_function(
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
        register_npuw_property<Opt>(_config);
    });
}

}  // namespace intel_npu
