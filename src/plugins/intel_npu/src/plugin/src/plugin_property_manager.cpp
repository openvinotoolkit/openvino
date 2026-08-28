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
// These properties are special because support is not guaranteed by either the compiler or the plugin.
bool isSpecialBothProperty(const std::string& key) {
    return key == ov::intel_npu::turbo.name();
}

void logCpuPinningDeprecationWarning(intel_npu::Logger& logger) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    logger.warning(intel_npu::ENABLE_CPU_PINNING::deprecationMessage());
    OPENVINO_SUPPRESS_DEPRECATED_END
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

PluginPropertyManager::PluginPropertyManager(const std::shared_ptr<FilteredConfig>& config,
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
    if (properties.find(ov::hint::enable_cpu_pinning.name()) != properties.end()) {
        logCpuPinningDeprecationWarning(_logger);
    }

    auto normalizedArguments = resolveRequestContext(properties,
                                                     _config->get<COMPILER_TYPE>(),
                                                     _config->get<DEVICE_ID>(),
                                                     _config->get<PLATFORM>());
    ov::AnyMap supportCheckArguments = {
        {ov::intel_npu::compiler_type.name(), normalizedArguments.compilerType},
        {ov::device::id.name(), normalizedArguments.deviceId},
        {ov::intel_npu::platform.name(), normalizedArguments.platform},
    };

    for (auto&& value : properties) {
        const auto propertyDescriptorIt = _properties.find(value.first);
        if (propertyDescriptorIt == _properties.end()) {
            // property doesn't exist - checking as internal now
            const auto resolvedCompilerType = resolveCompilerType(normalizedArguments.compilerType,
                                                                  normalizedArguments.deviceId,
                                                                  normalizedArguments.platform);
            if (!resolvedCompilerType.has_value()) {
                OPENVINO_THROW("Unsupported configuration key: ", value.first);
            }
            bool isSupported = false;
            try {
                isSupported =
                    _compilerOptionSupportHelper->isOptionSupported(resolvedCompilerType.value(), value.first);
            } catch (...) {
                // ignore any exceptions from the compiler and treat the property as unsupported
                isSupported = false;
            }
            if (!isSupported) {
                OPENVINO_THROW("Unsupported configuration key: ", value.first);
            }
        } else {
            const auto& descriptor = propertyDescriptorIt->second;
            if (descriptor.mutability == ov::PropertyMutability::RO) {
                OPENVINO_THROW("READ-ONLY configuration key: ", value.first);
            }
            if (!descriptor.isSupported(supportCheckArguments)) {
                OPENVINO_THROW("Unsupported configuration key: ", value.first);
            }
        }
    }

    for (auto&& value : properties) {
        const auto propertyDescriptorIt = _properties.find(value.first);
        if (propertyDescriptorIt == _properties.end()) {
            // if compiler reports it supported > registering as internal
            _config->addOrUpdateInternal(value.first, value.second.as<std::string>());
        } else {
            propertyDescriptorIt->second.set(value.second);
        }
    }
}

ov::Any PluginPropertyManager::getProperty(const std::string& name, const ov::AnyMap& arguments) const {
    if (name == ov::hint::enable_cpu_pinning.name()) {
        logCpuPinningDeprecationWarning(_logger);
    }

    auto normalizedArguments = resolveRequestContext(arguments,
                                                     _config->get<COMPILER_TYPE>(),
                                                     _config->get<DEVICE_ID>(),
                                                     _config->get<PLATFORM>());
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

    if (_config->hasInternal(name)) {
        auto resolvedCompilerType = resolveCompilerType(normalizedArguments.compilerType,
                                                        normalizedArguments.deviceId,
                                                        normalizedArguments.platform);
        if (!resolvedCompilerType.has_value()) {
            OPENVINO_THROW("Unsupported configuration key: ", name);
        }
        try {
            if (_compilerOptionSupportHelper->isOptionSupported(resolvedCompilerType.value(), name)) {
                return _config->getInternal(name);
            }
        } catch (...) {
        }
    }
    OPENVINO_THROW("Unsupported configuration key: ", name);
}

bool PluginPropertyManager::isPropertySupported(const std::string& name, const ov::AnyMap& arguments) const {
    if (name == ov::hint::enable_cpu_pinning.name()) {
        logCpuPinningDeprecationWarning(_logger);
    }
    if (!isPropertyRegistered(name)) {
        return false;
    }

    auto normalizedArguments = resolveRequestContext(arguments,
                                                     _config->get<COMPILER_TYPE>(),
                                                     _config->get<DEVICE_ID>(),
                                                     _config->get<PLATFORM>());
    auto propertyArguments = arguments;
    propertyArguments[ov::intel_npu::compiler_type.name()] = normalizedArguments.compilerType;
    propertyArguments[std::string(ov::device::id.name())] = normalizedArguments.deviceId;
    propertyArguments[ov::intel_npu::platform.name()] = normalizedArguments.platform;

    if (isSpecialBothProperty(name)) {
        // Fast path: Remove compiler type for special both properties and check if supported.
        propertyArguments.erase(ov::intel_npu::compiler_type.name());
        const auto it = _properties.find(name);
        if (it != _properties.end() && it->second.isPublic && it->second.isSupported(propertyArguments)) {
            return true;
        }
        propertyArguments[ov::intel_npu::compiler_type.name()] = normalizedArguments.compilerType;
    }

    const auto it = _properties.find(name);
    return it != _properties.end() && it->second.isPublic && it->second.isSupported(propertyArguments);
}

bool PluginPropertyManager::isPropertyAvailable(const std::string& name, const ov::AnyMap& arguments) const {
    if (name == ov::hint::enable_cpu_pinning.name()) {
        logCpuPinningDeprecationWarning(_logger);
    }

    auto normalizedArguments = resolveRequestContext(arguments,
                                                     _config->get<COMPILER_TYPE>(),
                                                     _config->get<DEVICE_ID>(),
                                                     _config->get<PLATFORM>());
    auto propertyArguments = arguments;
    propertyArguments[ov::intel_npu::compiler_type.name()] = normalizedArguments.compilerType;
    propertyArguments[std::string(ov::device::id.name())] = normalizedArguments.deviceId;
    propertyArguments[ov::intel_npu::platform.name()] = normalizedArguments.platform;

    if (isSpecialBothProperty(name)) {
        // Fast path: Remove compiler type for special both properties and check if supported.
        propertyArguments.erase(ov::intel_npu::compiler_type.name());
        const auto it = _properties.find(name);
        if (it != _properties.end() && it->second.isSupported(propertyArguments)) {
            return true;
        }
        propertyArguments[ov::intel_npu::compiler_type.name()] = normalizedArguments.compilerType;
    }

    const auto it = _properties.find(name);
    if (it != _properties.end()) {
        return it->second.isSupported(propertyArguments);
    } else if (_config->hasOpt(name)) {
        const auto mode = _config->getOpt(name).mode();
        if (mode != OptionMode::CompileTime) {
            return true;
        }
    }

    auto resolvedCompilerType = resolveCompilerType(normalizedArguments.compilerType,
                                                    normalizedArguments.deviceId,
                                                    normalizedArguments.platform);
    if (!resolvedCompilerType.has_value()) {
        return false;
    }
    try {
        return _compilerOptionSupportHelper->isOptionSupported(resolvedCompilerType.value(), name);
    } catch (...) {
        return false;
    }
}

bool PluginPropertyManager::isPropertyRegistered(const std::string& propertyName) const {
    return _properties.find(propertyName) != _properties.end();
}

std::optional<ov::intel_npu::CompilerType> PluginPropertyManager::resolveCompilerType(
    ov::intel_npu::CompilerType compilerType,
    const std::string& deviceId,
    const std::string& platform) const {
    if (compilerType != ov::intel_npu::CompilerType::PREFER_PLUGIN) {
        return compilerType;
    }

    try {
        auto device = utils::getDeviceById(_backend, deviceId);
        auto compilationPlatform = utils::getCompilationPlatform(
            platform,
            device == nullptr ? deviceId : device->getName(),
            _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());

        CompilerAdapterFactory factory;
        factory.decideCompilerType(compilerType, compilationPlatform);
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

    struct DeviceValidationCache {
        std::mutex mutex;
        std::optional<std::pair<std::string, bool>> value;
    };
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

                {
                    std::lock_guard<std::mutex> lock(hasBackendAndValidDeviceCache->mutex);
                    if (hasBackendAndValidDeviceCache->value.has_value() &&
                        hasBackendAndValidDeviceCache->value->first == specifiedDeviceName) {
                        return hasBackendAndValidDeviceCache->value->second;
                    }
                }

                const bool isValidDevice = utils::getDeviceById(_backend, specifiedDeviceName) != nullptr;

                {
                    std::lock_guard<std::mutex> lock(hasBackendAndValidDeviceCache->mutex);
                    hasBackendAndValidDeviceCache->value = std::make_pair(specifiedDeviceName, isValidDevice);
                }
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

    const auto isCompilerOptionSupported =
        [this, getCompilerTypeOrDefault, getDeviceId](const std::string& propertyName, const ov::AnyMap& arguments) {
            const auto compilerType = getCompilerTypeOrDefault(arguments);
            if (!compilerType.has_value()) {
                return false;
            }

            ov::intel_npu::CompilerType resolvedCompilerType;
            if (compilerType.value() == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
                const auto platformIt = arguments.find(ov::intel_npu::platform.name());
                const auto platform =
                    platformIt != arguments.end() ? platformIt->second.as<std::string>() : _config->get<PLATFORM>();
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

    const auto registerConfigProperty = [this](const auto optionTag, bool isPublic) {
        using OptionType = std::decay_t<decltype(optionTag)>;
        const auto propertyName = std::string(OptionType::key());
        register_property(
            propertyName,
            isPublic,
            ov::PropertyMutability::RW,
            [this, propertyName](const ov::AnyMap&) {
                return _config->hasOpt(propertyName);
            },
            [this](const ov::AnyMap&) {
                return _config->get<OptionType>();
            },
            [this, propertyName](const ov::Any& value) {
                _config->update({{propertyName, value.as<std::string>()}});
            });
    };

    registerConfigProperty(BYPASS_UMD_CACHING{}, true);
    registerConfigProperty(CACHE_DIR{}, true);
    registerConfigProperty(DEFER_WEIGHTS_LOAD{}, true);
    registerConfigProperty(MODEL_PRIORITY{}, true);
    registerConfigProperty(NUM_STREAMS{}, true);
    registerConfigProperty(PERF_COUNT{}, true);
    registerConfigProperty(PERFORMANCE_HINT{}, true);
    registerConfigProperty(PERFORMANCE_HINT_NUM_REQUESTS{}, true);
    registerConfigProperty(WEIGHTS_PATH{}, true);
    registerConfigProperty(WORKLOAD_TYPE{}, true);

    registerConfigProperty(COMPILED_BLOB{}, false);
    registerConfigProperty(CREATE_EXECUTOR{}, false);
    registerConfigProperty(DISABLE_VERSION_CHECK{}, false);
    registerConfigProperty(EXPORT_RAW_BLOB{}, false);
    registerConfigProperty(IMPORT_RAW_BLOB{}, false);
    registerConfigProperty(PROFILING_TYPE{}, false);
    registerConfigProperty(SHARED_COMMON_QUEUE{}, false);

    // Special case: this property is always registered because it's supported by the implementation,
    // but it's not visible in supported_properties if the driver doesn't support it.
    registerConfigProperty(RUN_INFERENCES_SEQUENTIALLY{}, [this] {
        if (_backend && _backend->getInitStructs()) {
            if (_backend->getInitStructs()->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 1)) {
                return true;
            }
        }
        return false;
    }());

    OPENVINO_SUPPRESS_DEPRECATED_START
    registerConfigProperty(ENABLE_CPU_PINNING{}, false);
    OPENVINO_SUPPRESS_DEPRECATED_END

    const auto registerCompilerProperty = [this, isCompilerOptionSupported](const auto optionTag, bool isPublic) {
        using OptionType = std::decay_t<decltype(optionTag)>;
        const auto propertyName = std::string(OptionType::key());
        register_property(
            propertyName,
            isPublic,
            ov::PropertyMutability::RW,
            [propertyName, isCompilerOptionSupported](const ov::AnyMap& arguments) {
                return isCompilerOptionSupported(propertyName, arguments);
            },
            [this](const ov::AnyMap&) {
                return _config->get<OptionType>();
            },
            [this, propertyName](const ov::Any& value) {
                _config->update({{propertyName, value.as<std::string>()}});
            });
    };

    registerCompilerProperty(CACHE_MODE{}, true);
    registerCompilerProperty(COMPILATION_MODE_PARAMS{}, true);
    registerCompilerProperty(COMPILATION_NUM_THREADS{}, true);
    registerCompilerProperty(COMPILER_DYNAMIC_QUANTIZATION{}, true);
    registerCompilerProperty(EXECUTION_MODE_HINT{}, true);
    registerCompilerProperty(INFERENCE_PRECISION_HINT{}, true);
    registerCompilerProperty(QDQ_OPTIMIZATION{}, true);
    registerCompilerProperty(QDQ_OPTIMIZATION_AGGRESSIVE{}, true);
    registerCompilerProperty(TILES{}, true);

    registerCompilerProperty(BACKEND_COMPILATION_PARAMS{}, false);
    registerCompilerProperty(BATCH_COMPILER_MODE_SETTINGS{}, false);
    registerCompilerProperty(BATCH_MODE{}, false);
    registerCompilerProperty(COMPILATION_MODE{}, false);
    registerCompilerProperty(DMA_ENGINES{}, false);
    registerCompilerProperty(DYNAMIC_SHAPE_TO_STATIC{}, false);
    registerCompilerProperty(ENABLE_WEIGHTLESS{}, false);
    registerCompilerProperty(MODEL_SERIALIZER_VERSION{}, false);
    registerCompilerProperty(SEPARATE_WEIGHTS_VERSION{}, false);

    // clang-format off
    register_property(ov::log::level.name(), true, ov::PropertyMutability::RW,
        [this](const ov::AnyMap&) {
            return _config->hasOpt(ov::log::level.name());
        },
        [this](const ov::AnyMap&) {
            return _config->get<LOG_LEVEL>();
        },
        [this](const ov::Any& value) {
            Logger::global().setLevel(value.as<ov::log::Level>());
            _logger.setLevel(value.as<ov::log::Level>());
            ov::AnyMap properties;
            properties[ov::log::level.name()] = value;
            if (_backend != nullptr) {
                _backend->updateInfo(properties);
            }
            _config->updateAny(properties);
        }
    );
    register_property(ov::intel_npu::disable_idle_memory_prunning.name(), true, ov::PropertyMutability::RW,
        [this](const ov::AnyMap&) {
            if (!_config->hasOpt(ov::intel_npu::disable_idle_memory_prunning.name())) {
                return false;
            }
            if (_backend != nullptr && _backend->isContextExtSupported()) {
                return true;
            }
            return false;
        },
        [this](const ov::AnyMap&) {
            return _config->get<DISABLE_IDLE_MEMORY_PRUNING>();
        },
        [this](const ov::Any& value) {
            ov::AnyMap properties;
            properties[ov::intel_npu::disable_idle_memory_prunning.name()] = value;
            if (_backend != nullptr) {
                _backend->updateInfo(properties);
            }
            _config->updateAny(properties);
        }
    );
    register_property(ov::device::id.name(), true, ov::PropertyMutability::RW,
        [this](const ov::AnyMap&) {
            return _config->hasOpt(ov::device::id.name());
        },
        [this, getDeviceId](const ov::AnyMap& arguments) -> ov::Any {
            const auto deviceId = getDeviceId(arguments);
            return deviceId.empty() ? ov::Any(_config->get<DEVICE_ID>()) : ov::Any(deviceId);
        },
        [this](const ov::Any& value) {
            _config->update({{ov::device::id.name(), value.as<std::string>()}});
        }
    );
    register_property(ov::intel_npu::compiler_type.name(), true, ov::PropertyMutability::RW,
        [this](const ov::AnyMap&) {
            return _config->hasOpt(ov::intel_npu::compiler_type.name());
        },
        [this, getCompilerTypeOrDefault](const ov::AnyMap& arguments) -> ov::Any {
            const auto compilerType = getCompilerTypeOrDefault(arguments);
            return compilerType.has_value() ? ov::Any(compilerType.value()) : ov::Any(_config->get<COMPILER_TYPE>());
        },
        [this](const ov::Any& value) {
            _config->update({{ov::intel_npu::compiler_type.name(), value.as<std::string>()}});
        }
    );
    register_property(ov::intel_npu::max_tiles.name(), true, ov::PropertyMutability::RO, 
        hasBackendAndValidDevice,
        [this, getDeviceId](const ov::AnyMap& arguments) {
            if (!_config->has<MAX_TILES>()) {
                try {
                    return static_cast<int64_t>(utils::getMaxTiles(_backend, getDeviceId(arguments)));
                } catch (...) {
                    _logger.warning("GetMaxTiles failed to get value from device.");
                }
            }
            return _config->get<MAX_TILES>();
        },
        [](const ov::Any&) {
            OPENVINO_THROW("READ-ONLY configuration key: ", ov::intel_npu::max_tiles.name());
        }
    );
    register_property(ov::intel_npu::platform.name(), true, ov::PropertyMutability::RW,
        [isCompilerOptionSupported](const ov::AnyMap& arguments) {
            return true;
        },
        [this](const ov::AnyMap& arguments) -> ov::Any {
            const auto platformIt = arguments.find(ov::intel_npu::platform.name());
            return platformIt != arguments.end() ? platformIt->second : ov::Any(_config->get<PLATFORM>());
        },
        [this](const ov::Any& value) {
            _config->update({{ov::intel_npu::platform.name(), value.as<std::string>()}});
        }
    );
    register_property(ov::intel_npu::turbo.name(), true, ov::PropertyMutability::RW,
        [this, isCompilerOptionSupported](const ov::AnyMap& arguments) {
            return isCompilerOptionSupported(ov::intel_npu::turbo.name(), arguments) || (_backend != nullptr && _backend->isCommandQueueExtSupported());
        },
        [this](const ov::AnyMap&) {
            return _config->get<TURBO>();
        },
        [this](const ov::Any& value) {
            _config->update({{ov::intel_npu::turbo.name(), value.as<std::string>()}});
        }
    );
    register_property(ov::intel_npu::enable_strides_for.name(), true, ov::PropertyMutability::RW,
        [this, isCompilerOptionSupported](const ov::AnyMap& arguments) {
            if (!isCompilerOptionSupported(ov::intel_npu::enable_strides_for.name(), arguments)) {
                return false;
            }
            if (_backend != nullptr && _backend->getGraphExtVersion() < ZE_MAKE_VERSION(1, 16)) {
                _logger.info("Config option %s not supported by the driver! Requirements not met.", ov::intel_npu::enable_strides_for.name());
                return false;
            }
            return true;
        },
        [this](const ov::AnyMap&) {
            return _config->get<ENABLE_STRIDES_FOR>();
        },
        [this](const ov::Any& value) {
            _config->update({{ov::intel_npu::enable_strides_for.name(), value.as<std::string>()}});
        }
    );
    register_property(ov::cache_encryption_callbacks.name(), true, ov::PropertyMutability::WO,
        [this](const ov::AnyMap&) {
            return _config->hasOpt(ov::cache_encryption_callbacks.name());
        },
        [](const ov::AnyMap&) {
            return ov::EncryptionCallbacks{nullptr, nullptr};
        },
        [this](const ov::Any& value) {
            _config->updateAny({{ov::cache_encryption_callbacks.name(), value}});
        }
    );

    register_property(ov::intel_npu::stepping.name(), false, ov::PropertyMutability::RW, 
        [this](const ov::AnyMap&) {
            return _config->hasOpt(ov::intel_npu::stepping.name());
        },
        [this, getDeviceId](const ov::AnyMap& arguments) {
            if (!_config->has<STEPPING>()) {
                try {
                    return static_cast<int64_t>(utils::getSteppingNumber(_backend, getDeviceId(arguments)));
                } catch (...) {
                    _logger.warning("GetSteppingNumber failed to get value from device.");
                }
            }
            return _config->get<STEPPING>();
        },
        [this](const ov::Any& value) {
            _config->update({{ov::intel_npu::stepping.name(), value.as<std::string>()}});
        }
    );
    register_property(ov::intel_npu::compile_log_level.name(), false, ov::PropertyMutability::RW,
        [this](const ov::AnyMap&) {
            return _config->hasOpt(ov::intel_npu::compile_log_level.name());
        },
        [this](const ov::AnyMap&) -> ov::Any {
            return COMPILE_LOG_LEVEL::resolve(*_config);
        },
        [this](const ov::Any& value) {
            _config->update({{ov::intel_npu::compile_log_level.name(), value.as<std::string>()}});
        }
    );

    const auto alwaysSupported = [](const ov::AnyMap&) {
        return true;
    };
    const auto readOnlySetter = [](const ov::Any&) {
        OPENVINO_THROW("Property is read-only");
    };

    register_property(ov::execution_devices.name(), true, ov::PropertyMutability::RO, alwaysSupported, [](const ov::AnyMap&) {
        return std::vector<std::string>{"NPU"};
    }, readOnlySetter);
    register_property(ov::device::capabilities.name(), true, ov::PropertyMutability::RO, alwaysSupported, [](const ov::AnyMap&) {
        return std::vector<std::string>{ov::device::capability::FP16, ov::device::capability::INT8, ov::device::capability::EXPORT_IMPORT};
    }, readOnlySetter);
    register_property(ov::range_for_async_infer_requests.name(), true, ov::PropertyMutability::RO, alwaysSupported, [](const ov::AnyMap&) {
        return std::tuple<uint32_t, uint32_t, uint32_t>{1u, 8u, 1u};
    }, readOnlySetter);
    register_property(ov::range_for_streams.name(), true, ov::PropertyMutability::RO, alwaysSupported, [](const ov::AnyMap&) {
        return std::tuple<uint32_t, uint32_t>{0u, 8u};
    }, readOnlySetter);
    register_property(ov::available_devices.name(), true, ov::PropertyMutability::RO, alwaysSupported, [this](const ov::AnyMap&) {
        return _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames();
    }, readOnlySetter);
    register_property(ov::hint::model.name(), true, ov::PropertyMutability::RO, alwaysSupported, [](const ov::AnyMap&) {
        return std::shared_ptr<const ov::Model>(nullptr);
    }, readOnlySetter);
    register_property(ov::supported_properties.name(), true, ov::PropertyMutability::RO, alwaysSupported, [this, getCompilerTypeOrDefault, getDeviceId](const ov::AnyMap& arguments) {
        auto resolvedArguments = arguments;
        if (const auto compilerType = getCompilerTypeOrDefault(arguments); compilerType.has_value() && compilerType.value() == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
            const auto platformIt = arguments.find(ov::intel_npu::platform.name());
            const auto platform = platformIt != arguments.end() ? platformIt->second.as<std::string>() : _config->get<PLATFORM>();
            const auto resolvedCompilerType = resolveCompilerType(compilerType.value(), getDeviceId(arguments), platform);
            if (resolvedCompilerType.has_value()) {
                resolvedArguments[ov::intel_npu::compiler_type.name()] = resolvedCompilerType.value();
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
    }, readOnlySetter);

    register_property(ov::internal::supported_properties.name(), false, ov::PropertyMutability::RO, alwaysSupported, [](const ov::AnyMap&) {
        return std::vector<ov::PropertyName>{ov::internal::caching_properties.name(),
                                             ov::internal::caching_with_mmap.name(),
                                             ov::internal::cache_header_alignment.name()};
    }, readOnlySetter);
    register_property(ov::internal::cache_header_alignment.name(), false, ov::PropertyMutability::RO, alwaysSupported, [](const ov::AnyMap&) {
        return utils::STANDARD_PAGE_SIZE;
    }, readOnlySetter);
    register_property(ov::internal::caching_properties.name(), false, ov::PropertyMutability::RO, alwaysSupported, [this, getCompilerTypeOrDefault, getDeviceId](const ov::AnyMap& arguments) {
        auto resolvedArguments = arguments;
        if (const auto compilerType = getCompilerTypeOrDefault(arguments); compilerType.has_value() && compilerType.value() == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
            const auto platformIt = arguments.find(ov::intel_npu::platform.name());
            const auto platform = platformIt != arguments.end() ? platformIt->second.as<std::string>() : _config->get<PLATFORM>();
            const auto resolvedCompilerType = resolveCompilerType(compilerType.value(), getDeviceId(arguments), platform);
            if (resolvedCompilerType.has_value()) {
                resolvedArguments[ov::intel_npu::compiler_type.name()] = resolvedCompilerType.value();
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
    }, readOnlySetter);

    register_property(ov::intel_npu::driver_version.name(), true, ov::PropertyMutability::RO, hasBackendPredicate, [this](const ov::AnyMap&) {
        if (_backend == nullptr) {
            OPENVINO_THROW("No available backend");
        }
        return _backend->getDriverVersion();
    }, readOnlySetter);
    register_property(ov::device::pci_info.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        return utils::getPciInfo(_backend, getDeviceId(arguments));
    }, readOnlySetter);
    register_property(ov::device::gops.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        return utils::getGops(_backend, getDeviceId(arguments));
    }, readOnlySetter);
    register_property(ov::device::type.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        return utils::getDeviceType(_backend, getDeviceId(arguments));
    }, readOnlySetter);
    register_property(ov::intel_npu::device_alloc_mem_size.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        return utils::getDeviceAllocMemSize(_backend, getDeviceId(arguments));
    }, readOnlySetter);
    register_property(ov::intel_npu::device_total_mem_size.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        return utils::getDeviceTotalMemSize(_backend, getDeviceId(arguments));
    }, readOnlySetter);
    register_property(ov::device::uuid.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        auto devUuid = utils::getDeviceUuid(_backend, getDeviceId(arguments));
        return decltype(ov::device::uuid)::value_type{devUuid};
    }, readOnlySetter);
    register_property(ov::device::architecture.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        const auto devName = utils::getDeviceName(_backend, getDeviceId(arguments));
        return utils::getPlatformByDeviceName(devName);
    }, readOnlySetter);
    register_property(ov::device::full_name.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        return utils::getFullDeviceName(_backend, getDeviceId(arguments));
    }, readOnlySetter);
    register_property(ov::device::luid.name(), _backend != nullptr && _backend->isLUIDExtSupported(), ov::PropertyMutability::RO, hasBackendAndValidDevice, [this, getDeviceId](const ov::AnyMap& arguments) {
        return utils::getDeviceLUID(_backend, getDeviceId(arguments));
    }, readOnlySetter);
    
    struct CompatibilityCheckSupportCache {
        std::mutex mutex;
        std::optional<bool> value;
    };
    auto compatibilityCheckSupportCache = std::make_shared<CompatibilityCheckSupportCache>();
    register_property(ov::compatibility_check.name(), true, ov::PropertyMutability::RO,
        [this, compatibilityCheckSupportCache](const ov::AnyMap&) {
            std::lock_guard<std::mutex> lock(compatibilityCheckSupportCache->mutex);
            if (!compatibilityCheckSupportCache->value.has_value()) {
                compatibilityCheckSupportCache->value = isCompatibilityCheckSupported(_backend, *_compilerOptionSupportHelper);
            }
            return compatibilityCheckSupportCache->value.value();
        },
        [this](const ov::AnyMap& arguments) { 
            return validateCompatibilityDescriptor(_backend, arguments, *_compilerOptionSupportHelper);
        },
        readOnlySetter
    );

    register_property(ov::intel_npu::backend_name.name(), false, ov::PropertyMutability::RO, hasBackendPredicate, [this](const ov::AnyMap&) {
        if (_backend == nullptr) {
            OPENVINO_THROW("No available backend");
        }
        return _backend->getName();
    }, readOnlySetter);

    register_property(ov::intel_npu::compiler_version.name(), true, ov::PropertyMutability::RO,
         [this, getCompilerTypeOrDefault, getDeviceId](const ov::AnyMap& arguments)  {  // support predicate
            auto compilerType = getCompilerTypeOrDefault(arguments);
            if (!compilerType.has_value()) {
                return false;
            }
            std::string compilationPlatform;
            if (compilerType.value() == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
                const auto platformIt = arguments.find(ov::intel_npu::platform.name());
                const auto platform = platformIt != arguments.end() ? platformIt->second.as<std::string>() : _config->get<PLATFORM>();
                auto deviceId = getDeviceId(arguments);
                auto device = utils::getDeviceById(_backend, deviceId);
                compilationPlatform = utils::getCompilationPlatform(
                    platform,
                    device == nullptr ? deviceId : device->getName(),
                    _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());
            }

            try {
                CompilerAdapterFactory factory;
                return factory.getCompiler(_backend, compilerType.value(), compilationPlatform) != nullptr;
            } catch (...) {
                return false;
            }
        },
        [this, getCompilerTypeOrDefault, getDeviceId](const ov::AnyMap& arguments) {  // value getter
            auto compilerType = getCompilerTypeOrDefault(arguments);
            if (!compilerType.has_value()) {
                OPENVINO_THROW("Compiler type is not specified in properties.");
            }
            std::string compilationPlatform;
            if (compilerType.value() == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
                const auto platformIt = arguments.find(ov::intel_npu::platform.name());
                const auto platform = platformIt != arguments.end() ? platformIt->second.as<std::string>() : _config->get<PLATFORM>();
                auto deviceId = getDeviceId(arguments);
                auto device = utils::getDeviceById(_backend, deviceId);
                compilationPlatform = utils::getCompilationPlatform(
                    platform,
                    device == nullptr ? deviceId : device->getName(),
                    _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());
            }

            CompilerAdapterFactory factory;
            return factory.getCompiler(_backend, compilerType.value(), compilationPlatform)->get_version();
        },
        [](const ov::Any&) {
            OPENVINO_THROW("READ-ONLY configuration key: ", ov::intel_npu::compiler_version.name());
        }
    );
    // clang-format on

    for_each_exposed_npuw_option([this](auto tag) {
        using OptionType = typename decltype(tag)::type;
        const auto propertyName = std::string(OptionType::key());
        register_property(
            propertyName,
            false,
            ov::PropertyMutability::RW,
            [this, propertyName](const ov::AnyMap&) {
                return _config->hasOpt(propertyName);
            },
            [this](const ov::AnyMap&) {
                return _config->get<OptionType>();
            },
            [this, propertyName](const ov::Any& value) {
                _config->update({{propertyName, value.as<std::string>()}});
            });
    });
}

}  // namespace intel_npu
