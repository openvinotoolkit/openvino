// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin_property_manager.hpp"

#include <algorithm>
#include <optional>
#include <tuple>
#include <utility>

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

inline bool isSpecialBothProperty(const std::string& key) {
    return key == ov::hint::performance_mode.name() || key == ov::enable_profiling.name() ||
           key == ov::log::level.name();
}

inline void logCpuPinningDeprecationWarning(intel_npu::Logger& logger) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    logger.warning(intel_npu::ENABLE_CPU_PINNING::deprecationMessage());
    OPENVINO_SUPPRESS_DEPRECATED_END
}

void filterPropertiesByCompilerSupport(intel_npu::FilteredConfig& config,
                                       const intel_npu::ICompilerAdapter* compiler,
                                       const ov::SoPtr<intel_npu::IEngineBackend>& backend,
                                       const intel_npu::Logger& logger) {
    using namespace intel_npu;

    bool legacy = false;
    std::optional<std::vector<std::string>> compilerSupportList{};
    uint32_t compilerVersion = 0;

    OPENVINO_ASSERT(compiler != nullptr, "Compiler must be present to filter properties by compiler support");

    compilerVersion = compiler->get_version();
    compilerSupportList = compiler->get_supported_options();

    if (!compilerSupportList.has_value()) {
        logger.info("No compiler support options list received! Fallback to version-based option registration");
        legacy = true;
    }

    logger.debug("Compiler version: %u", compilerVersion);
    logger.debug("Legacy registration: %s", legacy ? "true" : "false");
    if (!legacy) {
        const auto& supportedOptions = compilerSupportList.value();
        logger.debug("Compiler supported options list (%zu): ", supportedOptions.size());
        for (const auto& str : supportedOptions) {
            logger.debug("    %s ", str.c_str());
        }
    }

    config.walkEnables([&](const std::string& key) {
        bool isEnabled = false;
        auto opt = config.getOpt(key);
        if (opt.mode() != intel_npu::OptionMode::RunTime && !isSpecialBothProperty(key)) {
            if (legacy) {
                if (compilerVersion >= opt.compilerSupportVersion()) {
                    isEnabled = true;
                }
            } else {
                const auto& supportedOptions = compilerSupportList.value();
                auto it = std::find(supportedOptions.begin(), supportedOptions.end(), key);
                if (it != supportedOptions.end()) {
                    isEnabled = true;
                } else if (compiler != nullptr) {
                    isEnabled = compiler->is_option_supported(key);
                }
            }

            if (!isEnabled) {
                logger.debug("Config option %s not supported! Requirements not met.", key.c_str());
            } else {
                logger.debug("Enabled config option %s", key.c_str());
            }
            config.enable(key, isEnabled);
        }
    });

    if (backend && backend->isCommandQueueExtSupported()) {
        config.enable(ov::intel_npu::turbo.name(), true);
    }

    if (config.isAvailable(ov::intel_npu::enable_strides_for.name())) {
        if (backend && backend->getGraphExtVersion() < ZE_MAKE_VERSION(1, 16)) {
            logger.info("Config option %s not supported by the driver! Requirements not met.",
                        ov::intel_npu::enable_strides_for.name());
            config.enable(ov::intel_npu::enable_strides_for.name(), false);
        }
    }
}

void disableCompilerProperties(intel_npu::FilteredConfig& config, const ov::SoPtr<intel_npu::IEngineBackend>& backend) {
    using namespace intel_npu;

    config.walkEnables([&](const std::string& key) {
        auto opt = config.getOpt(key);
        if (opt.mode() != OptionMode::RunTime && !isSpecialBothProperty(key)) {
            config.enable(key, false);
        }
    });

    if (backend && backend->isCommandQueueExtSupported()) {
        config.enable(ov::intel_npu::turbo.name(), true);
    }
}

static auto get_specified_device_name(const intel_npu::Config& config) {
    if (config.has<intel_npu::DEVICE_ID>()) {
        return config.get<intel_npu::DEVICE_ID>();
    }
    return std::string();
}

std::shared_ptr<const ov::Model> exclude_model_ptr_from_map(ov::AnyMap& properties) {
    std::shared_ptr<const ov::Model> modelPtr = nullptr;
    if (properties.count(ov::hint::model.name())) {
        try {
            modelPtr = properties.at(ov::hint::model.name()).as<std::shared_ptr<const ov::Model>>();
        } catch (const ov::Exception&) {
            try {
                modelPtr = std::const_pointer_cast<const ov::Model>(
                    properties.at(ov::hint::model.name()).as<std::shared_ptr<ov::Model>>());
            } catch (const ov::Exception&) {
                OPENVINO_THROW("The value of the \"ov::hint::model\" configuration option (\"MODEL_PTR\") has the "
                               "wrong data type. Expected: std::shared_ptr<const ov::Model>.");
            }
        }
        properties.erase(ov::hint::model.name());
    }
    return modelPtr;
}

}  // namespace

namespace intel_npu {

PluginPropertyManager::PluginPropertyManager(const FilteredConfig& config,
                                             const std::shared_ptr<Metrics>& metrics,
                                             const ov::SoPtr<IEngineBackend>& backend,
                                             Logger& logger)
    : _config(config),
      _metrics(metrics),
      _backend(backend),
      _logger(logger) {
    registerProperties();
}

PluginPropertyManager::PluginPropertyManager(const PluginPropertyManager& other)
    : _config([&other]() {
          std::lock_guard<std::mutex> lock(other._mutex);
          return other._config;
      }()),
      _metrics(other._metrics),
      _backend(other._backend),
      _logger(other._logger) {
    std::lock_guard<std::mutex> lock(other._mutex);
    _currentlyUsedCompiler = other._currentlyUsedCompiler;
    _compilerForCompatibilityCheck = other._compilerForCompatibilityCheck;
    _currentlyUsedPlatform = other._currentlyUsedPlatform;
    _compilerConfigsFilteredByCompiler = other._compilerConfigsFilteredByCompiler;
    _compatibilityCheckFiltered = other._compatibilityCheckFiltered;
    _properties = other._properties;
    _supportedProperties = other._supportedProperties;
}

void PluginPropertyManager::registerProperties() const {
    _properties.clear();
    _supportedProperties.clear();

    registerPluginProperties();

    register_custom_metric(_properties, ov::supported_properties, true, [&](const Config&) {
        return _supportedProperties;
    });

    for (auto& property : _properties) {
        if (property.second.isPublic) {
            _supportedProperties.emplace_back(ov::PropertyName(property.first, property.second.mutability));
        }
    }
}

void PluginPropertyManager::registerPluginProperties() const {
    try_register_simple_property<PERF_COUNT>(_config, _properties, ov::enable_profiling);
    try_register_simple_property<PERFORMANCE_HINT>(_config, _properties, ov::hint::performance_mode);
    try_register_simple_property<EXECUTION_MODE_HINT>(_config, _properties, ov::hint::execution_mode);
    try_register_simple_property<PERFORMANCE_HINT_NUM_REQUESTS>(_config, _properties, ov::hint::num_requests);
    try_register_simple_property<COMPILATION_NUM_THREADS>(_config, _properties, ov::compilation_num_threads);
    try_register_simple_property<INFERENCE_PRECISION_HINT>(_config, _properties, ov::hint::inference_precision);
    try_register_simple_property<LOG_LEVEL>(_config, _properties, ov::log::level);
    try_register_simple_property<CACHE_DIR>(_config, _properties, ov::cache_dir);
    try_register_simple_property<CACHE_MODE>(_config, _properties, ov::cache_mode);
    try_register_simple_property<COMPILED_BLOB>(_config, _properties, ov::hint::compiled_blob);
    try_register_simple_property<DEVICE_ID>(_config, _properties, ov::device::id);
    try_register_simple_property<NUM_STREAMS>(_config, _properties, ov::num_streams);
    try_register_simple_property<WEIGHTS_PATH>(_config, _properties, ov::weights_path);
    try_register_simple_property<COMPILATION_MODE_PARAMS>(_config, _properties, ov::intel_npu::compilation_mode_params);
    try_register_simple_property<DMA_ENGINES>(_config, _properties, ov::intel_npu::dma_engines);
    try_register_simple_property<TILES>(_config, _properties, ov::intel_npu::tiles);
    try_register_simple_property<COMPILATION_MODE>(_config, _properties, ov::intel_npu::compilation_mode);
    try_register_simple_property<COMPILER_TYPE>(_config, _properties, ov::intel_npu::compiler_type);
    try_register_simple_property<PLATFORM>(_config, _properties, ov::intel_npu::platform);
    try_register_simple_property<CREATE_EXECUTOR>(_config, _properties, ov::intel_npu::create_executor);
    try_register_simple_property<DYNAMIC_SHAPE_TO_STATIC>(_config, _properties, ov::intel_npu::dynamic_shape_to_static);
    try_register_simple_property<PROFILING_TYPE>(_config, _properties, ov::intel_npu::profiling_type);
    try_register_simple_property<BACKEND_COMPILATION_PARAMS>(_config,
                                                             _properties,
                                                             ov::intel_npu::backend_compilation_params);
    try_register_simple_property<BATCH_MODE>(_config, _properties, ov::intel_npu::batch_mode);
    try_register_simple_property<TURBO>(_config, _properties, ov::intel_npu::turbo);
    try_register_simple_property<MODEL_PRIORITY>(_config, _properties, ov::hint::model_priority);
    try_register_simple_property<BYPASS_UMD_CACHING>(_config, _properties, ov::intel_npu::bypass_umd_caching);
    try_register_simple_property<DEFER_WEIGHTS_LOAD>(_config, _properties, ov::intel_npu::defer_weights_load);
    try_register_simple_property<COMPILER_DYNAMIC_QUANTIZATION>(_config,
                                                                _properties,
                                                                ov::intel_npu::compiler_dynamic_quantization);
    try_register_simple_property<QDQ_OPTIMIZATION>(_config, _properties, ov::intel_npu::qdq_optimization);
    try_register_simple_property<QDQ_OPTIMIZATION_AGGRESSIVE>(_config,
                                                              _properties,
                                                              ov::intel_npu::qdq_optimization_aggressive);
    try_register_simple_property<DISABLE_VERSION_CHECK>(_config, _properties, ov::intel_npu::disable_version_check);
    try_register_simple_property<EXPORT_RAW_BLOB>(_config, _properties, ov::intel_npu::export_raw_blob);
    try_register_simple_property<IMPORT_RAW_BLOB>(_config, _properties, ov::intel_npu::import_raw_blob);
    try_register_simple_property<BATCH_COMPILER_MODE_SETTINGS>(_config,
                                                               _properties,
                                                               ov::intel_npu::batch_compiler_mode_settings);
    OPENVINO_SUPPRESS_DEPRECATED_START
    try_register_simple_property<ENABLE_CPU_PINNING>(_config, _properties, ov::hint::enable_cpu_pinning);
    OPENVINO_SUPPRESS_DEPRECATED_END
    try_register_simple_property<WORKLOAD_TYPE>(_config, _properties, ov::workload_type);
    try_register_simple_property<ENABLE_WEIGHTLESS>(_config, _properties, ov::enable_weightless);
    try_register_simple_property<SEPARATE_WEIGHTS_VERSION>(_config,
                                                           _properties,
                                                           ov::intel_npu::separate_weights_version);
    try_register_simple_property<MODEL_SERIALIZER_VERSION>(_config,
                                                           _properties,
                                                           ov::intel_npu::model_serializer_version);
    try_register_simple_property<ENABLE_STRIDES_FOR>(_config, _properties, ov::intel_npu::enable_strides_for);
    try_register_simple_property<DISABLE_IDLE_MEMORY_PRUNING>(_config,
                                                              _properties,
                                                              ov::intel_npu::disable_idle_memory_prunning);
    try_register_simple_property<SHARED_COMMON_QUEUE>(_config, _properties, ov::intel_npu::shared_common_queue);

    try_register_customfunc_property(_config, _properties, ov::intel_npu::stepping, [&](const Config& config) {
        if (!config.has<STEPPING>()) {
            try {
                const auto specifiedDeviceName = get_specified_device_name(config);
                return static_cast<int64_t>(_metrics->GetSteppingNumber(specifiedDeviceName));
            } catch (...) {
                _logger.warning("Metrics GetSteppingNumber failed to get value from device.");
            }
        }
        return config.get<STEPPING>();
    });
    try_register_customfunc_property(_config, _properties, ov::intel_npu::max_tiles, [&](const Config& config) {
        if (!config.has<MAX_TILES>()) {
            try {
                const auto specifiedDeviceName = get_specified_device_name(config);
                return static_cast<int64_t>(_metrics->GetMaxTiles(specifiedDeviceName));
            } catch (...) {
                _logger.warning("Metrics GetMaxTiles failed to get value from device.");
            }
            return config.get<MAX_TILES>();
        }
    });

    try_register_varpub_property<RUN_INFERENCES_SEQUENTIALLY>(
        _config,
        _properties,
        ov::intel_npu::run_inferences_sequentially,
        [&] {
            if (_backend && _backend->getInitStructs()) {
                if (_backend->getInitStructs()->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 1)) {
                    return true;
                }
            }
            return false;
        }());
    if (_config.isAvailable(ov::compatibility_check.name())) {
        register_named_property_with_args(_properties,
                                          ov::compatibility_check.name(),
                                          true,
                                          ov::PropertyMutability::RO,
                                          [this](const Config&, const ov::AnyMap& arguments) {
                                              return validateCompatibilityDescriptor(
                                                  determineCompilerTypeForCompatibilityCheck(),
                                                  arguments);
                                          });
    }
    try_register_custom_property(_config,
                                 _properties,
                                 ov::cache_encryption_callbacks,
                                 true,
                                 ov::PropertyMutability::WO,
                                 [](const Config&) {
                                     return ov::EncryptionCallbacks{nullptr, nullptr};
                                 });
    force_register_custom_property(_properties, ov::hint::model, true, ov::PropertyMutability::RO, [](const Config&) {
        return std::shared_ptr<const ov::Model>(nullptr);
    });

    for_each_exposed_npuw_option([&](auto tag) {
        using Opt = typename decltype(tag)::type;
        try_register_npuw_option_property<Opt>(_config, _properties);
    });

    if (_metrics != nullptr) {
        register_simple_metric(_properties, ov::available_devices, true, [&](const Config&) {
            return _metrics->GetAvailableDevicesNames();
        });
        register_simple_metric(_properties, ov::device::capabilities, true, [&](const Config&) {
            return _metrics->GetOptimizationCapabilities();
        });
        register_simple_metric(_properties, ov::optimal_number_of_infer_requests, true, [&](const Config& config) {
            return utils::getOptimalNumberOfInferRequestsInParallel(
                utils::getCompilationPlatform(
                    config.get<PLATFORM>(),
                    _backend == nullptr ? config.get<DEVICE_ID>()
                                        : _backend->getDevice(config.get<DEVICE_ID>())->getName(),
                    _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames()),
                config.get<PERFORMANCE_HINT>());
        });
        register_simple_metric(_properties, ov::range_for_async_infer_requests, true, [&](const Config&) {
            return _metrics->GetRangeForAsyncInferRequest();
        });
        register_simple_metric(_properties, ov::range_for_streams, true, [&](const Config&) {
            return _metrics->GetRangeForStreams();
        });
        register_simple_metric(_properties, ov::device::pci_info, true, [&](const Config& config) {
            return _metrics->GetPciInfo(get_specified_device_name(config));
        });
        register_simple_metric(_properties, ov::device::gops, true, [&](const Config& config) {
            return _metrics->GetGops(get_specified_device_name(config));
        });
        register_simple_metric(_properties, ov::device::type, true, [&](const Config& config) {
            return _metrics->GetDeviceType(get_specified_device_name(config));
        });
        register_custom_metric(_properties, ov::internal::supported_properties, false, [&](const Config&) {
            return _internalSupportedProperties;
        });
        register_simple_metric(_properties, ov::internal::cache_header_alignment, false, [&](const Config&) {
            return utils::STANDARD_PAGE_SIZE;
        });
        register_simple_metric(_properties, ov::intel_npu::device_alloc_mem_size, true, [&](const Config& config) {
            return _metrics->GetDeviceAllocMemSize(get_specified_device_name(config));
        });
        register_simple_metric(_properties, ov::intel_npu::device_total_mem_size, true, [&](const Config& config) {
            return _metrics->GetDeviceTotalMemSize(get_specified_device_name(config));
        });
        register_simple_metric(_properties, ov::intel_npu::driver_version, true, [&](const Config&) {
            return _metrics->GetDriverVersion();
        });
        register_simple_metric(_properties, ov::intel_npu::backend_name, false, [&](const Config&) {
            return _metrics->GetBackendName();
        });
        register_custom_metric(_properties,
                               ov::device::architecture,
                               !_metrics->GetAvailableDevicesNames().empty(),
                               [&](const Config& config) {
                                   const auto specifiedDeviceName = get_specified_device_name(config);
                                   return _metrics->GetDeviceArchitecture(specifiedDeviceName);
                               });
        register_custom_metric(_properties,
                               ov::device::full_name,
                               !_metrics->GetAvailableDevicesNames().empty(),
                               [&](const Config& config) {
                                   const auto specifiedDeviceName = get_specified_device_name(config);
                                   return _metrics->GetFullDeviceName(specifiedDeviceName);
                               });
        register_custom_metric(_properties,
                               ov::device::luid,
                               _backend == nullptr ? false : _backend->isLUIDExtSupported(),
                               [&](const Config& config) {
                                   const auto specifiedDeviceName = get_specified_device_name(config);
                                   return _metrics->GetDeviceLUID(specifiedDeviceName);
                               });
        register_custom_metric(_properties, ov::device::uuid, true, [&](const Config& config) {
            const auto specifiedDeviceName = get_specified_device_name(config);
            auto devUuid = _metrics->GetDeviceUuid(specifiedDeviceName);
            return decltype(ov::device::uuid)::value_type{devUuid};
        });
        register_custom_metric(_properties, ov::execution_devices, true, [&](const Config& config) {
            if (_metrics->GetAvailableDevicesNames().size() > 1) {
                return std::string("NPU." + config.get<DEVICE_ID>());
            }
            return std::string("NPU");
        });
        register_custom_metric(_properties, ov::intel_npu::compiler_version, true, [&](const Config& config) {
            auto compilerType = config.get<COMPILER_TYPE>();
            auto deviceId = config.get<DEVICE_ID>();
            auto device = utils::getDeviceById(_backend, deviceId);

            auto compilationPlatform = utils::getCompilationPlatform(
                config.get<PLATFORM>(),
                device == nullptr ? std::move(deviceId) : device->getName(),
                _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());

            CompilerAdapterFactory factory;
            auto dummyCompiler = factory.getCompiler(_backend, compilerType, compilationPlatform);

            return dummyCompiler->get_version();
        });
        register_custom_metric(_properties, ov::internal::caching_properties, false, [&](const Config&) {
            std::vector<ov::PropertyName> caching_props{};
            for (auto prop : _cachingProperties) {
                if (_config.isAvailable(prop)) {
                    caching_props.emplace_back(prop);
                }
            }
            for (auto prop : _cachingProperties) {
                if (prop.find("NPUW") != prop.npos) {
                    caching_props.emplace_back(prop);
                }
            }
            return caching_props;
        });
    }
}

void PluginPropertyManager::setProperty(const ov::AnyMap& properties) {
    std::lock_guard<std::mutex> lock(_mutex);

    auto resolveCompilerTypeWithoutLock = [&](const ov::AnyMap& propertyMap) {
        auto compilerTypeIt = propertyMap.find(ov::intel_npu::compiler_type.name());
        if (compilerTypeIt != propertyMap.end()) {
            return COMPILER_TYPE::parse(compilerTypeIt->second.as<std::string>());
        }
        return _config.get<COMPILER_TYPE>();
    };

    auto resolveDeviceIdWithoutLock = [&](const ov::AnyMap& propertyMap) {
        auto deviceIdIt = propertyMap.find(std::string(ov::device::id.name()));
        if (deviceIdIt != propertyMap.end()) {
            return deviceIdIt->second.as<std::string>();
        }
        return _config.get<DEVICE_ID>();
    };

    auto resolvePlatformWithoutLock = [&](const ov::AnyMap& propertyMap) {
        auto platformIt = propertyMap.find(ov::intel_npu::platform.name());
        if (platformIt != propertyMap.end()) {
            return platformIt->second.as<std::string>();
        }
        return _config.get<PLATFORM>();
    };

    if (properties.find(ov::log::level.name()) != properties.end()) {
        _logger.setLevel(properties.at(ov::log::level.name()).as<ov::log::Level>());
    }

    if (properties.find(ov::hint::enable_cpu_pinning.name()) != properties.end()) {
        logCpuPinningDeprecationWarning(_logger);
    }

    std::unique_ptr<ICompilerAdapter> compiler = nullptr;
    bool propertyIsCompilerConfig = false;
    bool propertyIsRegistered = true;
    for (const auto& property : properties) {
        if (!isPropertyRegistered(property.first)) {
            propertyIsRegistered = false;
            break;
        }
        const bool isNotSpecialBothProperty = !isSpecialBothProperty(property.first);
        if (_config.hasOpt(property.first) && isNotSpecialBothProperty) {
            auto opt = _config.getOpt(property.first);
            if (opt.mode() != OptionMode::RunTime) {
                propertyIsCompilerConfig = true;
                break;
            }
        }
    }

    if (propertyIsCompilerConfig || !propertyIsRegistered) {
        auto compilerType = resolveCompilerTypeWithoutLock(properties);
        auto deviceId = resolveDeviceIdWithoutLock(properties);
        auto device = utils::getDeviceById(_backend, deviceId);

        auto compilationPlatform = utils::getCompilationPlatform(
            resolvePlatformWithoutLock(properties),
            device == nullptr ? std::move(deviceId) : device->getName(),
            _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());

        CompilerAdapterFactory factory;
        compiler = factory.getCompiler(_backend, compilerType, compilationPlatform);

        if (!(_compilerConfigsFilteredByCompiler && compilerType == _currentlyUsedCompiler &&
              compilationPlatform == _currentlyUsedPlatform)) {
            filterPropertiesByCompilerSupport(_config, compiler.get(), _backend, _logger);
            registerProperties();
            _compilerConfigsFilteredByCompiler = true;
            _currentlyUsedCompiler = compilerType;
            _currentlyUsedPlatform = std::move(compilationPlatform);
        }
    }

    std::map<std::string, std::string> cfgs_to_set;
    ov::AnyMap special_cfgs_to_set;
    for (auto&& value : properties) {
        if (_properties.find(value.first) == _properties.end()) {
            if (compiler != nullptr) {
                if (compiler->is_option_supported(value.first)) {
                    _config.addOrUpdateInternal(value.first, value.second.as<std::string>());
                } else {
                    OPENVINO_THROW("Unsupported configuration key: ", value.first);
                }
            } else {
                OPENVINO_THROW("Unsupported configuration key: ", value.first);
            }
        } else {
            if (_properties[value.first].mutability == ov::PropertyMutability::RO) {
                OPENVINO_THROW("READ-ONLY configuration key: ", value.first);
            } else if (value.first == ov::cache_encryption_callbacks.name()) {
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

ov::Any PluginPropertyManager::getProperty(const std::string& name, const ov::AnyMap& arguments) const {
    if (!arguments.empty() && name != ov::compatibility_check.name()) {
        auto pluginArguments = arguments;
        exclude_model_ptr_from_map(pluginArguments);

        auto copyProperties = PluginPropertyManager(*this);
        copyProperties.setProperty(pluginArguments);
        return copyProperties.getProperty(name);
    }

    std::lock_guard<std::mutex> lock(_mutex);

    if (name == ov::hint::enable_cpu_pinning.name()) {
        logCpuPinningDeprecationWarning(_logger);
    }

    bool propertyIsCompilerConfig = false;
    bool propertyIsRegistered = true;
    if (!isPropertyRegistered(name)) {
        propertyIsRegistered = false;
    } else if (_config.hasOpt(name) && !isSpecialBothProperty(name)) {
        auto opt = _config.getOpt(name);
        if (opt.mode() != OptionMode::RunTime) {
            propertyIsCompilerConfig = true;
        }
    }

    bool needToResetProperties = false;
    if (name == ov::compatibility_check.name() || name == ov::supported_properties.name()) {
        needToResetProperties = disableCompatibilityCheckIfNeeded();
    }
    if (propertyIsCompilerConfig || !propertyIsRegistered || name == ov::supported_properties.name() ||
        name == ov::internal::caching_properties.name()) {
        std::unique_ptr<ICompilerAdapter> compiler = nullptr;
        auto compilerType = _config.get<COMPILER_TYPE>();
        auto deviceId = _config.get<DEVICE_ID>();
        auto device = utils::getDeviceById(_backend, deviceId);

        auto compilationPlatform = utils::getCompilationPlatform(
            _config.get<PLATFORM>(),
            device == nullptr ? std::move(deviceId) : device->getName(),
            _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());

        CompilerAdapterFactory factory;
        try {
            compiler = factory.getCompiler(_backend, compilerType, compilationPlatform);
        } catch (const std::exception& ex) {
            if (_config.hasOpt(name) && _config.getOpt(name).mode() == OptionMode::CompileTime) {
                OPENVINO_THROW("Failed to create compiler for getting property ", name, " with error: ", ex.what());
            }

            _logger.warning("Failed to create compiler for getting property %s with error: %s."
                            "Returning only runtime properties and metrics that do not require compiler support.",
                            name.c_str(),
                            ex.what());
        }

        if (compiler != nullptr && !(_compilerConfigsFilteredByCompiler && compilerType == _currentlyUsedCompiler &&
                                     compilationPlatform == _currentlyUsedPlatform)) {
            filterPropertiesByCompilerSupport(_config, compiler.get(), _backend, _logger);

            _compilerConfigsFilteredByCompiler = true;
            _currentlyUsedCompiler = compilerType;
            _currentlyUsedPlatform = std::move(compilationPlatform);
            needToResetProperties = true;
        }
    }

    if (needToResetProperties) {
        registerProperties();
    }

    auto&& configIterator = _properties.find(name);
    if (configIterator != _properties.cend()) {
        if (configIterator->second.getWithArgs) {
            return configIterator->second.getWithArgs(_config, arguments);
        }
        if (configIterator->second.mutability == ov::PropertyMutability::WO) {
            _logger.warning("Trying to get WRITE-ONLY property: %s. Returning empty `ov::Any` object", name.c_str());
            return ov::Any();
        }
        return configIterator->second.get(_config);
    }
    try {
        return _config.getInternal(name);
    } catch (...) {
        OPENVINO_THROW("Unsupported configuration key: ", name);
    }
}

bool PluginPropertyManager::isPropertySupported(const std::string& name, const ov::AnyMap& arguments) const {
    if (!arguments.empty()) {
        auto pluginArguments = arguments;
        exclude_model_ptr_from_map(pluginArguments);

        auto copyProperties = PluginPropertyManager(*this);
        try {
            copyProperties.setProperty(pluginArguments);
        } catch (...) {
            return false;
        }

        return copyProperties.isPropertySupported(name);
    }

    std::lock_guard<std::mutex> lock(_mutex);
    if (name == ov::hint::enable_cpu_pinning.name()) {
        logCpuPinningDeprecationWarning(_logger);
    }

    const bool isRegistered = isPropertyRegistered(name);
    const bool isConfigOption = _config.hasOpt(name);

    if (!isRegistered && !isConfigOption) {
        return false;
    }

    if (name == ov::compatibility_check.name()) {
        bool disabled = disableCompatibilityCheckIfNeeded();
        if (disabled) {
            registerProperties();
            return false;
        }
    }

    if (isRegistered) {
        if (!isConfigOption || isSpecialBothProperty(name)) {
            return true;
        }

        auto opt = _config.getOpt(name);
        if (opt.mode() == OptionMode::RunTime) {
            return true;
        }
    }

    std::unique_ptr<ICompilerAdapter> compiler = nullptr;
    auto compilerType = _config.get<COMPILER_TYPE>();
    auto deviceId = _config.get<DEVICE_ID>();
    auto device = utils::getDeviceById(_backend, deviceId);

    auto compilationPlatform =
        utils::getCompilationPlatform(_config.get<PLATFORM>(),
                                      device == nullptr ? std::move(deviceId) : device->getName(),
                                      _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());

    CompilerAdapterFactory factory;
    try {
        compiler = factory.getCompiler(_backend, compilerType, compilationPlatform);
    } catch (const std::exception& ex) {
        if (_config.hasOpt(name) && _config.getOpt(name).mode() == OptionMode::CompileTime) {
            return false;
        }

        _logger.warning("Failed to create compiler to query property %s with error: %s. "
                        "Registering only runtime properties and metrics that do not require compiler support.",
                        name.c_str(),
                        ex.what());
    }

    if (compiler != nullptr && !(_compilerConfigsFilteredByCompiler && compilerType == _currentlyUsedCompiler &&
                                 compilationPlatform == _currentlyUsedPlatform)) {
        filterPropertiesByCompilerSupport(_config, compiler.get(), _backend, _logger);

        registerProperties();
        _compilerConfigsFilteredByCompiler = true;
        _currentlyUsedCompiler = compilerType;
        _currentlyUsedPlatform = std::move(compilationPlatform);
    }

    return isPropertyRegistered(name);
}

FilteredConfig PluginPropertyManager::getConfigWithCompilerPropertiesDisabled(const ov::AnyMap& properties) const {
    auto [updatedConfig, compilerConfigsFilteredByCompiler, logger] = [&]() {
        std::lock_guard<std::mutex> lock(_mutex);
        return std::make_tuple(_config, _compilerConfigsFilteredByCompiler, _logger);
    }();

    if (compilerConfigsFilteredByCompiler) {
        disableCompilerProperties(updatedConfig, _backend);
    }

    if (properties.find(ov::hint::enable_cpu_pinning.name()) != properties.end()) {
        logCpuPinningDeprecationWarning(logger);
    }

    if (properties.empty()) {
        return std::move(updatedConfig);
    }

    const std::map<std::string, std::string> rawConfig = any_copy(properties);
    std::map<std::string, std::string> cfgsToSet;
    ov::AnyMap specialCfgsToSet;
    for (const auto& [key, value] : rawConfig) {
        if (updatedConfig.hasOpt(key)) {
            const auto optionMode = updatedConfig.getOpt(key).mode();

            if (optionMode == OptionMode::CompileTime) {
                logger.info(
                    "Config key '%s' is recognized as a compiler option, will not be used for current configuration.",
                    key.c_str());
                continue;
            }

            if (optionMode == OptionMode::Both && !updatedConfig.isAvailable(key)) {
                logger.info("Config key '%s' is not enabled by the plugin, will not be used for current configuration.",
                            key.c_str());
                continue;
            }
        }

        if (key == ov::cache_encryption_callbacks.name()) {
            specialCfgsToSet.emplace(key, properties.at(key));
        } else {
            cfgsToSet.emplace(key, value);
        }
    }

    updatedConfig.update(cfgsToSet);
    updatedConfig.updateAny(specialCfgsToSet);

    return std::move(updatedConfig);
}

FilteredConfig PluginPropertyManager::getConfigForSpecificCompiler(const ov::AnyMap& properties,
                                                                   const ICompilerAdapter* compiler) const {
    auto [updatedConfig, compilerConfigsFilteredByCompiler, currentlyUsedCompiler, currentlyUsedPlatform, logger] =
        [&]() {
            std::lock_guard<std::mutex> lock(_mutex);
            return std::make_tuple(_config,
                                   _compilerConfigsFilteredByCompiler,
                                   _currentlyUsedCompiler,
                                   _currentlyUsedPlatform,
                                   _logger);
        }();

    if (properties.find(ov::hint::enable_cpu_pinning.name()) != properties.end()) {
        logCpuPinningDeprecationWarning(logger);
    }

    std::optional<ov::intel_npu::CompilerType> propertiesCompilerType = std::nullopt;
    std::optional<std::string> propertiesPlatform = std::nullopt;
    if (compilerConfigsFilteredByCompiler) {
        auto compilerType = properties.find(ov::intel_npu::compiler_type.name());
        if (compilerType != properties.end()) {
            propertiesCompilerType = compilerType->second.as<ov::intel_npu::CompilerType>();
        }
    }
    auto platform = properties.find(ov::intel_npu::platform.name());
    if (platform != properties.end()) {
        propertiesPlatform = platform->second.as<std::string>();
    }

    if (!(compilerConfigsFilteredByCompiler &&
          propertiesCompilerType.value_or(currentlyUsedCompiler) == currentlyUsedCompiler &&
          propertiesPlatform.value_or(currentlyUsedPlatform) == currentlyUsedPlatform)) {
        filterPropertiesByCompilerSupport(updatedConfig, compiler, _backend, logger);
    }

    const std::map<std::string, std::string> rawConfig = any_copy(properties);
    std::map<std::string, std::string> cfgsToSet;
    ov::AnyMap specialCfgsToSet;
    for (const auto& [key, value] : rawConfig) {
        if (!updatedConfig.hasOpt(key)) {
            if (!compiler->is_option_supported(key)) {
                OPENVINO_THROW("[ NOT_FOUND ] Option '", key, "' is not supported for current configuration");
            }
            updatedConfig.addOrUpdateInternal(key, value);
        } else if (key == ov::cache_encryption_callbacks.name()) {
            specialCfgsToSet.emplace(key, properties.at(key));
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

bool PluginPropertyManager::disableCompatibilityCheckIfNeeded() const {
    if (_compatibilityCheckFiltered) {
        return false;
    }

    _compatibilityCheckFiltered = true;

    CompilerAdapterFactory factory;
    auto compilerType = ov::intel_npu::CompilerType::DRIVER;
    try {
        auto tempCompiler = factory.getCompiler(_backend, compilerType, std::string_view{});
        if (!tempCompiler->is_option_supported(ov::compatibility_check.name())) {
            compilerType = ov::intel_npu::CompilerType::PLUGIN;
            try {
                tempCompiler = factory.getCompiler(_backend, compilerType, std::string_view{});
                if (!tempCompiler->is_option_supported(ov::compatibility_check.name())) {
                    _logger.debug("Neither CID nor CIP support the compatibility check! Disabling the property.");
                    _config.enable(ov::compatibility_check.name(), false);
                    return true;
                }
                _compilerForCompatibilityCheck = ov::intel_npu::CompilerType::PLUGIN;
            } catch (const std::exception&) {
                _logger.debug("CIP is not present! Disabling the compatibility check property.");
                _config.enable(ov::compatibility_check.name(), false);
                return true;
            }
        } else {
            _compilerForCompatibilityCheck = ov::intel_npu::CompilerType::DRIVER;
        }
    } catch (const std::exception&) {
        _logger.debug("Driver is not present! Disabling the compatibility check property.");
        _config.enable(ov::compatibility_check.name(), false);
        return true;
    }

    return false;
}

ov::intel_npu::CompilerType PluginPropertyManager::determineCompilerTypeForCompatibilityCheck() const {
    return _compilerForCompatibilityCheck;
}

ov::CompatibilityCheck PluginPropertyManager::validateCompatibilityDescriptor(ov::intel_npu::CompilerType compilerType,
                                                                              const ov::AnyMap& arguments) const {
    if (arguments.empty() || arguments.find(ov::runtime_requirements.name()) == arguments.end()) {
        return ov::CompatibilityCheck::NOT_APPLICABLE;
    }

    const auto& runtimeRequirements = arguments.at(ov::runtime_requirements.name()).as<const std::string&>();
    _logger.debug("Received runtime_requirements: %s length: %zu",
                  runtimeRequirements.c_str(),
                  runtimeRequirements.length());

    std::unique_ptr<MetadataBase> metadata = nullptr;
    try {
        metadata = read_as_text(runtimeRequirements);
    } catch (const std::exception& ex) {
        _logger.debug("Failed to read metadata from the runtime requirements. The requirements are not met. %s",
                      ex.what());
        return ov::CompatibilityCheck::UNSUPPORTED;
    }

    const auto descriptorView = metadata->get_compatibility_descriptor();
    std::string compatibilityDescriptor = descriptorView.has_value() ? std::string(descriptorView.value()) : "";
    _logger.debug("Retrieved compatibility descriptor from metadata: %s length: %zu",
                  compatibilityDescriptor.c_str(),
                  compatibilityDescriptor.length());

    std::unique_ptr<ICompilerAdapter> compiler = nullptr;
    CompilerAdapterFactory factory;
    try {
        compiler = factory.getCompiler(_backend, compilerType, std::string_view{});

        auto result = compiler->validate_compatibility_descriptor(compatibilityDescriptor);
        _logger.debug("Compatibility check result: %s", result ? "met" : "not met");
        return result ? ov::CompatibilityCheck::SUPPORTED : ov::CompatibilityCheck::UNSUPPORTED;
    } catch (const std::exception&) {
        _logger.error("Failed to create the recommended compiler type for the compatibility check %d. The requirements "
                      "are not met.",
                      static_cast<int>(compilerType));
        return ov::CompatibilityCheck::NOT_APPLICABLE;
    }
}

}  // namespace intel_npu
