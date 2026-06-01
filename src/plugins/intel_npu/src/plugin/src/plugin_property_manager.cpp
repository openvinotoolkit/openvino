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

static intel_npu::Config add_platform_to_the_config(intel_npu::Config config, const std::string_view platform) {
    config.update({{ov::intel_npu::platform.name(), std::string(platform)}});
    return config;
}

static auto get_specified_device_name(const intel_npu::Config& config) {
    if (config.has<intel_npu::DEVICE_ID>()) {
        return config.get<intel_npu::DEVICE_ID>();
    }
    return std::string();
}

static int64_t getOptimalNumberOfInferRequestsInParallel(const intel_npu::Config& config) {
    const std::string platform = config.get<intel_npu::PLATFORM>();

    if (platform == ov::intel_npu::Platform::NPU3720) {
        if (config.get<intel_npu::PERFORMANCE_HINT>() == ov::hint::PerformanceMode::THROUGHPUT) {
            return 4;
        }
        return 1;
    }

    if (config.get<intel_npu::PERFORMANCE_HINT>() == ov::hint::PerformanceMode::THROUGHPUT) {
        return 8;
    }
    return 1;
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

    REGISTER_CUSTOM_METRIC(_properties, ov::supported_properties, true, [&](const Config&) {
        return _supportedProperties;
    });

    for (auto& property : _properties) {
        if (property.second.isPublic) {
            _supportedProperties.emplace_back(ov::PropertyName(property.first, property.second.mutability));
        }
    }
}

void PluginPropertyManager::registerPluginProperties() const {
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::enable_profiling, PERF_COUNT);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::hint::performance_mode, PERFORMANCE_HINT);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::hint::execution_mode, EXECUTION_MODE_HINT);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::hint::num_requests, PERFORMANCE_HINT_NUM_REQUESTS);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::compilation_num_threads, COMPILATION_NUM_THREADS);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::hint::inference_precision, INFERENCE_PRECISION_HINT);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::log::level, LOG_LEVEL);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::cache_dir, CACHE_DIR);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::cache_mode, CACHE_MODE);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::hint::compiled_blob, COMPILED_BLOB);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::device::id, DEVICE_ID);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::num_streams, NUM_STREAMS);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::weights_path, WEIGHTS_PATH);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::compilation_mode_params, COMPILATION_MODE_PARAMS);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::dma_engines, DMA_ENGINES);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::tiles, TILES);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::compilation_mode, COMPILATION_MODE);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::compiler_type, COMPILER_TYPE);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::platform, PLATFORM);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::create_executor, CREATE_EXECUTOR);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::dynamic_shape_to_static, DYNAMIC_SHAPE_TO_STATIC);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::profiling_type, PROFILING_TYPE);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::backend_compilation_params, BACKEND_COMPILATION_PARAMS);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::batch_mode, BATCH_MODE);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::turbo, TURBO);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::hint::model_priority, MODEL_PRIORITY);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::bypass_umd_caching, BYPASS_UMD_CACHING);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::defer_weights_load, DEFER_WEIGHTS_LOAD);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::compiler_dynamic_quantization, COMPILER_DYNAMIC_QUANTIZATION);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::qdq_optimization, QDQ_OPTIMIZATION);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::qdq_optimization_aggressive, QDQ_OPTIMIZATION_AGGRESSIVE);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::disable_version_check, DISABLE_VERSION_CHECK);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::export_raw_blob, EXPORT_RAW_BLOB);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::import_raw_blob, IMPORT_RAW_BLOB);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::batch_compiler_mode_settings, BATCH_COMPILER_MODE_SETTINGS);
    OPENVINO_SUPPRESS_DEPRECATED_START
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::hint::enable_cpu_pinning, ENABLE_CPU_PINNING);
    OPENVINO_SUPPRESS_DEPRECATED_END
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::workload_type, WORKLOAD_TYPE);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::enable_weightless, ENABLE_WEIGHTLESS);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::separate_weights_version, SEPARATE_WEIGHTS_VERSION);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::model_serializer_version, MODEL_SERIALIZER_VERSION);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::enable_strides_for, ENABLE_STRIDES_FOR);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::disable_idle_memory_prunning, DISABLE_IDLE_MEMORY_PRUNING);
    TRY_REGISTER_SIMPLE_PROPERTY(_config, _properties, ov::intel_npu::shared_common_queue, SHARED_COMMON_QUEUE);

    TRY_REGISTER_CUSTOMFUNC_PROPERTY(_config, _properties, ov::intel_npu::stepping, STEPPING, [&](const Config& config) {
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
    TRY_REGISTER_CUSTOMFUNC_PROPERTY(_config, _properties, ov::intel_npu::max_tiles, MAX_TILES, [&](const Config& config) {
        if (!config.has<MAX_TILES>()) {
            try {
                const auto specifiedDeviceName = get_specified_device_name(config);
                return static_cast<int64_t>(_metrics->GetMaxTiles(specifiedDeviceName));
            } catch (...) {
                _logger.warning("Metrics GetMaxTiles failed to get value from device.");
            }
        }
        return config.get<MAX_TILES>();
    });

    TRY_REGISTER_VARPUB_PROPERTY(_config, _properties, ov::intel_npu::run_inferences_sequentially, RUN_INFERENCES_SEQUENTIALLY, [&] {
        if (_backend && _backend->getInitStructs()) {
            if (_backend->getInitStructs()->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 1)) {
                return true;
            }
        }
        return false;
    }());
    TRY_REGISTER_CUSTOM_PROPERTY(_config,
                                 _properties,
                                 ov::compatibility_check,
                                 COMPATIBILITY_CHECK,
                                 true,
                                 ov::PropertyMutability::RO,
                                 [](const Config&) {
                                     return false;
                                 });
    TRY_REGISTER_CUSTOM_PROPERTY(_config,
                                 _properties,
                                 ov::cache_encryption_callbacks,
                                 CACHE_ENCRYPTION_CALLBACKS,
                                 true,
                                 ov::PropertyMutability::WO,
                                 [](const Config&) {
                                     return ov::EncryptionCallbacks{nullptr, nullptr};
                                 });
    FORCE_REGISTER_CUSTOM_PROPERTY(_properties,
                                   ov::hint::model,
                                   MODEL_PTR,
                                   true,
                                   ov::PropertyMutability::RO,
                                   [](const Config&) {
                                       return std::shared_ptr<const ov::Model>(nullptr);
                                   });

    for_each_exposed_npuw_option([&](auto tag) {
        using Opt = typename decltype(tag)::type;
        TRY_REGISTER_NPUW_OPTION_PROPERTY(_config, _properties, Opt);
    });

    if (_metrics != nullptr) {
        REGISTER_SIMPLE_METRIC(_properties, ov::available_devices, true, _metrics->GetAvailableDevicesNames());
        REGISTER_SIMPLE_METRIC(_properties, ov::device::capabilities, true, _metrics->GetOptimizationCapabilities());
        REGISTER_SIMPLE_METRIC(_properties,
            ov::optimal_number_of_infer_requests,
            true,
            static_cast<uint32_t>(getOptimalNumberOfInferRequestsInParallel(add_platform_to_the_config(
                config,
                utils::getCompilationPlatform(
                    config.get<PLATFORM>(),
                    _backend == nullptr ? config.get<DEVICE_ID>()
                                        : _backend->getDevice(config.get<DEVICE_ID>())->getName(),
                    _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames())))));
        REGISTER_SIMPLE_METRIC(_properties, ov::range_for_async_infer_requests, true, _metrics->GetRangeForAsyncInferRequest());
        REGISTER_SIMPLE_METRIC(_properties, ov::range_for_streams, true, _metrics->GetRangeForStreams());
        REGISTER_SIMPLE_METRIC(_properties, ov::device::pci_info, true, _metrics->GetPciInfo(get_specified_device_name(config)));
        REGISTER_SIMPLE_METRIC(_properties, ov::device::gops, true, _metrics->GetGops(get_specified_device_name(config)));
        REGISTER_SIMPLE_METRIC(_properties, ov::device::type, true, _metrics->GetDeviceType(get_specified_device_name(config)));
        REGISTER_CUSTOM_METRIC(_properties, ov::internal::supported_properties, false, [&](const Config&) {
            return _internalSupportedProperties;
        });
        REGISTER_SIMPLE_METRIC(_properties, ov::internal::cache_header_alignment, false, utils::STANDARD_PAGE_SIZE);
        REGISTER_SIMPLE_METRIC(_properties, ov::intel_npu::device_alloc_mem_size,
                               true,
                               _metrics->GetDeviceAllocMemSize(get_specified_device_name(config)));
        REGISTER_SIMPLE_METRIC(_properties, ov::intel_npu::device_total_mem_size,
                               true,
                               _metrics->GetDeviceTotalMemSize(get_specified_device_name(config)));
        REGISTER_SIMPLE_METRIC(_properties, ov::intel_npu::driver_version, true, _metrics->GetDriverVersion());
        REGISTER_SIMPLE_METRIC(_properties, ov::intel_npu::backend_name, false, _metrics->GetBackendName());
        REGISTER_CUSTOM_METRIC(_properties, ov::device::architecture,
                               !_metrics->GetAvailableDevicesNames().empty(),
                               [&](const Config& config) {
                                   const auto specifiedDeviceName = get_specified_device_name(config);
                                   return _metrics->GetDeviceArchitecture(specifiedDeviceName);
                               });
        REGISTER_CUSTOM_METRIC(_properties, ov::device::full_name,
                               !_metrics->GetAvailableDevicesNames().empty(),
                               [&](const Config& config) {
                                   const auto specifiedDeviceName = get_specified_device_name(config);
                                   return _metrics->GetFullDeviceName(specifiedDeviceName);
                               });
        REGISTER_CUSTOM_METRIC(_properties, ov::device::luid,
                               _backend == nullptr ? false : _backend->isLUIDExtSupported(),
                               [&](const Config& config) {
                                   const auto specifiedDeviceName = get_specified_device_name(config);
                                   return _metrics->GetDeviceLUID(specifiedDeviceName);
                               });
        REGISTER_CUSTOM_METRIC(_properties, ov::device::uuid, true, [&](const Config& config) {
            const auto specifiedDeviceName = get_specified_device_name(config);
            auto devUuid = _metrics->GetDeviceUuid(specifiedDeviceName);
            return decltype(ov::device::uuid)::value_type{devUuid};
        });
        REGISTER_CUSTOM_METRIC(_properties, ov::execution_devices, true, [&](const Config& config) {
            if (_metrics->GetAvailableDevicesNames().size() > 1) {
                return std::string("NPU." + config.get<DEVICE_ID>());
            }
            return std::string("NPU");
        });
        REGISTER_CUSTOM_METRIC(_properties, ov::intel_npu::compiler_version, true, [&](const Config& config) {
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
        REGISTER_CUSTOM_METRIC(_properties, ov::internal::caching_properties, false, [&](const Config&) {
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
        if (name == ov::compatibility_check.name()) {
            return validateCompatibilityDescriptor(determineCompilerTypeForCompatibilityCheck(), arguments);
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

    auto compilationPlatform = utils::getCompilationPlatform(
        _config.get<PLATFORM>(),
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
                logger.info("Config key '%s' is recognized as a compiler option, will not be used for current configuration.",
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
    auto [updatedConfig, compilerConfigsFilteredByCompiler, currentlyUsedCompiler, currentlyUsedPlatform, logger] = [&]() {
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

ov::CompatibilityCheck PluginPropertyManager::validateCompatibilityDescriptor(
    ov::intel_npu::CompilerType compilerType,
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