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
        // Special case for some both configs. Don't need compiler for these Both properties.
        // Runtime (plugin-only) options are always enabled
        if (opt.mode() != OptionMode::RunTime && !isSpecialBothProperty(key)) {
            if (legacy) {
                // Compiler or common option in Legacy mode? Checking its supported version
                if (compilerVersion >= opt.compilerSupportVersion()) {
                    isEnabled = true;
                }
            } else {
                // We have compiler, we are not in legacy mode = we have a valid list of supported options
                // Searching in the list
                const auto& supportedOptions = compilerSupportList.value();
                auto it = std::find(supportedOptions.begin(), supportedOptions.end(), key);
                if (it != supportedOptions.end()) {
                    isEnabled = true;
                } else {
                    // Not found in the supported options list.
                    if (compiler != nullptr) {
                        // Checking if it is a private option?
                        isEnabled = compiler->is_option_supported(key);
                    } else {
                        // Not in the list and not a private option = disabling
                        isEnabled = false;
                    }
                }
            }

            if (!isEnabled) {
                logger.debug("Config option %s not supported! Requirements not met.", key.c_str());
            } else {
                logger.debug("Enabled config option %s", key.c_str());
            }

            // update enable flag
            config.enable(key, isEnabled);
        }
    });

    // Special cases
    // NPU_TURBO which might not be supported by compiler, but driver will still use it
    // if it exists in config = driver supports it
    // if compiler->is_option_suported is false = compiler doesn't support it and gets marked disabled by default logic
    // however, if driver supports it, we still need it (and will skip giving it to compiler) = force-enable
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
        // Special case for some both configs. Don't need compiler for these Both properties.
        // Runtime (plugin-only) options are always enabled
        if (opt.mode() != OptionMode::RunTime && !isSpecialBothProperty(key)) {
            // Disable all compiler options
            config.enable(key, false);
        }
    });

    // Special cases
    // NPU_TURBO might be supported by the driver
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

void exclude_model_ptr_from_map(ov::AnyMap& properties) {
    if (properties.count(ov::hint::model.name())) {
        properties.erase(ov::hint::model.name());
    }
}

bool isCompatibilityCheckSupported(const ov::SoPtr<intel_npu::IEngineBackend>& backend) {
    using namespace intel_npu;

    if (!backend || !backend->getDevice()) {
        return false;
    }

    const auto initStructs = backend->getInitStructs();
    if (initStructs != nullptr && initStructs->getZeDrvApiVersion() >= ZE_MAKE_VERSION(1, 16)) {
        return true;
    }

    // Fallback to plugin compiler if the driver does not expose compatibility check API.
    CompilerAdapterFactory compilerFactory;
    auto compilerType = ov::intel_npu::CompilerType::PLUGIN;
    try {
        auto tempCompiler = compilerFactory.getCompiler(backend, compilerType, std::string_view{});
        return tempCompiler->is_option_supported(ov::compatibility_check.name());
    } catch (...) {
        return false;
    }
}

ov::CompatibilityCheck validateCompatibilityDescriptor(const ov::SoPtr<intel_npu::IEngineBackend>& backend,
                                                       const ov::AnyMap& arguments) {
    using namespace intel_npu;

    if (arguments.empty() || arguments.find(ov::runtime_requirements.name()) == arguments.end()) {
        return ov::CompatibilityCheck::NOT_APPLICABLE;
    }

    const auto& runtimeRequirements = arguments.at(ov::runtime_requirements.name()).as<const std::string&>();

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

    // fallback on compiler in plugin if driver does not support compatibility check
    CompilerAdapterFactory factory;
    auto compilerType = ov::intel_npu::CompilerType::PLUGIN;
    try {
        auto compiler = factory.getCompiler(backend, compilerType, std::string_view{});

        auto result =
            compiler->is_option_supported(ov::compatibility_check.name(), std::make_optional(compatibilityDescriptor));
        return result ? ov::CompatibilityCheck::SUPPORTED : ov::CompatibilityCheck::UNSUPPORTED;
    } catch (...) {
        return ov::CompatibilityCheck::NOT_APPLICABLE;
    }
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
    : PluginPropertyManager([&other]() {
          std::lock_guard<std::mutex> lock(other._mutex);
          return CopyState{other._config,
                           other._metrics,
                           other._backend,
                           other._logger,
                           other._currentlyUsedCompiler,
                           other._compilerForCompatibilityCheck,
                           other._compatibilityCheckSupported,
                           other._currentlyUsedPlatform,
                           other._compilerConfigsFilteredByCompiler,
                           other._compatibilityCheckFiltered};
      }()) {}

PluginPropertyManager::PluginPropertyManager(CopyState&& state)
    : _config(std::move(state.config)),
      _metrics(std::move(state.metrics)),
      _backend(std::move(state.backend)),
      _logger(state.logger),
      _currentlyUsedCompiler(state.currentlyUsedCompiler),
      _compilerForCompatibilityCheck(state._compilerForCompatibilityCheck),
      _compatibilityCheckSupported(state.compatibilityCheckSupported),
      _currentlyUsedPlatform(std::move(state.currentlyUsedPlatform)),
      _compilerConfigsFilteredByCompiler(state.compilerConfigsFilteredByCompiler),
      _compatibilityCheckFiltered(state.compatibilityCheckFiltered) {
    registerProperties();
}

void PluginPropertyManager::registerProperties() const {
    _properties.clear();
    _supportedProperties.clear();

    // clang-format off
    try_register_property_based_on_config<PERF_COUNT>(_config, _properties, ov::enable_profiling.name());
    try_register_property_based_on_config<PERFORMANCE_HINT>(_config, _properties, ov::hint::performance_mode.name());
    try_register_property_based_on_config<EXECUTION_MODE_HINT>(_config, _properties, ov::hint::execution_mode.name());
    try_register_property_based_on_config<PERFORMANCE_HINT_NUM_REQUESTS>(_config, _properties, ov::hint::num_requests.name());
    try_register_property_based_on_config<COMPILATION_NUM_THREADS>(_config, _properties, ov::compilation_num_threads.name());
    try_register_property_based_on_config<INFERENCE_PRECISION_HINT>(_config, _properties, ov::hint::inference_precision.name());
    try_register_property_based_on_config<LOG_LEVEL>(_config, _properties, ov::log::level.name());
    try_register_property_based_on_config<CACHE_DIR>(_config, _properties, ov::cache_dir.name());
    try_register_property_based_on_config<CACHE_MODE>(_config, _properties, ov::cache_mode.name());
    try_register_property_based_on_config<COMPILED_BLOB>(_config, _properties, ov::hint::compiled_blob.name());
    try_register_property_based_on_config<DEVICE_ID>(_config, _properties, ov::device::id.name());
    try_register_property_based_on_config<NUM_STREAMS>(_config, _properties, ov::num_streams.name());
    try_register_property_based_on_config<WEIGHTS_PATH>(_config, _properties, ov::weights_path.name());
    try_register_property_based_on_config<COMPILATION_MODE_PARAMS>(_config, _properties, ov::intel_npu::compilation_mode_params.name());
    try_register_property_based_on_config<DMA_ENGINES>(_config, _properties, ov::intel_npu::dma_engines.name());
    try_register_property_based_on_config<TILES>(_config, _properties, ov::intel_npu::tiles.name());
    try_register_property_based_on_config<COMPILATION_MODE>(_config, _properties, ov::intel_npu::compilation_mode.name());
    try_register_property_based_on_config<COMPILER_TYPE>(_config, _properties, ov::intel_npu::compiler_type.name());
    try_register_property_based_on_config<PLATFORM>(_config, _properties, ov::intel_npu::platform.name());
    try_register_property_based_on_config<CREATE_EXECUTOR>(_config, _properties, ov::intel_npu::create_executor.name());
    try_register_property_based_on_config<DYNAMIC_SHAPE_TO_STATIC>(_config, _properties, ov::intel_npu::dynamic_shape_to_static.name());
    try_register_property_based_on_config<PROFILING_TYPE>(_config, _properties, ov::intel_npu::profiling_type.name());
    try_register_property_based_on_config<BACKEND_COMPILATION_PARAMS>(_config, _properties, ov::intel_npu::backend_compilation_params.name());
    try_register_property_based_on_config<BATCH_MODE>(_config, _properties, ov::intel_npu::batch_mode.name());
    try_register_property_based_on_config<TURBO>(_config, _properties, ov::intel_npu::turbo.name());
    try_register_property_based_on_config<MODEL_PRIORITY>(_config, _properties, ov::hint::model_priority.name());
    try_register_property_based_on_config<BYPASS_UMD_CACHING>(_config, _properties, ov::intel_npu::bypass_umd_caching.name());
    try_register_property_based_on_config<DEFER_WEIGHTS_LOAD>(_config, _properties, ov::intel_npu::defer_weights_load.name());
    try_register_property_based_on_config<COMPILER_DYNAMIC_QUANTIZATION>(_config, _properties, ov::intel_npu::compiler_dynamic_quantization.name());
    try_register_property_based_on_config<QDQ_OPTIMIZATION>(_config, _properties, ov::intel_npu::qdq_optimization.name());
    try_register_property_based_on_config<QDQ_OPTIMIZATION_AGGRESSIVE>(_config, _properties, ov::intel_npu::qdq_optimization_aggressive.name());
    try_register_property_based_on_config<DISABLE_VERSION_CHECK>(_config, _properties, ov::intel_npu::disable_version_check.name());
    try_register_property_based_on_config<EXPORT_RAW_BLOB>(_config, _properties, ov::intel_npu::export_raw_blob.name());
    try_register_property_based_on_config<IMPORT_RAW_BLOB>(_config, _properties, ov::intel_npu::import_raw_blob.name());
    try_register_property_based_on_config<BATCH_COMPILER_MODE_SETTINGS>(_config, _properties, ov::intel_npu::batch_compiler_mode_settings.name());
    OPENVINO_SUPPRESS_DEPRECATED_START
    try_register_property_based_on_config<ENABLE_CPU_PINNING>(_config, _properties, ov::hint::enable_cpu_pinning.name());
    OPENVINO_SUPPRESS_DEPRECATED_END
    try_register_property_based_on_config<WORKLOAD_TYPE>(_config, _properties, ov::workload_type.name());
    try_register_property_based_on_config<ENABLE_WEIGHTLESS>(_config, _properties, ov::enable_weightless.name());
    try_register_property_based_on_config<SEPARATE_WEIGHTS_VERSION>(_config, _properties, ov::intel_npu::separate_weights_version.name());
    try_register_property_based_on_config<MODEL_SERIALIZER_VERSION>(_config, _properties, ov::intel_npu::model_serializer_version.name());
    try_register_property_based_on_config<ENABLE_STRIDES_FOR>(_config, _properties, ov::intel_npu::enable_strides_for.name());
    try_register_property_based_on_config<DISABLE_IDLE_MEMORY_PRUNING>(_config, _properties, ov::intel_npu::disable_idle_memory_prunning.name());
    try_register_property_based_on_config<SHARED_COMMON_QUEUE>(_config, _properties, ov::intel_npu::shared_common_queue.name());

    try_register_property_based_on_config_with_custom_function(_config, _properties, ov::intel_npu::stepping.name(), [&](const Config& config) {
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
    try_register_property_based_on_config_with_custom_function(_config, _properties, ov::intel_npu::max_tiles.name(), [&](const Config& config) {
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
    try_register_property_based_on_config_with_custom_function(_config, _properties, ov::cache_encryption_callbacks.name(), [](const Config&) {
        return ov::EncryptionCallbacks{nullptr, nullptr};
    });
    // clang-format on

    // Special case: this property is always registered because it's supported by the implementation,
    // but it's not visible in supported_properties if the driver doesn't support it.
    try_register_property_based_on_config_with_visibility<RUN_INFERENCES_SEQUENTIALLY>(
        _config,
        _properties,
        ov::intel_npu::run_inferences_sequentially.name(),
        [&] {
            if (_backend && _backend->getInitStructs()) {
                if (_backend->getInitStructs()->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 1)) {
                    return true;
                }
            }
            return false;
        }());

    for_each_exposed_npuw_option([&](auto tag) {
        using Opt = typename decltype(tag)::type;
        try_register_npuw_option_property<Opt>(_config, _properties);
    });

    // clang-format off
    if (_metrics != nullptr) {
        register_property_with_custom_function(_properties, ov::available_devices.name(), true, [&](const Config&) {
            return _metrics->GetAvailableDevicesNames();
        });
        register_property_with_custom_function(_properties, ov::device::capabilities.name(), true, [&](const Config&) {
            return _metrics->GetOptimizationCapabilities();
        });
        register_property_with_custom_function(_properties, ov::range_for_async_infer_requests.name(), true, [&](const Config&) {
            return _metrics->GetRangeForAsyncInferRequest();
        });
        register_property_with_custom_function(_properties, ov::range_for_streams.name(), true, [&](const Config&) {
            return _metrics->GetRangeForStreams();
        });
        register_property_with_custom_function(_properties, ov::device::pci_info.name(), true, [&](const Config& config) {
            return _metrics->GetPciInfo(get_specified_device_name(config));
        });
        register_property_with_custom_function(_properties, ov::device::gops.name(), true, [&](const Config& config) {
            return _metrics->GetGops(get_specified_device_name(config));
        });
        register_property_with_custom_function(_properties, ov::device::type.name(), true, [&](const Config& config) {
            return _metrics->GetDeviceType(get_specified_device_name(config));
        });
        register_property_with_custom_function(_properties, ov::intel_npu::device_alloc_mem_size.name(), true, [&](const Config& config) {
            return _metrics->GetDeviceAllocMemSize(get_specified_device_name(config));
        });
        register_property_with_custom_function(_properties, ov::intel_npu::device_total_mem_size.name(), true, [&](const Config& config) {
            return _metrics->GetDeviceTotalMemSize(get_specified_device_name(config));
        });
        register_property_with_custom_function(_properties, ov::intel_npu::driver_version.name(), true, [&](const Config&) {
            return _metrics->GetDriverVersion();
        });
        register_property_with_custom_function(_properties, ov::device::uuid.name(), true, [&](const Config& config) {
            auto devUuid = _metrics->GetDeviceUuid(get_specified_device_name(config));
            return decltype(ov::device::uuid)::value_type{devUuid};
        });
        register_property_with_custom_function(_properties, ov::device::luid.name(), _backend == nullptr ? false : _backend->isLUIDExtSupported(), [&](const Config& config) {
            return _metrics->GetDeviceLUID(get_specified_device_name(config));
        });
        register_property_with_custom_function(_properties, ov::execution_devices.name(), true, [&](const Config& config) {
            if (_metrics->GetAvailableDevicesNames().size() > 1) {
                return std::string("NPU." + config.get<DEVICE_ID>());
            }
            return std::string("NPU");
        });
        register_property_with_custom_function(_properties, ov::intel_npu::backend_name.name(), false, [&](const Config&) {
            return _metrics->GetBackendName();
        });

        try_register_property_with_custom_function(_properties, ov::device::architecture.name(), !_metrics->GetAvailableDevicesNames().empty(), true, [&](const Config& config) {
            return _metrics->GetDeviceArchitecture(get_specified_device_name(config));
        });
        try_register_property_with_custom_function(_properties, ov::device::full_name.name(), !_metrics->GetAvailableDevicesNames().empty(), true, [&](const Config& config) {
            return _metrics->GetFullDeviceName(get_specified_device_name(config));
        });
    }

    register_property_with_custom_function(_properties, ov::hint::model.name(), true, [](const Config&) {
        return std::shared_ptr<const ov::Model>(nullptr);
    });
    register_property_with_custom_function(_properties, ov::optimal_number_of_infer_requests.name(), true, [&](const Config& config) {
        return utils::getOptimalNumberOfInferRequestsInParallel(
            utils::getCompilationPlatform(
                config.get<PLATFORM>(),
                _backend == nullptr ? config.get<DEVICE_ID>()
                                    : _backend->getDevice(config.get<DEVICE_ID>())->getName(),
                _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames()),
            config.get<PERFORMANCE_HINT>());
    });
    register_property_with_custom_function(_properties, ov::intel_npu::compiler_version.name(), true, [&](const Config& config) {
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
    register_property_with_custom_function(_properties, ov::internal::supported_properties.name(), false, [&](const Config&) {
        return _internalSupportedProperties;
    });
    register_property_with_custom_function(_properties, ov::internal::cache_header_alignment.name(), false, [&](const Config&) {
        return utils::STANDARD_PAGE_SIZE;
    });
    register_property_with_custom_function(_properties, ov::internal::caching_properties.name(), false, [&](const Config&) {
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
    register_property_with_custom_function(_properties, ov::supported_properties.name(), true, [&](const Config&) {
        return _supportedProperties;
    });
    // clang-format on

    try_register_property_with_custom_function_and_args(_properties,
                                                        ov::compatibility_check.name(),
                                                        _compatibilityCheckFiltered && _compatibilityCheckSupported,
                                                        true,
                                                        [this](const Config&, const ov::AnyMap& arguments) {
                                                            return validateCompatibilityDescriptor(_backend, arguments);
                                                        });

    for (auto& property : _properties) {
        if (property.second.isPublic) {
            _supportedProperties.emplace_back(ov::PropertyName(property.first, property.second.mutability));
        }
    }
}

void PluginPropertyManager::initializeCompatibilityCheckSupportIfNeeded() const {
    if (_compatibilityCheckFiltered) {
        return;
    }

    _compatibilityCheckSupported = isCompatibilityCheckSupported(_backend);
    _compatibilityCheckFiltered = true;

    const auto compatibilityCheckName = std::string(ov::compatibility_check.name());

    // Keep only one descriptor for this property and update it after the one-time probe.
    _properties.erase(compatibilityCheckName);

    try_register_property_with_custom_function_and_args(_properties,
                                                        compatibilityCheckName,
                                                        _compatibilityCheckSupported,
                                                        true,
                                                        [this](const Config&, const ov::AnyMap& arguments) {
                                                            return validateCompatibilityDescriptor(_backend, arguments);
                                                        });

    // Update supported_properties incrementally for compatibility_check only.
    _supportedProperties.erase(
        std::remove(_supportedProperties.begin(), _supportedProperties.end(), compatibilityCheckName),
        _supportedProperties.end());
    if (_compatibilityCheckSupported) {
        _supportedProperties.emplace_back(ov::PropertyName(compatibilityCheckName, ov::PropertyMutability::RO));
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

    if (properties.find(ov::compatibility_check.name()) != properties.end()) {
        initializeCompatibilityCheckSupportIfNeeded();
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
        // Special case for some both configs. Don't need to check compiler support for these Both properties.
        const bool isNotSpecialBothProperty = !isSpecialBothProperty(property.first);
        if (_config.hasOpt(property.first) && isNotSpecialBothProperty) {
            auto opt = _config.getOpt(property.first);
            if (opt.mode() != OptionMode::RunTime) {
                propertyIsCompilerConfig = true;
                break;
            }
        }
    }

    // Check if one of the properties is compiler config which needs to return different values based on compiler
    // and platform configuration
    if (propertyIsCompilerConfig || !propertyIsRegistered) {
        auto compilerType = resolveCompilerTypeWithoutLock(properties);
        auto deviceId = resolveDeviceIdWithoutLock(properties);
        auto device = utils::getDeviceById(_backend, deviceId);

        auto compilationPlatform = utils::getCompilationPlatform(
            resolvePlatformWithoutLock(properties),
            device == nullptr ? std::move(deviceId) : device->getName(),
            _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());

        // Create a compiler to get the type and fetch version and supported options if needed
        CompilerAdapterFactory factory;
        compiler = factory.getCompiler(_backend, compilerType, compilationPlatform);

        if (!(_compilerConfigsFilteredByCompiler && compilerType == _currentlyUsedCompiler &&
              compilationPlatform == _currentlyUsedPlatform)) {
            // In case properties are not initialized or the compiler/platform was changed since last call -
            // filter out options again
            filterPropertiesByCompilerSupport(_config, compiler.get(), _backend, _logger);

            // reset properties for the new options
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
            // property doesn't exist - checking as internal now
            if (compiler != nullptr) {
                if (compiler->is_option_supported(value.first)) {
                    // if compiler reports it supported > registering as internal
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

    if (name == ov::supported_properties.name() || name == ov::compatibility_check.name()) {
        initializeCompatibilityCheckSupportIfNeeded();
    }

    bool propertyIsCompilerConfig = false;
    bool propertyIsRegistered = true;
    // If the property is not registered, there is no point of checking the config.
    if (!isPropertyRegistered(name)) {
        propertyIsRegistered = false;
    } else if (_config.hasOpt(name) && !isSpecialBothProperty(name)) {
        // Property is already registered but need to re-check if the CompilerTime config is still supported by the
        // current compiler.
        auto opt = _config.getOpt(name);
        if (opt.mode() != OptionMode::RunTime) {
            propertyIsCompilerConfig = true;
        }
    }

    // Special case for Supported Properties and Caching Properties as they are compiler dependent. So we need to
    // check compiler support for those properties on each getProperty call as well.
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

        // Create a compiler to get the type and fetch version and supported options if needed
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
            // In case properties are not initialized or the compiler/platform was changed since last call -
            // filter out options again
            filterPropertiesByCompilerSupport(_config, compiler.get(), _backend, _logger);

            // reset properties for the new options
            registerProperties();
            _compilerConfigsFilteredByCompiler = true;
            _currentlyUsedCompiler = compilerType;
            _currentlyUsedPlatform = std::move(compilationPlatform);
        }
    }

    auto&& configIterator = _properties.find(name);
    if (configIterator != _properties.cend()) {
        if (configIterator->second.mutability == ov::PropertyMutability::WO) {
            _logger.warning("Trying to get WRITE-ONLY property: %s. Returning empty `ov::Any` object", name.c_str());
            return ov::Any();
        }
        // Check if property requires arguments
        if (configIterator->second.getWithArgs) {
            return configIterator->second.getWithArgs(_config, arguments);
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
    if (!arguments.empty() && name != ov::compatibility_check.name()) {
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

    if (name == ov::compatibility_check.name()) {
        initializeCompatibilityCheckSupportIfNeeded();
    }

    const bool isRegistered = isPropertyRegistered(name);
    const bool isConfigOption = _config.hasOpt(name);

    if (!isRegistered && !isConfigOption) {
        // Property is neither registered nor known by config
        return false;
    }

    if (isRegistered) {
        const auto it = _properties.find(name);
        return it->second.isPublic;
    }

    // Property is compiler config, need to check compiler support
    std::unique_ptr<ICompilerAdapter> compiler = nullptr;
    auto compilerType = _config.get<COMPILER_TYPE>();
    auto deviceId = _config.get<DEVICE_ID>();
    auto device = utils::getDeviceById(_backend, deviceId);

    auto compilationPlatform =
        utils::getCompilationPlatform(_config.get<PLATFORM>(),
                                      device == nullptr ? std::move(deviceId) : device->getName(),
                                      _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());

    // Create a compiler to get the type and fetch version and supported options if needed
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
        // In case properties are not initialized or the compiler/platform was changed since last call -
        // filter out options again
        filterPropertiesByCompilerSupport(_config, compiler.get(), _backend, _logger);

        // reset properties for the new options
        registerProperties();
        _compilerConfigsFilteredByCompiler = true;
        _currentlyUsedCompiler = compilerType;
        _currentlyUsedPlatform = std::move(compilationPlatform);
    }

    const auto it = _properties.find(name);
    return it != _properties.end() && it->second.isPublic;
}

FilteredConfig PluginPropertyManager::getConfigWithCompilerPropertiesDisabled(const ov::AnyMap& properties) const {
    auto [updatedConfig, compilerConfigsFilteredByCompiler, logger] = [&]() {
        std::lock_guard<std::mutex> lock(_mutex);
        return std::make_tuple(_config, _compilerConfigsFilteredByCompiler, _logger);
    }();

    auto pluginProperties = properties;
    exclude_model_ptr_from_map(pluginProperties);

    if (compilerConfigsFilteredByCompiler) {
        disableCompilerProperties(updatedConfig, _backend);
    }

    if (pluginProperties.find(ov::hint::enable_cpu_pinning.name()) != pluginProperties.end()) {
        logCpuPinningDeprecationWarning(logger);
    }

    if (pluginProperties.empty()) {
        return std::move(updatedConfig);
    }

    const std::map<std::string, std::string> rawConfig = any_copy(pluginProperties);
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
            specialCfgsToSet.emplace(key, pluginProperties.at(key));
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

    auto pluginProperties = properties;
    exclude_model_ptr_from_map(pluginProperties);

    std::optional<ov::intel_npu::CompilerType> propertiesCompilerType = std::nullopt;
    std::optional<std::string> propertiesPlatform = std::nullopt;
    if (compilerConfigsFilteredByCompiler) {
        auto compilerType = pluginProperties.find(ov::intel_npu::compiler_type.name());
        if (compilerType != pluginProperties.end()) {
            propertiesCompilerType = compilerType->second.as<ov::intel_npu::CompilerType>();
        }
    }
    auto platform = pluginProperties.find(ov::intel_npu::platform.name());
    if (platform != pluginProperties.end()) {
        propertiesPlatform = platform->second.as<std::string>();
    }

    // filter out unsupported options
    if (!(compilerConfigsFilteredByCompiler &&
          propertiesCompilerType.value_or(currentlyUsedCompiler) == currentlyUsedCompiler &&
          propertiesPlatform.value_or(currentlyUsedPlatform) == currentlyUsedPlatform)) {
        // In case the compiler properties are not initialized or the compiler/platform was changed since last call -
        // filter out options again
        filterPropertiesByCompilerSupport(updatedConfig, compiler, _backend, logger);
    }

    const std::map<std::string, std::string> rawConfig = any_copy(pluginProperties);
    std::map<std::string, std::string> cfgsToSet;
    ov::AnyMap specialCfgsToSet;
    for (const auto& [key, value] : rawConfig) {
        if (!updatedConfig.hasOpt(key)) {
            // not a known config key
            if (!compiler->is_option_supported(key)) {
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

}  // namespace intel_npu
