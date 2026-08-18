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

inline bool isSpecialBothProperty(const std::string& key) {
    return key == ov::hint::performance_mode.name() || key == ov::enable_profiling.name() ||
           key == ov::log::level.name();
}

inline void logCpuPinningDeprecationWarning(intel_npu::Logger& logger) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    logger.warning(intel_npu::ENABLE_CPU_PINNING::deprecationMessage());
    OPENVINO_SUPPRESS_DEPRECATED_END
}

void exclude_model_ptr_from_map(ov::AnyMap& properties) {
    if (properties.count(ov::hint::model.name())) {
        properties.erase(ov::hint::model.name());
    }
}

std::vector<std::string> getAvailableDevicesNames(const ov::SoPtr<intel_npu::IEngineBackend>& backend) {
    return backend == nullptr ? std::vector<std::string>() : backend->getDeviceNames();
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
                                                     ov::compatibility_check.name(),
                                                     std::nullopt);
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

PluginPropertyManager::PluginPropertyManager(const PluginPropertyManager& other)
    : PluginPropertyManager([&other]() {
          std::lock_guard<std::mutex> lock(other._mutex);
          return CopyState{other._config,
                           other._backend,
                           other._compilerOptionSupportHelper,
                           other._logger,
                           other._currentlyUsedCompiler,
                           other._currentlyUsedPlatform,
                           other._compilerConfigsFilteredByCompiler};
      }()) {}

PluginPropertyManager::PluginPropertyManager(CopyState&& state)
    : _config(std::move(state.config)),
      _backend(std::move(state.backend)),
      _compilerOptionSupportHelper(std::move(state.optionSupportHelper)),
      _logger(state.logger),
      _currentlyUsedCompiler(state.currentlyUsedCompiler),
      _currentlyUsedPlatform(std::move(state.currentlyUsedPlatform)),
      _compilerConfigsFilteredByCompiler(state.compilerConfigsFilteredByCompiler) {
    registerProperties();
}

void PluginPropertyManager::refreshCompilerPropertiesIfNeeded(ov::intel_npu::CompilerType compilerType,
                                                              std::string compilationPlatform) {
    if (_compilerConfigsFilteredByCompiler && compilerType == _currentlyUsedCompiler &&
        compilationPlatform == _currentlyUsedPlatform) {
        return;
    }

    _compilerConfigsFilteredByCompiler = true;
    _currentlyUsedCompiler = compilerType;
    _currentlyUsedPlatform = std::move(compilationPlatform);
}

void PluginPropertyManager::registerProperties() {
    _properties.clear();

    const bool hasBackend = _backend != nullptr;
    const auto hasBackendPredicate = [hasBackend]() {
        return hasBackend;
    };

    using DeviceValidationCache = std::optional<std::pair<std::string, bool>>;
    auto hasBackendAndValidDeviceCache = std::make_shared<DeviceValidationCache>();
    const auto hasBackendAndValidDevice = [this, hasBackend, hasBackendAndValidDeviceCache]() {
        if (!hasBackend) {
            return false;
        }

        try {
            const auto specifiedDeviceName = _config.get<intel_npu::DEVICE_ID>();

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

    const auto isCompilerOptionSupported = [this](std::string_view propertyName) {
        try {
            return _compilerOptionSupportHelper->isOptionSupported(_currentlyUsedCompiler, std::string(propertyName));
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

    register_property_with_support<CACHE_MODE>(_config, _properties, true, ov::PropertyMutability::RW, [isCompilerOptionSupported] {
        return isCompilerOptionSupported(CACHE_MODE::key());
    });
    register_property_with_support<COMPILATION_MODE_PARAMS>(_config, _properties, true, ov::PropertyMutability::RW, [isCompilerOptionSupported] {
        return isCompilerOptionSupported(COMPILATION_MODE_PARAMS::key());
    });
    register_property_with_support<COMPILATION_NUM_THREADS>(_config, _properties, true, ov::PropertyMutability::RW, [isCompilerOptionSupported] {
        return isCompilerOptionSupported(COMPILATION_NUM_THREADS::key());
    });
    register_property_with_support<COMPILER_DYNAMIC_QUANTIZATION>(_config, _properties, true, ov::PropertyMutability::RW, [isCompilerOptionSupported] {
        return isCompilerOptionSupported(COMPILER_DYNAMIC_QUANTIZATION::key());
    });
    register_property_with_support<EXECUTION_MODE_HINT>(_config, _properties, true, ov::PropertyMutability::RW, [isCompilerOptionSupported] {
        return isCompilerOptionSupported(EXECUTION_MODE_HINT::key());
    });
    register_property_with_support<INFERENCE_PRECISION_HINT>(_config, _properties, true, ov::PropertyMutability::RW, [isCompilerOptionSupported] {
        return isCompilerOptionSupported(INFERENCE_PRECISION_HINT::key());
    });
    register_property_with_support<PLATFORM>(_config, _properties, true, ov::PropertyMutability::RW, [isCompilerOptionSupported] {
        return isCompilerOptionSupported(PLATFORM::key());
    });
    register_property_with_support<QDQ_OPTIMIZATION>(_config, _properties, true, ov::PropertyMutability::RW, [isCompilerOptionSupported] {
        return isCompilerOptionSupported(QDQ_OPTIMIZATION::key());
    });
    register_property_with_support<QDQ_OPTIMIZATION_AGGRESSIVE>(_config, _properties, true, ov::PropertyMutability::RW, [isCompilerOptionSupported] {
        return isCompilerOptionSupported(QDQ_OPTIMIZATION_AGGRESSIVE::key());
    });
    register_property_with_support<TILES>(_config, _properties, true, ov::PropertyMutability::RW, [isCompilerOptionSupported] {
        return isCompilerOptionSupported(TILES::key());
    });

    register_property_with_support<BACKEND_COMPILATION_PARAMS>(_config, _properties, false, ov::PropertyMutability::RW, [isCompilerOptionSupported] {
        return isCompilerOptionSupported(BACKEND_COMPILATION_PARAMS::key());
    });
    register_property_with_support<BATCH_COMPILER_MODE_SETTINGS>(_config, _properties, false, ov::PropertyMutability::RW, [isCompilerOptionSupported] {
        return isCompilerOptionSupported(BATCH_COMPILER_MODE_SETTINGS::key());
    });
    register_property_with_support<BATCH_MODE>(_config, _properties, false, ov::PropertyMutability::RW, [isCompilerOptionSupported] {
        return isCompilerOptionSupported(BATCH_MODE::key());
    });
    register_property_with_support<COMPILATION_MODE>(_config, _properties, false, ov::PropertyMutability::RW, [isCompilerOptionSupported] {
        return isCompilerOptionSupported(COMPILATION_MODE::key());
    });
    register_property_with_support<DMA_ENGINES>(_config, _properties, false, ov::PropertyMutability::RW, [isCompilerOptionSupported] {
        return isCompilerOptionSupported(DMA_ENGINES::key());
    });
    register_property_with_support<DYNAMIC_SHAPE_TO_STATIC>(_config, _properties, false, ov::PropertyMutability::RW, [isCompilerOptionSupported] {
        return isCompilerOptionSupported(DYNAMIC_SHAPE_TO_STATIC::key());
    });
    register_property_with_support<ENABLE_WEIGHTLESS>(_config, _properties, false, ov::PropertyMutability::RW, [isCompilerOptionSupported] {
        return isCompilerOptionSupported(ENABLE_WEIGHTLESS::key());
    });
    register_property_with_support<MODEL_SERIALIZER_VERSION>(_config, _properties, false, ov::PropertyMutability::RW, [isCompilerOptionSupported] {
        return isCompilerOptionSupported(MODEL_SERIALIZER_VERSION::key());
    });
    register_property_with_support<SEPARATE_WEIGHTS_VERSION>(_config, _properties, false, ov::PropertyMutability::RW, [isCompilerOptionSupported] {
        return isCompilerOptionSupported(SEPARATE_WEIGHTS_VERSION::key());
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
            if (property.second.isPublic && property.second.isSupported()) {
                supportedProperties.emplace_back(ov::PropertyName(property.first, property.second.mutability));
            }
        }
        return supportedProperties;
    });

    register_property_with_custom_function(_properties, ov::internal::supported_properties.name(), false, ov::PropertyMutability::RO, [this](const ov::AnyMap&) {
        return _internalSupportedProperties;
    });
    register_property_with_custom_function(_properties, ov::internal::cache_header_alignment.name(), false, ov::PropertyMutability::RO, [](const ov::AnyMap&) {
        return utils::STANDARD_PAGE_SIZE;
    });
    register_property_with_custom_function(_properties, ov::internal::caching_properties.name(), false, ov::PropertyMutability::RO, [this](const ov::AnyMap&) {
        std::vector<ov::PropertyName> caching_props{};
        for (auto prop : _cachingProperties) {
            const auto propertyIt = _properties.find(prop);
            if (propertyIt != _properties.end() && propertyIt->second.isSupported()) {
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
    register_property_with_support_and_custom_function(_properties, ov::device::pci_info.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this](const ov::AnyMap&) {
        return utils::getPciInfo(_backend, _config.get<intel_npu::DEVICE_ID>());
    });
    register_property_with_support_and_custom_function(_properties, ov::device::gops.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this](const ov::AnyMap&) {
        return utils::getGops(_backend, _config.get<intel_npu::DEVICE_ID>());
    });
    register_property_with_support_and_custom_function(_properties, ov::device::type.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this](const ov::AnyMap&) {
        return utils::getDeviceType(_backend, _config.get<intel_npu::DEVICE_ID>());
    });
    register_property_with_support_and_custom_function(_properties, ov::intel_npu::device_alloc_mem_size.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this](const ov::AnyMap&) {
        return utils::getDeviceAllocMemSize(_backend, _config.get<intel_npu::DEVICE_ID>());
    });
    register_property_with_support_and_custom_function(_properties, ov::intel_npu::device_total_mem_size.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this](const ov::AnyMap&) {
        return utils::getDeviceTotalMemSize(_backend, _config.get<intel_npu::DEVICE_ID>());
    });
    register_property_with_support_and_custom_function(_properties, ov::device::uuid.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this](const ov::AnyMap&) {
        auto devUuid = utils::getDeviceUuid(_backend, _config.get<intel_npu::DEVICE_ID>());
        return decltype(ov::device::uuid)::value_type{devUuid};
    });
    register_property_with_support_and_custom_function(_properties, ov::device::architecture.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this](const ov::AnyMap&) {
        const auto devName = utils::getDeviceName(_backend, _config.get<intel_npu::DEVICE_ID>());
        return utils::getPlatformByDeviceName(devName);
    });
    register_property_with_support_and_custom_function(_properties, ov::device::full_name.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this](const ov::AnyMap&) {
        return utils::getFullDeviceName(_backend, _config.get<intel_npu::DEVICE_ID>());
    });
    register_property_with_support_and_custom_function(_properties, ov::device::luid.name(), _backend != nullptr && _backend->isLUIDExtSupported(), ov::PropertyMutability::RO, hasBackendAndValidDevice, [this](const ov::AnyMap&) {
        return utils::getDeviceLUID(_backend, _config.get<intel_npu::DEVICE_ID>());
    });
    register_property_with_support_and_custom_function(_properties, ov::intel_npu::stepping.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this](const ov::AnyMap&) {
        if (!_config.has<STEPPING>()) {
            try {
                const auto specifiedDeviceName = _config.get<intel_npu::DEVICE_ID>();
                return static_cast<int64_t>(utils::getSteppingNumber(_backend, specifiedDeviceName));
            } catch (...) {
                _logger.warning("GetSteppingNumber failed to get value from device.");
            }
        }
        return _config.get<STEPPING>();
    });
    register_property_with_support_and_custom_function(_properties, ov::intel_npu::max_tiles.name(), true, ov::PropertyMutability::RO, hasBackendAndValidDevice, [this](const ov::AnyMap&) { 
        if (!_config.has<MAX_TILES>()) {
            try {
                const auto specifiedDeviceName = _config.get<intel_npu::DEVICE_ID>();
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
        [this, isCompilerOptionSupported]() {  // support predicate
            if (!isCompilerOptionSupported(ENABLE_STRIDES_FOR::key())) {
                return false;
            }
            // Return true if the backend is not available, in case of offline compilation.
            if (_backend == nullptr) {
                return true;
            }
            // If a backend is present, check if the driver supports this property. If not, return false.
            if (_backend->getGraphExtVersion() < ZE_MAKE_VERSION(1, 16)) {
                _logger.info("Config option %s not supported by the driver! Requirements not met.", ENABLE_STRIDES_FOR::key());
                return false;
            }
            return true;
        },
        [this](const ov::AnyMap&) { // value getter
            return _config.get<ENABLE_STRIDES_FOR>();
        });
    register_property_with_support_and_custom_function<TURBO>(_config, _properties, true, ov::PropertyMutability::RW,
        [this, isCompilerOptionSupported]() {  // support predicate
            if (isCompilerOptionSupported(TURBO::key())) {
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
         compilerVersionSupportCache =
             std::optional<std::tuple<ov::intel_npu::CompilerType, std::string, bool>>{}]() mutable {  // support
                                                                                                       // predicate
            try {
                auto compilerType = _config.get<COMPILER_TYPE>();
                auto deviceId = _config.get<DEVICE_ID>();
                auto device = utils::getDeviceById(_backend, deviceId);

                auto compilationPlatform = utils::getCompilationPlatform(
                    _config.get<PLATFORM>(),
                    device == nullptr ? std::move(deviceId) : device->getName(),
                    _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());

                if (compilerVersionSupportCache.has_value() &&
                    std::get<0>(compilerVersionSupportCache.value()) == compilerType &&
                    std::get<1>(compilerVersionSupportCache.value()) == compilationPlatform) {
                    return std::get<2>(compilerVersionSupportCache.value());
                }

                CompilerAdapterFactory factory;
                const bool isSupported = factory.getCompiler(_backend, compilerType, compilationPlatform) != nullptr;
                compilerVersionSupportCache = std::make_tuple(compilerType, compilationPlatform, isSupported);
                return isSupported;
            } catch (...) {
                return false;
            }
        },
        [this](const ov::AnyMap&) {  // value getter
            auto compilerType = _config.get<COMPILER_TYPE>();
            auto deviceId = _config.get<DEVICE_ID>();
            auto device = utils::getDeviceById(_backend, deviceId);

            auto compilationPlatform = utils::getCompilationPlatform(
                _config.get<PLATFORM>(),
                device == nullptr ? std::move(deviceId) : device->getName(),
                _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());

            CompilerAdapterFactory factory;
            auto dummyCompiler = factory.getCompiler(_backend, compilerType, compilationPlatform);

            return dummyCompiler->get_version();
        });

    register_property_with_support_custom_function_and_args(
        _properties,
        ov::compatibility_check.name(),
        true,
        ov::PropertyMutability::RO,
        [this, compatibilityCheckSupported = std::optional<bool>{}]() mutable {  // support predicate
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

void PluginPropertyManager::setProperty(const ov::AnyMap& properties) {
    std::lock_guard<std::mutex> lock(_mutex);

    auto resolveCompilerTypeWithoutLock = [this](const ov::AnyMap& propertyMap) {
        auto compilerTypeIt = propertyMap.find(ov::intel_npu::compiler_type.name());
        if (compilerTypeIt != propertyMap.end()) {
            return COMPILER_TYPE::parse(compilerTypeIt->second.as<std::string>());
        }
        return _config.get<COMPILER_TYPE>();
    };

    auto resolveDeviceIdWithoutLock = [this](const ov::AnyMap& propertyMap) {
        auto deviceIdIt = propertyMap.find(std::string(ov::device::id.name()));
        if (deviceIdIt != propertyMap.end()) {
            return deviceIdIt->second.as<std::string>();
        }
        return _config.get<DEVICE_ID>();
    };

    auto resolvePlatformWithoutLock = [this](const ov::AnyMap& propertyMap) {
        auto platformIt = propertyMap.find(ov::intel_npu::platform.name());
        if (platformIt != propertyMap.end()) {
            return platformIt->second.as<std::string>();
        }
        return _config.get<PLATFORM>();
    };

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
        refreshCompilerPropertiesIfNeeded(compilerType, std::move(compilationPlatform));
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
                    _compilerOptionSupportHelper->isOptionSupported(_currentlyUsedCompiler, value.first, std::nullopt);
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
            if (!descriptor.isSupported()) {
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
            compiler = factory.getCompiler(_backend,
                                           compilerType,
                                           compilationPlatform,
                                           _compilerOptionSupportHelper->getOptionSupportCache());
            refreshCompilerPropertiesIfNeeded(compilerType, std::move(compilationPlatform));
        } catch (const std::exception& ex) {
            if (_config.hasOpt(name) && _config.getOpt(name).mode() == OptionMode::CompileTime) {
                OPENVINO_THROW("Failed to create compiler for getting property ", name, " with error: ", ex.what());
            }

            _logger.warning("Failed to create compiler for getting property %s with error: %s. Returning only "
                            "properties that do not require compiler support.",
                            name.c_str(),
                            ex.what());
        }
    }

    auto&& configIterator = _properties.find(name);
    if (configIterator != _properties.cend()) {
        if (!configIterator->second.isSupported()) {
            OPENVINO_THROW("Unsupported configuration key: ", name);
        }
        if (configIterator->second.mutability == ov::PropertyMutability::WO) {
            _logger.warning("Trying to get WRITE-ONLY property: %s. Returning empty `ov::Any` object", name.c_str());
            return ov::Any();
        }
        return configIterator->second.get(arguments);
    }
    try {
        return _config.getInternal(name);
    } catch (...) {
        OPENVINO_THROW("Unsupported configuration key: ", name);
    }
}

bool PluginPropertyManager::isPropertySupported(const std::string& name, const ov::AnyMap& arguments) {
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
    if (!isPropertyRegistered(name)) {
        return false;
    }

    if (!_config.hasOpt(name) || isSpecialBothProperty(name)) {
        const auto it = _properties.find(name);
        return it->second.isPublic && it->second.isSupported();
    }

    auto opt = _config.getOpt(name);
    if (opt.mode() == OptionMode::RunTime) {
        const auto it = _properties.find(name);
        return it->second.isPublic && it->second.isSupported();
    }

    if (name == ov::intel_npu::turbo.name()) {
        // Fast path: if turbo is already exposed and supported by the driver, return immediately.
        // Otherwise, fall through to compiler-based support check.
        const auto it = _properties.find(name);
        if (_backend != nullptr && _backend->isCommandQueueExtSupported() && it != _properties.end() &&
            it->second.isPublic) {
            return true;
        }
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
        compiler = factory.getCompiler(_backend,
                                       compilerType,
                                       compilationPlatform,
                                       _compilerOptionSupportHelper->getOptionSupportCache());
        refreshCompilerPropertiesIfNeeded(compilerType, std::move(compilationPlatform));
    } catch (const std::exception& ex) {
        if (_config.hasOpt(name) && _config.getOpt(name).mode() == OptionMode::CompileTime) {
            return false;
        }

        _logger.warning("Failed to create compiler to query property %s with error: %s. Registering only properties "
                        "that do not require compiler support.",
                        name.c_str(),
                        ex.what());
    }

    const auto it = _properties.find(name);
    return it != _properties.end() && it->second.isPublic && it->second.isSupported();
}

FilteredConfig PluginPropertyManager::getConfigWithCompilerPropertiesDisabled(const ov::AnyMap& properties) const {
    // TODO
    return _config;
}

FilteredConfig PluginPropertyManager::getConfigForSpecificCompiler(const ov::AnyMap& properties) const {
    auto [updatedConfig, currentlyUsedCompiler, currentlyUsedPlatform, logger] = [&]() {
        std::lock_guard<std::mutex> lock(_mutex);
        return std::make_tuple(_config, _currentlyUsedCompiler, _currentlyUsedPlatform, _logger);
    }();

    if (properties.find(ov::hint::enable_cpu_pinning.name()) != properties.end()) {
        logCpuPinningDeprecationWarning(logger);
    }

    auto pluginProperties = properties;
    exclude_model_ptr_from_map(pluginProperties);

    std::optional<ov::intel_npu::CompilerType> propertiesCompilerType = std::nullopt;
    std::optional<std::string> propertiesPlatform = std::nullopt;
    auto compilerType = pluginProperties.find(ov::intel_npu::compiler_type.name());
    if (compilerType != pluginProperties.end()) {
        propertiesCompilerType = compilerType->second.as<ov::intel_npu::CompilerType>();
    }
    auto platform = pluginProperties.find(ov::intel_npu::platform.name());
    if (platform != pluginProperties.end()) {
        propertiesPlatform = platform->second.as<std::string>();
    }

    const std::map<std::string, std::string> rawConfig = any_copy(pluginProperties);
    std::map<std::string, std::string> cfgsToSet;
    ov::AnyMap specialCfgsToSet;
    for (const auto& [key, value] : rawConfig) {
        if (!updatedConfig.hasOpt(key)) {
            // not a known config key
            bool isSupported = false;
            try {
                isSupported = _compilerOptionSupportHelper->isOptionSupported(
                    propertiesCompilerType.value_or(currentlyUsedCompiler),
                    key,
                    std::nullopt);
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

}  // namespace intel_npu
