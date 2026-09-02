// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model_property_manager.hpp"

#include <sstream>

#include "intel_npu/common/device_helpers.hpp"
#include "intel_npu/config/options.hpp"
#include "metadata.hpp"

namespace {

inline void logCpuPinningDeprecationWarning(intel_npu::Logger& logger) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    logger.warning(intel_npu::ENABLE_CPU_PINNING::deprecationMessage());
    OPENVINO_SUPPRESS_DEPRECATED_END
}

std::string buildRuntimeRequirements(const std::shared_ptr<intel_npu::IGraph>& graph,
                                     const std::optional<int64_t>& batchSize,
                                     intel_npu::Logger& logger) {
    OPENVINO_ASSERT(graph != nullptr, "Missing graph");

    auto compatibilityDescriptor = graph->get_compatibility_descriptor();
    OPENVINO_ASSERT(compatibilityDescriptor.has_value(),
                    "RUNTIME_REQUIREMENTS cannot be generated for this compiled model.");
    const auto descriptorView = compatibilityDescriptor.value();
    logger.debug("Runtime requirements from the graph %.*s length: %zu",
                 static_cast<int>(descriptorView.size()),
                 descriptorView.data(),
                 descriptorView.size());

    std::ostringstream requirementsString;
    intel_npu::Metadata<intel_npu::CURRENT_METADATA_VERSION>(0,
                                                             intel_npu::CURRENT_OPENVINO_VERSION,
                                                             std::nullopt,
                                                             batchSize,
                                                             std::nullopt,
                                                             std::nullopt,
                                                             std::nullopt,
                                                             std::nullopt,
                                                             compatibilityDescriptor)
        .write_as_text(requirementsString);

    logger.debug("Runtime requirements string: %s length: %zu",
                 requirementsString.str().c_str(),
                 requirementsString.str().length());

    return requirementsString.str();
}

}  // namespace

namespace intel_npu {

CompiledModelPropertyManager::CompiledModelPropertyManager(const FilteredConfig& config,
                                                           const ov::AnyMap& properties,
                                                           const std::shared_ptr<IDevice>& device,
                                                           const std::shared_ptr<IGraph>& graph,
                                                           const std::optional<int64_t>& batchSize,
                                                           Logger& logger)
    : _config(config),
      _device(device),
      _graph(graph),
      _batchSize(batchSize),
      _logger(logger) {
    registerProperties();

    // Set the properties from the provided ov::AnyMap into the internal property descriptors
    for (const auto& property : properties) {
        const auto propertyDescriptorIt = _properties.find(property.first);
        OPENVINO_ASSERT(propertyDescriptorIt != _properties.end(), "Unsupported configuration key: ", property.first);
        OPENVINO_ASSERT(propertyDescriptorIt->second.isSupported(properties),
                        "Unsupported configuration key: ",
                        property.first);
        propertyDescriptorIt->second.set(property.second);
    }
}

void CompiledModelPropertyManager::setProperty(const ov::AnyMap& properties) {
    std::lock_guard<std::mutex> lock(_mutex);

    for (const auto& property : properties) {
        const auto propertyIt = _properties.find(property.first);
        OPENVINO_ASSERT(propertyIt != _properties.end(), "Unsupported configuration key: ", property.first);
        OPENVINO_ASSERT(propertyIt->second.mutability != ov::PropertyMutability::RO,
                        "READ-ONLY configuration key: ",
                        property.first);
        OPENVINO_ASSERT(propertyIt->second.isSupported(properties), "Unsupported configuration key: ", property.first);
    }

    for (const auto& property : properties) {
        const auto propertyIt = _properties.find(property.first);
        propertyIt->second.set(property.second);
    }
}

ov::Any CompiledModelPropertyManager::getProperty(const std::string& name) const {
    std::lock_guard<std::mutex> lock(_mutex);

    if (name == ov::hint::enable_cpu_pinning.name()) {
        logCpuPinningDeprecationWarning(_logger);
    }

    const auto propertyIt = _properties.find(name);
    if (propertyIt != _properties.end()) {
        OPENVINO_ASSERT(propertyIt->second.isSupported(ov::AnyMap{}), "Unsupported configuration key: ", name);
        if (propertyIt->second.mutability == ov::PropertyMutability::WO) {
            _logger.warning("Trying to get WRITE-ONLY property: %s. Returning empty `ov::Any` object", name.c_str());
            return ov::Any();
        }
        return propertyIt->second.get(ov::AnyMap{});
    }

    try {
        return _config.getInternal(name);
    } catch (...) {
        OPENVINO_THROW("Unsupported configuration key: ", name);
    }
}

FilteredConfig CompiledModelPropertyManager::getConfig() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _config;
}

void CompiledModelPropertyManager::registerProperties() {
    _properties.clear();

    const auto hasPropertyValue = [this](const std::string& property_name) {
        return _config.has(property_name);
    };

    const auto readOnlySetter = [](const ov::Any&) {
        OPENVINO_THROW("READ-ONLY configuration key");
    };

    const auto registerConfigProperty =
        [this,
         hasPropertyValue](const auto optionTag, bool isPublic, ov::PropertyMutability mutability, bool requireValue) {
            using OptionType = std::decay_t<decltype(optionTag)>;
            const auto propertyName = std::string(OptionType::key());
            const auto isSupported = [this, propertyName, hasPropertyValue, requireValue](const ov::AnyMap&) {
                return requireValue ? hasPropertyValue(propertyName) : _config.hasOpt(propertyName);
            };
            const auto setter = mutability == ov::PropertyMutability::RO
                                    ? std::function<void(const ov::Any&)>([](const ov::Any&) {
                                          OPENVINO_THROW("READ-ONLY configuration key");
                                      })
                                    : std::function<void(const ov::Any&)>([this, propertyName](const ov::Any& value) {
                                          _config.update(propertyName, value.as<std::string>());
                                      });
            register_property(
                propertyName,
                isPublic,
                mutability,
                isSupported,
                [this](const ov::AnyMap&) {
                    return _config.get<OptionType>();
                },
                setter);
        };

    registerConfigProperty(CACHE_MODE{}, true, ov::PropertyMutability::RO, false);
    registerConfigProperty(COMPILATION_NUM_THREADS{}, true, ov::PropertyMutability::RO, false);
    registerConfigProperty(EXECUTION_MODE_HINT{}, true, ov::PropertyMutability::RO, false);
    registerConfigProperty(INFERENCE_PRECISION_HINT{}, true, ov::PropertyMutability::RO, false);
    registerConfigProperty(LOADED_FROM_CACHE{}, true, ov::PropertyMutability::RO, false);
    registerConfigProperty(LOG_LEVEL{}, true, ov::PropertyMutability::RO, false);
    registerConfigProperty(PERFORMANCE_HINT{}, true, ov::PropertyMutability::RO, false);
    registerConfigProperty(PERFORMANCE_HINT_NUM_REQUESTS{}, true, ov::PropertyMutability::RO, false);

    OPENVINO_SUPPRESS_DEPRECATED_START
    registerConfigProperty(ENABLE_CPU_PINNING{}, false, ov::PropertyMutability::RO, false);
    OPENVINO_SUPPRESS_DEPRECATED_END

    registerConfigProperty(BYPASS_UMD_CACHING{}, true, ov::PropertyMutability::RO, true);
    registerConfigProperty(CACHE_DIR{}, true, ov::PropertyMutability::RO, true);
    registerConfigProperty(COMPILATION_MODE_PARAMS{}, true, ov::PropertyMutability::RO, true);
    registerConfigProperty(COMPILER_DYNAMIC_QUANTIZATION{}, true, ov::PropertyMutability::RO, true);
    registerConfigProperty(COMPILER_TYPE{}, true, ov::PropertyMutability::RO, true);
    registerConfigProperty(COMPILER_VERSION{}, true, ov::PropertyMutability::RO, true);
    registerConfigProperty(DEFER_WEIGHTS_LOAD{}, true, ov::PropertyMutability::RO, true);
    registerConfigProperty(ENABLE_STRIDES_FOR{}, true, ov::PropertyMutability::RO, true);
    registerConfigProperty(NUM_STREAMS{}, true, ov::PropertyMutability::RO, true);
    registerConfigProperty(PERF_COUNT{}, true, ov::PropertyMutability::RO, true);
    registerConfigProperty(PLATFORM{}, true, ov::PropertyMutability::RO, true);
    registerConfigProperty(QDQ_OPTIMIZATION{}, true, ov::PropertyMutability::RO, true);
    registerConfigProperty(QDQ_OPTIMIZATION_AGGRESSIVE{}, true, ov::PropertyMutability::RO, true);
    registerConfigProperty(RUN_INFERENCES_SEQUENTIALLY{}, true, ov::PropertyMutability::RO, true);
    registerConfigProperty(TILES{}, true, ov::PropertyMutability::RO, true);
    registerConfigProperty(TURBO{}, true, ov::PropertyMutability::RO, true);
    registerConfigProperty(WEIGHTS_PATH{}, true, ov::PropertyMutability::RO, true);

    registerConfigProperty(BACKEND_COMPILATION_PARAMS{}, false, ov::PropertyMutability::RO, true);
    registerConfigProperty(BATCH_COMPILER_MODE_SETTINGS{}, false, ov::PropertyMutability::RO, true);
    registerConfigProperty(BATCH_MODE{}, false, ov::PropertyMutability::RO, true);
    registerConfigProperty(COMPILATION_MODE{}, false, ov::PropertyMutability::RO, true);
    registerConfigProperty(COMPILE_LOG_LEVEL{}, false, ov::PropertyMutability::RO, true);
    registerConfigProperty(DISABLE_VERSION_CHECK{}, false, ov::PropertyMutability::RO, true);
    registerConfigProperty(DMA_ENGINES{}, false, ov::PropertyMutability::RO, true);
    registerConfigProperty(DYNAMIC_SHAPE_TO_STATIC{}, false, ov::PropertyMutability::RO, true);
    registerConfigProperty(ENABLE_WEIGHTLESS{}, false, ov::PropertyMutability::RO, true);
    registerConfigProperty(EXPORT_RAW_BLOB{}, false, ov::PropertyMutability::RO, true);
    registerConfigProperty(IMPORT_RAW_BLOB{}, false, ov::PropertyMutability::RO, true);
    registerConfigProperty(PROFILING_TYPE{}, false, ov::PropertyMutability::RO, true);
    registerConfigProperty(SEPARATE_WEIGHTS_VERSION{}, false, ov::PropertyMutability::RO, true);
    registerConfigProperty(SHARED_COMMON_QUEUE{}, false, ov::PropertyMutability::RO, true);

    // clang-format off
    register_property(ov::hint::model_priority.name(), true, ov::PropertyMutability::RW,
        [this](const ov::AnyMap&) {
            return _config.hasOpt(ov::hint::model_priority.name());
        },
        [this](const ov::AnyMap&) {
            return _config.get<MODEL_PRIORITY>();
        },
        [this](const ov::Any& value) {
            _config.update(ov::hint::model_priority.name(), value.as<std::string>());
            if (_graph != nullptr) {
                _graph->set_model_priority(value.as<ov::hint::Priority>());
            }
        }
    );
    register_property(ov::workload_type.name(), true, ov::PropertyMutability::RW,
        [this](const ov::AnyMap&) {
            return _config.hasOpt(ov::workload_type.name());
        },
        [this](const ov::AnyMap&) {
            return _config.get<WORKLOAD_TYPE>();
        },
        [this](const ov::Any& value) {
            _config.update(ov::workload_type.name(), value.as<std::string>());
            if (_graph != nullptr) {
                _graph->set_workload_type(value.as<ov::WorkloadType>());
            }
        }
    );
    register_property(ov::cache_encryption_callbacks.name(), true, ov::PropertyMutability::WO,
        [this](const ov::AnyMap&) {
            return _config.hasOpt(ov::cache_encryption_callbacks.name());
        },
        [](const ov::AnyMap&) {
            return ov::EncryptionCallbacks{nullptr, nullptr};
        },
        [this](const ov::Any& value) {
            _config.updateAny(ov::cache_encryption_callbacks.name(), value);
        }
    );
    register_property(ov::hint::model.name(), true, ov::PropertyMutability::RO,
        [](const ov::AnyMap&) {
            return true;
        },
        [this](const ov::AnyMap&) {
            return _config.get<MODEL_PTR>().lock();
        },
        readOnlySetter
    );
    register_property(ov::model_name.name(), true, ov::PropertyMutability::RO,
        [this](const ov::AnyMap&) {
            return _graph != nullptr;
        },
        [this](const ov::AnyMap&) {
            OPENVINO_ASSERT(_graph != nullptr, "Missing graph");
            return ov::Any(_graph->get_metadata().name);
        },
        readOnlySetter
    );
    register_property(ov::optimal_number_of_infer_requests.name(), true, ov::PropertyMutability::RO,
        [](const ov::AnyMap&) {
            return true;
        },
        [this](const ov::AnyMap&) {
            if (!_device) {
                return ov::Any(uint32_t{0});
            }
            return ov::Any(utils::getOptimalNumberOfInferRequestsInParallel(_device->getName(), _config.get<PERFORMANCE_HINT>()));
        },
        readOnlySetter
    );
    register_property(ov::execution_devices.name(), true, ov::PropertyMutability::RO,
        [](const ov::AnyMap&) {
            return true;
        },
        [](const ov::AnyMap&) {
            return ov::Any(std::vector<std::string>{"NPU"});
        },
        readOnlySetter
    );
    register_property(ov::supported_properties.name(), true, ov::PropertyMutability::RO,
        [](const ov::AnyMap&) {
            return true;
        },
        [this](const ov::AnyMap& arguments) {
            std::vector<ov::PropertyName> supportedProperties;
            for (const auto& property : _properties) {
                if (property.second.isPublic && property.second.isSupported(arguments)) {
                    supportedProperties.emplace_back(property.first, property.second.mutability);
                }
            }
            return ov::Any(supportedProperties);
        },
        readOnlySetter
    );
    register_property(ov::runtime_requirements.name(), true, ov::PropertyMutability::RO,
        [this](const ov::AnyMap&) {
            return  _graph != nullptr && _graph->get_compatibility_descriptor().has_value();
        },
        [this](const ov::AnyMap&) {
            return ov::Any(buildRuntimeRequirements(_graph, _batchSize, _logger));
        },
        readOnlySetter
    );
    // clang-format on
}

}  // namespace intel_npu
