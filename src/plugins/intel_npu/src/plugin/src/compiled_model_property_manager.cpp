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
    if (!compatibilityDescriptor.has_value()) {
        OPENVINO_THROW("RUNTIME_REQUIREMENTS cannot be generated for this compiled model.");
    }
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
                                                           const std::shared_ptr<IGraph>& graph,
                                                           const std::optional<int64_t>& batchSize,
                                                           Logger& logger)
    : _config(config),
      _graph(graph),
      _batchSize(batchSize),
      _logger(logger) {
    registerProperties();
}

void CompiledModelPropertyManager::setProperty(const ov::AnyMap& properties) {
    std::lock_guard<std::mutex> lock(_mutex);

    std::map<std::string, std::string> configsToSet;
    ov::AnyMap specialConfigsToSet;

    for (const auto& property : properties) {
        const auto propertyIt = _properties.find(property.first);
        if (propertyIt == _properties.end() || !propertyIt->second.isSupported()) {
            OPENVINO_THROW("Unsupported configuration key: ", property.first);
        }
        if (propertyIt->second.mutability == ov::PropertyMutability::RO) {
            OPENVINO_THROW("READ-ONLY configuration key: ", property.first);
        }

        if (property.first == ov::cache_encryption_callbacks.name()) {
            specialConfigsToSet.emplace(property.first, property.second);
        } else {
            configsToSet.emplace(property.first, property.second.as<std::string>());
        }
    }

    if (!configsToSet.empty()) {
        _config.update(configsToSet);
    }

    if (!specialConfigsToSet.empty()) {
        _config.updateAny(specialConfigsToSet);
    }
}

ov::Any CompiledModelPropertyManager::getProperty(const std::string& name) const {
    std::lock_guard<std::mutex> lock(_mutex);

    if (name == ov::hint::enable_cpu_pinning.name()) {
        logCpuPinningDeprecationWarning(_logger);
    }

    const auto propertyIt = _properties.find(name);
    if (propertyIt != _properties.end()) {
        if (!propertyIt->second.isSupported()) {
            OPENVINO_THROW("Unsupported configuration key: ", name);
        }
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

void CompiledModelPropertyManager::registerProperties() {
    _properties.clear();

    // clang-format off
    register_property<MODEL_PRIORITY>(_config, _properties, true, ov::PropertyMutability::RW);
    register_property<WORKLOAD_TYPE>(_config, _properties, true, ov::PropertyMutability::RW); // TODO

    register_property<CACHE_MODE>(_config, _properties, true, ov::PropertyMutability::RO);
    register_property<COMPILATION_NUM_THREADS>(_config, _properties, true, ov::PropertyMutability::RO);
    register_property<EXECUTION_MODE_HINT>(_config, _properties, true, ov::PropertyMutability::RO);
    register_property<INFERENCE_PRECISION_HINT>(_config, _properties, true, ov::PropertyMutability::RO);
    register_property<LOADED_FROM_CACHE>(_config, _properties, true, ov::PropertyMutability::RO);
    register_property<LOG_LEVEL>(_config, _properties, true, ov::PropertyMutability::RO);
    register_property<PERFORMANCE_HINT>(_config, _properties, true, ov::PropertyMutability::RO);
    register_property<PERFORMANCE_HINT_NUM_REQUESTS>(_config, _properties, true, ov::PropertyMutability::RO);

    OPENVINO_SUPPRESS_DEPRECATED_START
    register_property<ENABLE_CPU_PINNING>(_config, _properties, false, ov::PropertyMutability::RO);
    OPENVINO_SUPPRESS_DEPRECATED_END

    const auto hasPropertyValue = [this](std::string_view property_name) {
        return _config.hasOpt(property_name) && _config.has(std::string(property_name));
    };
    register_property_with_support<BYPASS_UMD_CACHING>(_config, _properties, true, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(BYPASS_UMD_CACHING::key()); });
    register_property_with_support<CACHE_DIR>(_config, _properties, true, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(CACHE_DIR::key()); });
    register_property_with_support<COMPILATION_MODE_PARAMS>(_config, _properties, true, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(COMPILATION_MODE_PARAMS::key()); });
    register_property_with_support<COMPILER_DYNAMIC_QUANTIZATION>(_config, _properties, true, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(COMPILER_DYNAMIC_QUANTIZATION::key()); });
    register_property_with_support<COMPILER_TYPE>(_config, _properties, true, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(COMPILER_TYPE::key()); });
    register_property_with_support<COMPILER_VERSION>(_config, _properties, true, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(COMPILER_VERSION::key()); });
    register_property_with_support<DEFER_WEIGHTS_LOAD>(_config, _properties, true, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(DEFER_WEIGHTS_LOAD::key()); });
    register_property_with_support<ENABLE_STRIDES_FOR>(_config, _properties, true, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(ENABLE_STRIDES_FOR::key()); });
    register_property_with_support<NUM_STREAMS>(_config, _properties, true, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(NUM_STREAMS::key()); });
    register_property_with_support<PERF_COUNT>(_config, _properties, true, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(PERF_COUNT::key()); });
    register_property_with_support<PLATFORM>(_config, _properties, true, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(PLATFORM::key()); });
    register_property_with_support<QDQ_OPTIMIZATION>(_config, _properties, true, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(QDQ_OPTIMIZATION::key()); });
    register_property_with_support<QDQ_OPTIMIZATION_AGGRESSIVE>(_config, _properties, true, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(QDQ_OPTIMIZATION_AGGRESSIVE::key()); });
    register_property_with_support<RUN_INFERENCES_SEQUENTIALLY>(_config, _properties, true, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(RUN_INFERENCES_SEQUENTIALLY::key()); });
    register_property_with_support<TILES>(_config, _properties, true, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(TILES::key()); });
    register_property_with_support<TURBO>(_config, _properties, true, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(TURBO::key()); });
    register_property_with_support<WEIGHTS_PATH>(_config, _properties, true, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(WEIGHTS_PATH::key()); });

    register_property_with_support<BACKEND_COMPILATION_PARAMS>(_config, _properties, false, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(BACKEND_COMPILATION_PARAMS::key()); });
    register_property_with_support<BATCH_COMPILER_MODE_SETTINGS>(_config, _properties, false, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(BATCH_COMPILER_MODE_SETTINGS::key()); });
    register_property_with_support<BATCH_MODE>(_config, _properties, false, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(BATCH_MODE::key()); });
    register_property_with_support<COMPILATION_MODE>(_config, _properties, false, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(COMPILATION_MODE::key()); });
    register_property_with_support<COMPILE_LOG_LEVEL>(_config, _properties, false, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(COMPILE_LOG_LEVEL::key()); });
    register_property_with_support<DISABLE_VERSION_CHECK>(_config, _properties, false, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(DISABLE_VERSION_CHECK::key()); });
    register_property_with_support<DMA_ENGINES>(_config, _properties, false, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(DMA_ENGINES::key()); });
    register_property_with_support<DYNAMIC_SHAPE_TO_STATIC>(_config, _properties, false, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(DYNAMIC_SHAPE_TO_STATIC::key()); });
    register_property_with_support<ENABLE_WEIGHTLESS>(_config, _properties, false, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(ENABLE_WEIGHTLESS::key()); });
    register_property_with_support<EXPORT_RAW_BLOB>(_config, _properties, false, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(EXPORT_RAW_BLOB::key()); });
    register_property_with_support<IMPORT_RAW_BLOB>(_config, _properties, false, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(IMPORT_RAW_BLOB::key()); });
    register_property_with_support<PROFILING_TYPE>(_config, _properties, false, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(PROFILING_TYPE::key()); });
    register_property_with_support<SEPARATE_WEIGHTS_VERSION>(_config, _properties, false, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(SEPARATE_WEIGHTS_VERSION::key()); });
    register_property_with_support<SHARED_COMMON_QUEUE>(_config, _properties, false, ov::PropertyMutability::RO, [hasPropertyValue] { return hasPropertyValue(SHARED_COMMON_QUEUE::key()); });

    register_property_with_custom_function<CACHE_ENCRYPTION_CALLBACKS>(_config, _properties, true, ov::PropertyMutability::WO, [](const ov::AnyMap&) {
        return ov::EncryptionCallbacks{nullptr, nullptr};
    });
    register_property_with_custom_function(_properties, ov::hint::model.name(), true, ov::PropertyMutability::RO, [](const ov::AnyMap&) {
        return std::shared_ptr<const ov::Model>(nullptr);
    });
    register_property_with_custom_function(_properties, ov::model_name.name(), true, ov::PropertyMutability::RO, [this](const ov::AnyMap&) {
        OPENVINO_ASSERT(_graph != nullptr, "Missing graph");
        return ov::Any(_graph->get_metadata().name);
    });
    register_property_with_custom_function(_properties, ov::optimal_number_of_infer_requests.name(), true, ov::PropertyMutability::RO, [this](const ov::AnyMap&) {
        return ov::Any(utils::getOptimalNumberOfInferRequestsInParallel(_config.get<PLATFORM>(), _config.get<PERFORMANCE_HINT>()));
    });
    register_property_with_custom_function(_properties, ov::execution_devices.name(), true, ov::PropertyMutability::RO, [](const ov::AnyMap&) {
        return ov::Any(std::vector<std::string>{"NPU"});
    });
    register_property_with_custom_function(_properties, ov::supported_properties.name(), true, ov::PropertyMutability::RO, [this](const ov::AnyMap&) {
        std::vector<ov::PropertyName> supportedProperties;
        for (const auto& property : _properties) {
            if (property.second.isPublic && property.second.isSupported()) {
                supportedProperties.emplace_back(property.first, property.second.mutability);
            }
        }
        return ov::Any(supportedProperties);
    });
    // clang-format on

    const bool hasRuntimeRequirementsSupport = _graph != nullptr && _graph->get_compatibility_descriptor().has_value();
    register_property_with_support_and_custom_function(
        _properties,
        ov::runtime_requirements.name(),
        true,
        ov::PropertyMutability::RO,
        [hasRuntimeRequirementsSupport]() {  // support predicate
            return hasRuntimeRequirementsSupport;
        },
        [this](const ov::AnyMap&) {  // value getter
            return ov::Any(buildRuntimeRequirements(_graph, _batchSize, _logger));
        });
}

}  // namespace intel_npu
