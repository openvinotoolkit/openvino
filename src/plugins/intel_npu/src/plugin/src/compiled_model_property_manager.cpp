// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model_property_manager.hpp"

#include <sstream>

#include "intel_npu/config/options.hpp"
#include "metadata.hpp"

namespace intel_npu {

namespace {

inline void logCpuPinningDeprecationWarning(Logger& logger) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    logger.warning(ENABLE_CPU_PINNING::deprecationMessage());
    OPENVINO_SUPPRESS_DEPRECATED_END
}

}  // namespace

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

    if (properties.find(ov::log::level.name()) != properties.end()) {
        _logger.setLevel(properties.at(ov::log::level.name()).as<ov::log::Level>());
    }

    if (properties.find(ov::hint::enable_cpu_pinning.name()) != properties.end()) {
        logCpuPinningDeprecationWarning(_logger);
    }

    std::map<std::string, std::string> configsToSet;
    ov::AnyMap specialConfigsToSet;

    for (const auto& property : properties) {
        const auto propertyIt = _properties.find(property.first);
        if (propertyIt == _properties.end()) {
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
        if (propertyIt->second.mutability == ov::PropertyMutability::WO) {
            _logger.warning("Trying to get WRITE-ONLY property: %s. Returning empty `ov::Any` object", name.c_str());
            return ov::Any();
        }
        return propertyIt->second.get(_config);
    }

    try {
        return _config.getInternal(name);
    } catch (...) {
        OPENVINO_THROW("Unsupported configuration key: ", name);
    }
}

void CompiledModelPropertyManager::registerProperties() {
    _properties.clear();
    _supportedProperties.clear();

    OPENVINO_SUPPRESS_DEPRECATED_START
    TRY_REGISTER_COMPILEDMODEL_PROPERTY(_config, _properties, ov::hint::enable_cpu_pinning, ENABLE_CPU_PINNING);
    OPENVINO_SUPPRESS_DEPRECATED_END
    TRY_REGISTER_COMPILEDMODEL_PROPERTY(_config, _properties, ov::log::level, LOG_LEVEL);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY(_config, _properties, ov::loaded_from_cache, LOADED_FROM_CACHE);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY(_config, _properties, ov::hint::performance_mode, PERFORMANCE_HINT);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY(_config, _properties, ov::hint::execution_mode, EXECUTION_MODE_HINT);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY(_config, _properties, ov::hint::num_requests, PERFORMANCE_HINT_NUM_REQUESTS);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY(_config, _properties, ov::compilation_num_threads, COMPILATION_NUM_THREADS);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY(_config, _properties, ov::hint::inference_precision, INFERENCE_PRECISION_HINT);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY(_config, _properties, ov::cache_mode, CACHE_MODE);

    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config, _properties, ov::num_streams, NUM_STREAMS);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config, _properties, ov::intel_npu::compiler_type, COMPILER_TYPE);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config, _properties, ov::intel_npu::compiler_version, COMPILER_VERSION);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config, _properties, ov::weights_path, WEIGHTS_PATH);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config, _properties, ov::cache_dir, CACHE_DIR);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config, _properties, ov::enable_profiling, PERF_COUNT);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config, _properties, ov::intel_npu::profiling_type, PROFILING_TYPE);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config, _properties, ov::intel_npu::turbo, TURBO);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config,
                                              _properties,
                                              ov::intel_npu::compilation_mode_params,
                                              COMPILATION_MODE_PARAMS);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config, _properties, ov::intel_npu::dma_engines, DMA_ENGINES);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config, _properties, ov::intel_npu::tiles, TILES);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config, _properties, ov::intel_npu::compilation_mode, COMPILATION_MODE);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config, _properties, ov::intel_npu::platform, PLATFORM);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config,
                                              _properties,
                                              ov::intel_npu::dynamic_shape_to_static,
                                              DYNAMIC_SHAPE_TO_STATIC);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config,
                                              _properties,
                                              ov::intel_npu::backend_compilation_params,
                                              BACKEND_COMPILATION_PARAMS);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config,
                                              _properties,
                                              ov::intel_npu::bypass_umd_caching,
                                              BYPASS_UMD_CACHING);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config,
                                              _properties,
                                              ov::intel_npu::defer_weights_load,
                                              DEFER_WEIGHTS_LOAD);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config,
                                              _properties,
                                              ov::intel_npu::compiler_dynamic_quantization,
                                              COMPILER_DYNAMIC_QUANTIZATION);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config, _properties, ov::intel_npu::qdq_optimization, QDQ_OPTIMIZATION);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config,
                                              _properties,
                                              ov::intel_npu::qdq_optimization_aggressive,
                                              QDQ_OPTIMIZATION_AGGRESSIVE);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config,
                                              _properties,
                                              ov::intel_npu::disable_version_check,
                                              DISABLE_VERSION_CHECK);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config, _properties, ov::intel_npu::export_raw_blob, EXPORT_RAW_BLOB);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config, _properties, ov::intel_npu::import_raw_blob, IMPORT_RAW_BLOB);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config,
                                              _properties,
                                              ov::intel_npu::batch_compiler_mode_settings,
                                              BATCH_COMPILER_MODE_SETTINGS);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config,
                                              _properties,
                                              ov::intel_npu::run_inferences_sequentially,
                                              RUN_INFERENCES_SEQUENTIALLY);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config, _properties, ov::enable_weightless, ENABLE_WEIGHTLESS);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config,
                                              _properties,
                                              ov::intel_npu::separate_weights_version,
                                              SEPARATE_WEIGHTS_VERSION);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(_config,
                                              _properties,
                                              ov::intel_npu::enable_strides_for,
                                              ENABLE_STRIDES_FOR);

    TRY_REGISTER_CUSTOM_PROPERTY(_config,
                                 _properties,
                                 ov::intel_npu::batch_mode,
                                 BATCH_MODE,
                                 false,
                                 ov::PropertyMutability::RO,
                                 [](const Config& config) {
                                     return config.get<BATCH_MODE>();
                                 });
    TRY_REGISTER_CUSTOM_PROPERTY(_config,
                                 _properties,
                                 ov::intel_npu::shared_common_queue,
                                 SHARED_COMMON_QUEUE,
                                 false,
                                 ov::PropertyMutability::RO,
                                 [](const Config& config) {
                                     return config.get<SHARED_COMMON_QUEUE>();
                                 });
    TRY_REGISTER_CUSTOM_PROPERTY(_config,
                                 _properties,
                                 ov::hint::model_priority,
                                 MODEL_PRIORITY,
                                 true,
                                 ov::PropertyMutability::RW,
                                 [](const Config& config) {
                                     return config.get<MODEL_PRIORITY>();
                                 });
    TRY_REGISTER_CUSTOM_PROPERTY(_config,
                                 _properties,
                                 ov::workload_type,
                                 WORKLOAD_TYPE,
                                 true,
                                 ov::PropertyMutability::RW,
                                 [](const Config& config) {
                                     return config.get<WORKLOAD_TYPE>();
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
    REGISTER_CUSTOM_METRIC(_properties, ov::model_name, true, [this](const Config&) {
        OPENVINO_ASSERT(_graph != nullptr, "Missing graph");
        return ov::Any(_graph->get_metadata().name);
    });
    REGISTER_CUSTOM_METRIC(_properties, ov::optimal_number_of_infer_requests, true, [](const Config& config) {
        return ov::Any(
            utils::getOptimalNumberOfInferRequestsInParallel(config.get<PLATFORM>(), config.get<PERFORMANCE_HINT>()));
    });
    REGISTER_CUSTOM_METRIC(_properties, ov::execution_devices, true, [](const Config&) {
        return ov::Any(std::string("NPU"));
    });
    if (_config.isAvailable(ov::runtime_requirements.name())) {
        REGISTER_CUSTOM_METRIC(_properties, ov::runtime_requirements, true, [this](const Config&) {
            return ov::Any(buildRuntimeRequirements());
        });
    }
    REGISTER_CUSTOM_METRIC(_properties, ov::supported_properties, true, [this](const Config&) {
        return ov::Any(_supportedProperties);
    });

    for (const auto& property : _properties) {
        if (property.second.isPublic) {
            _supportedProperties.emplace_back(property.first, property.second.mutability);
        }
    }
}

std::string CompiledModelPropertyManager::buildRuntimeRequirements() const {
    OPENVINO_ASSERT(_graph != nullptr, "Missing graph");

    auto compatibilityDescriptor = _graph->get_compatibility_descriptor();
    if (compatibilityDescriptor.has_value()) {
        const auto descriptorView = compatibilityDescriptor.value();
        _logger.debug("Runtime requirements from the graph %.*s length: %zu",
                      static_cast<int>(descriptorView.size()),
                      descriptorView.data(),
                      descriptorView.size());
    }

    std::ostringstream requirementsString;
    Metadata<CURRENT_METADATA_VERSION>(0,
                                       CURRENT_OPENVINO_VERSION,
                                       std::nullopt,
                                       _batchSize,
                                       std::nullopt,
                                       std::nullopt,
                                       std::nullopt,
                                       std::nullopt,
                                       compatibilityDescriptor)
        .write_as_text(requirementsString);

    _logger.debug("Runtime requirements string: %s length: %zu",
                  requirementsString.str().c_str(),
                  requirementsString.str().length());

    return requirementsString.str();
}

}  // namespace intel_npu