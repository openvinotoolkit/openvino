// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model_property_manager.hpp"

#include <sstream>

#include "intel_npu/common/device_helpers.hpp"
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
    try_register_compiled_model_property<ENABLE_CPU_PINNING>(_config, _properties, ov::hint::enable_cpu_pinning);
    OPENVINO_SUPPRESS_DEPRECATED_END
    try_register_compiled_model_property<LOG_LEVEL>(_config, _properties, ov::log::level);
    try_register_compiled_model_property<LOADED_FROM_CACHE>(_config, _properties, ov::loaded_from_cache);
    try_register_compiled_model_property<PERFORMANCE_HINT>(_config, _properties, ov::hint::performance_mode);
    try_register_compiled_model_property<EXECUTION_MODE_HINT>(_config, _properties, ov::hint::execution_mode);
    try_register_compiled_model_property<PERFORMANCE_HINT_NUM_REQUESTS>(_config, _properties, ov::hint::num_requests);
    try_register_compiled_model_property<COMPILATION_NUM_THREADS>(_config, _properties, ov::compilation_num_threads);
    try_register_compiled_model_property<INFERENCE_PRECISION_HINT>(_config, _properties, ov::hint::inference_precision);
    try_register_compiled_model_property<CACHE_MODE>(_config, _properties, ov::cache_mode);

    try_register_compiled_model_property_ifset<NUM_STREAMS>(_config, _properties, ov::num_streams);
    try_register_compiled_model_property_ifset<COMPILER_TYPE>(_config, _properties, ov::intel_npu::compiler_type);
    try_register_compiled_model_property_ifset<COMPILER_VERSION>(_config, _properties, ov::intel_npu::compiler_version);
    try_register_compiled_model_property_ifset<WEIGHTS_PATH>(_config, _properties, ov::weights_path);
    try_register_compiled_model_property_ifset<CACHE_DIR>(_config, _properties, ov::cache_dir);
    try_register_compiled_model_property_ifset<PERF_COUNT>(_config, _properties, ov::enable_profiling);
    try_register_compiled_model_property_ifset<PROFILING_TYPE>(_config, _properties, ov::intel_npu::profiling_type);
    try_register_compiled_model_property_ifset<TURBO>(_config, _properties, ov::intel_npu::turbo);
    try_register_compiled_model_property_ifset<COMPILATION_MODE_PARAMS>(_config,
                                                                        _properties,
                                                                        ov::intel_npu::compilation_mode_params);
    try_register_compiled_model_property_ifset<DMA_ENGINES>(_config, _properties, ov::intel_npu::dma_engines);
    try_register_compiled_model_property_ifset<TILES>(_config, _properties, ov::intel_npu::tiles);
    try_register_compiled_model_property_ifset<COMPILATION_MODE>(_config, _properties, ov::intel_npu::compilation_mode);
    try_register_compiled_model_property_ifset<PLATFORM>(_config, _properties, ov::intel_npu::platform);
    try_register_compiled_model_property_ifset<DYNAMIC_SHAPE_TO_STATIC>(_config,
                                                                        _properties,
                                                                        ov::intel_npu::dynamic_shape_to_static);
    try_register_compiled_model_property_ifset<BACKEND_COMPILATION_PARAMS>(_config,
                                                                           _properties,
                                                                           ov::intel_npu::backend_compilation_params);
    try_register_compiled_model_property_ifset<BYPASS_UMD_CACHING>(_config,
                                                                   _properties,
                                                                   ov::intel_npu::bypass_umd_caching);
    try_register_compiled_model_property_ifset<DEFER_WEIGHTS_LOAD>(_config,
                                                                   _properties,
                                                                   ov::intel_npu::defer_weights_load);
    try_register_compiled_model_property_ifset<COMPILER_DYNAMIC_QUANTIZATION>(
        _config,
        _properties,
        ov::intel_npu::compiler_dynamic_quantization);
    try_register_compiled_model_property_ifset<QDQ_OPTIMIZATION>(_config, _properties, ov::intel_npu::qdq_optimization);
    try_register_compiled_model_property_ifset<QDQ_OPTIMIZATION_AGGRESSIVE>(_config,
                                                                            _properties,
                                                                            ov::intel_npu::qdq_optimization_aggressive);
    try_register_compiled_model_property_ifset<DISABLE_VERSION_CHECK>(_config,
                                                                      _properties,
                                                                      ov::intel_npu::disable_version_check);
    try_register_compiled_model_property_ifset<EXPORT_RAW_BLOB>(_config, _properties, ov::intel_npu::export_raw_blob);
    try_register_compiled_model_property_ifset<IMPORT_RAW_BLOB>(_config, _properties, ov::intel_npu::import_raw_blob);
    try_register_compiled_model_property_ifset<BATCH_COMPILER_MODE_SETTINGS>(
        _config,
        _properties,
        ov::intel_npu::batch_compiler_mode_settings);
    try_register_compiled_model_property_ifset<RUN_INFERENCES_SEQUENTIALLY>(_config,
                                                                            _properties,
                                                                            ov::intel_npu::run_inferences_sequentially);
    try_register_compiled_model_property_ifset<ENABLE_WEIGHTLESS>(_config, _properties, ov::enable_weightless);
    try_register_compiled_model_property_ifset<SEPARATE_WEIGHTS_VERSION>(_config,
                                                                         _properties,
                                                                         ov::intel_npu::separate_weights_version);
    try_register_compiled_model_property_ifset<ENABLE_STRIDES_FOR>(_config,
                                                                   _properties,
                                                                   ov::intel_npu::enable_strides_for);

    try_register_custom_property(_config,
                                 _properties,
                                 ov::intel_npu::batch_mode,
                                 false,
                                 ov::PropertyMutability::RO,
                                 [](const Config& config) {
                                     return config.get<BATCH_MODE>();
                                 });
    try_register_custom_property(_config,
                                 _properties,
                                 ov::intel_npu::shared_common_queue,
                                 false,
                                 ov::PropertyMutability::RO,
                                 [](const Config& config) {
                                     return config.get<SHARED_COMMON_QUEUE>();
                                 });
    try_register_custom_property(_config,
                                 _properties,
                                 ov::hint::model_priority,
                                 true,
                                 ov::PropertyMutability::RW,
                                 [](const Config& config) {
                                     return config.get<MODEL_PRIORITY>();
                                 });
    try_register_custom_property(_config,
                                 _properties,
                                 ov::workload_type,
                                 true,
                                 ov::PropertyMutability::RW,
                                 [](const Config& config) {
                                     return config.get<WORKLOAD_TYPE>();
                                 });
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
    register_custom_metric(_properties, ov::model_name, true, [this](const Config&) {
        OPENVINO_ASSERT(_graph != nullptr, "Missing graph");
        return ov::Any(_graph->get_metadata().name);
    });
    register_custom_metric(_properties, ov::optimal_number_of_infer_requests, true, [](const Config& config) {
        return ov::Any(
            utils::getOptimalNumberOfInferRequestsInParallel(config.get<PLATFORM>(), config.get<PERFORMANCE_HINT>()));
    });
    register_custom_metric(_properties, ov::execution_devices, true, [](const Config&) {
        return ov::Any(std::string("NPU"));
    });
    if (_config.isAvailable(ov::runtime_requirements.name())) {
        register_custom_metric(_properties, ov::runtime_requirements, true, [this](const Config&) {
            return ov::Any(buildRuntimeRequirements());
        });
    }
    register_custom_metric(_properties, ov::supported_properties, true, [this](const Config&) {
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
