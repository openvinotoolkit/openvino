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

    try_register_property_based_on_config<MODEL_PRIORITY>(_config, _properties, ov::hint::model_priority.name());
    try_register_property_based_on_config<WORKLOAD_TYPE>(_config, _properties, ov::workload_type.name());

    // clang-format off
    OPENVINO_SUPPRESS_DEPRECATED_START
    try_register_property_based_on_config_as_read_only<ENABLE_CPU_PINNING>(_config, _properties, ov::hint::enable_cpu_pinning.name());
    OPENVINO_SUPPRESS_DEPRECATED_END
    try_register_property_based_on_config_as_read_only<LOG_LEVEL>(_config, _properties, ov::log::level.name());
    try_register_property_based_on_config_as_read_only<LOADED_FROM_CACHE>(_config, _properties, ov::loaded_from_cache.name());
    try_register_property_based_on_config_as_read_only<PERFORMANCE_HINT>(_config, _properties, ov::hint::performance_mode.name());
    try_register_property_based_on_config_as_read_only<EXECUTION_MODE_HINT>(_config, _properties, ov::hint::execution_mode.name());
    try_register_property_based_on_config_as_read_only<PERFORMANCE_HINT_NUM_REQUESTS>(_config, _properties, ov::hint::num_requests.name());
    try_register_property_based_on_config_as_read_only<COMPILATION_NUM_THREADS>(_config, _properties, ov::compilation_num_threads.name());
    try_register_property_based_on_config_as_read_only<INFERENCE_PRECISION_HINT>(_config, _properties, ov::hint::inference_precision.name());
    try_register_property_based_on_config_as_read_only<CACHE_MODE>(_config, _properties, ov::cache_mode.name());

    try_register_property_based_on_config_if_set_as_read_only<NUM_STREAMS>(_config, _properties, ov::num_streams.name());
    try_register_property_based_on_config_if_set_as_read_only<COMPILER_TYPE>(_config, _properties, ov::intel_npu::compiler_type.name());
    try_register_property_based_on_config_if_set_as_read_only<COMPILER_VERSION>(_config, _properties, ov::intel_npu::compiler_version.name());
    try_register_property_based_on_config_if_set_as_read_only<WEIGHTS_PATH>(_config, _properties, ov::weights_path.name());
    try_register_property_based_on_config_if_set_as_read_only<CACHE_DIR>(_config, _properties, ov::cache_dir.name());
    try_register_property_based_on_config_if_set_as_read_only<PERF_COUNT>(_config, _properties, ov::enable_profiling.name());
    try_register_property_based_on_config_if_set_as_read_only<PROFILING_TYPE>(_config, _properties, ov::intel_npu::profiling_type.name());
    try_register_property_based_on_config_if_set_as_read_only<TURBO>(_config, _properties, ov::intel_npu::turbo.name());
    try_register_property_based_on_config_if_set_as_read_only<COMPILATION_MODE_PARAMS>(_config, _properties, ov::intel_npu::compilation_mode_params.name());
    try_register_property_based_on_config_if_set_as_read_only<DMA_ENGINES>(_config, _properties, ov::intel_npu::dma_engines.name());
    try_register_property_based_on_config_if_set_as_read_only<TILES>(_config, _properties, ov::intel_npu::tiles.name());
    try_register_property_based_on_config_if_set_as_read_only<COMPILATION_MODE>(_config, _properties, ov::intel_npu::compilation_mode.name());
    try_register_property_based_on_config_if_set_as_read_only<PLATFORM>(_config, _properties, ov::intel_npu::platform.name());
    try_register_property_based_on_config_if_set_as_read_only<DYNAMIC_SHAPE_TO_STATIC>(_config, _properties, ov::intel_npu::dynamic_shape_to_static.name());
    try_register_property_based_on_config_if_set_as_read_only<BACKEND_COMPILATION_PARAMS>(_config, _properties, ov::intel_npu::backend_compilation_params.name());
    try_register_property_based_on_config_if_set_as_read_only<BYPASS_UMD_CACHING>(_config, _properties, ov::intel_npu::bypass_umd_caching.name());
    try_register_property_based_on_config_if_set_as_read_only<DEFER_WEIGHTS_LOAD>(_config, _properties, ov::intel_npu::defer_weights_load.name());
    try_register_property_based_on_config_if_set_as_read_only<COMPILER_DYNAMIC_QUANTIZATION>(_config, _properties, ov::intel_npu::compiler_dynamic_quantization.name());
    try_register_property_based_on_config_if_set_as_read_only<QDQ_OPTIMIZATION>(_config, _properties, ov::intel_npu::qdq_optimization.name());
    try_register_property_based_on_config_if_set_as_read_only<QDQ_OPTIMIZATION_AGGRESSIVE>(_config, _properties, ov::intel_npu::qdq_optimization_aggressive.name());
    try_register_property_based_on_config_if_set_as_read_only<DISABLE_VERSION_CHECK>(_config, _properties, ov::intel_npu::disable_version_check.name());
    try_register_property_based_on_config_if_set_as_read_only<EXPORT_RAW_BLOB>(_config, _properties, ov::intel_npu::export_raw_blob.name());
    try_register_property_based_on_config_if_set_as_read_only<IMPORT_RAW_BLOB>(_config, _properties, ov::intel_npu::import_raw_blob.name());
    try_register_property_based_on_config_if_set_as_read_only<BATCH_COMPILER_MODE_SETTINGS>(_config, _properties, ov::intel_npu::batch_compiler_mode_settings.name());
    try_register_property_based_on_config_if_set_as_read_only<RUN_INFERENCES_SEQUENTIALLY>(_config, _properties, ov::intel_npu::run_inferences_sequentially.name());
    try_register_property_based_on_config_if_set_as_read_only<ENABLE_WEIGHTLESS>(_config, _properties, ov::enable_weightless.name());
    try_register_property_based_on_config_if_set_as_read_only<SEPARATE_WEIGHTS_VERSION>(_config, _properties, ov::intel_npu::separate_weights_version.name());
    try_register_property_based_on_config_if_set_as_read_only<ENABLE_STRIDES_FOR>(_config, _properties, ov::intel_npu::enable_strides_for.name());
    try_register_property_based_on_config_if_set_as_read_only<BATCH_MODE>(_config, _properties, ov::intel_npu::batch_mode.name());
    try_register_property_based_on_config_if_set_as_read_only<SHARED_COMMON_QUEUE>(_config, _properties, ov::intel_npu::shared_common_queue.name());

    try_register_property_based_on_config_with_custom_function(_config, _properties, ov::cache_encryption_callbacks.name(), [](const Config&) {
        return ov::EncryptionCallbacks{nullptr, nullptr};
    });

    register_property_with_custom_function(_properties, ov::hint::model.name(), true, [](const Config&) {
        return std::shared_ptr<const ov::Model>(nullptr);
    });
    register_property_with_custom_function(_properties, ov::model_name.name(), true, [this](const Config&) {
        OPENVINO_ASSERT(_graph != nullptr, "Missing graph");
        return ov::Any(_graph->get_metadata().name);
    });
    register_property_with_custom_function(_properties, ov::optimal_number_of_infer_requests.name(), true, [](const Config& config) {
        return ov::Any(utils::getOptimalNumberOfInferRequestsInParallel(config.get<PLATFORM>(), config.get<PERFORMANCE_HINT>()));
    });
    register_property_with_custom_function(_properties, ov::execution_devices.name(), true, [](const Config&) {
        return ov::Any(std::string("NPU"));
    });
    register_property_with_custom_function(_properties, ov::supported_properties.name(), true, [this](const Config&) {
        return ov::Any(_supportedProperties);
    });
    // clang-format on

    try_register_property_with_custom_function(
        _properties,
        ov::runtime_requirements.name(),
        _graph != nullptr && _graph->get_compatibility_descriptor().has_value(),
        true,
        [this](const Config&) {
            return ov::Any(buildRuntimeRequirements(_graph, _batchSize, _logger));
        });

    for (const auto& property : _properties) {
        if (property.second.isPublic) {
            _supportedProperties.emplace_back(property.first, property.second.mutability);
        }
    }
}

}  // namespace intel_npu
