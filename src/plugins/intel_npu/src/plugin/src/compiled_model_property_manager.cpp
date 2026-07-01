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
        if (propertyIt == _properties.end() || !propertyIt->second.isSupported(_config)) {
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
        if (!propertyIt->second.isSupported(_config)) {
            OPENVINO_THROW("Unsupported configuration key: ", name);
        }
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

    // clang-format off
    register_property<MODEL_PRIORITY>(_config, _properties, ov::hint::model_priority.name());
    register_property<WORKLOAD_TYPE>(_config, _properties, ov::workload_type.name());

    OPENVINO_SUPPRESS_DEPRECATED_START
    register_property_as_read_only<ENABLE_CPU_PINNING>(_config, _properties, ov::hint::enable_cpu_pinning.name());
    OPENVINO_SUPPRESS_DEPRECATED_END
    register_property_as_read_only<LOG_LEVEL>(_config, _properties, ov::log::level.name());
    register_property_as_read_only<LOADED_FROM_CACHE>(_config, _properties, ov::loaded_from_cache.name());
    register_property_as_read_only<PERFORMANCE_HINT>(_config, _properties, ov::hint::performance_mode.name());
    register_property_as_read_only<EXECUTION_MODE_HINT>(_config, _properties, ov::hint::execution_mode.name());
    register_property_as_read_only<PERFORMANCE_HINT_NUM_REQUESTS>(_config, _properties, ov::hint::num_requests.name());
    register_property_as_read_only<COMPILATION_NUM_THREADS>(_config, _properties, ov::compilation_num_threads.name());
    register_property_as_read_only<INFERENCE_PRECISION_HINT>(_config, _properties, ov::hint::inference_precision.name());
    register_property_as_read_only<CACHE_MODE>(_config, _properties, ov::cache_mode.name());

    register_property_as_read_only_mark_supported_if_set<NUM_STREAMS>(_config, _properties, ov::num_streams.name());
    register_property_as_read_only_mark_supported_if_set<COMPILER_TYPE>(_config, _properties, ov::intel_npu::compiler_type.name());
    register_property_as_read_only_mark_supported_if_set<COMPILER_VERSION>(_config, _properties, ov::intel_npu::compiler_version.name());
    register_property_as_read_only_mark_supported_if_set<WEIGHTS_PATH>(_config, _properties, ov::weights_path.name());
    register_property_as_read_only_mark_supported_if_set<CACHE_DIR>(_config, _properties, ov::cache_dir.name());
    register_property_as_read_only_mark_supported_if_set<PERF_COUNT>(_config, _properties, ov::enable_profiling.name());
    register_property_as_read_only_mark_supported_if_set<PROFILING_TYPE>(_config, _properties, ov::intel_npu::profiling_type.name());
    register_property_as_read_only_mark_supported_if_set<TURBO>(_config, _properties, ov::intel_npu::turbo.name());
    register_property_as_read_only_mark_supported_if_set<COMPILATION_MODE_PARAMS>(_config, _properties, ov::intel_npu::compilation_mode_params.name());
    register_property_as_read_only_mark_supported_if_set<DMA_ENGINES>(_config, _properties, ov::intel_npu::dma_engines.name());
    register_property_as_read_only_mark_supported_if_set<TILES>(_config, _properties, ov::intel_npu::tiles.name());
    register_property_as_read_only_mark_supported_if_set<COMPILATION_MODE>(_config, _properties, ov::intel_npu::compilation_mode.name());
    register_property_as_read_only_mark_supported_if_set<PLATFORM>(_config, _properties, ov::intel_npu::platform.name());
    register_property_as_read_only_mark_supported_if_set<DYNAMIC_SHAPE_TO_STATIC>(_config, _properties, ov::intel_npu::dynamic_shape_to_static.name());
    register_property_as_read_only_mark_supported_if_set<BACKEND_COMPILATION_PARAMS>(_config, _properties, ov::intel_npu::backend_compilation_params.name());
    register_property_as_read_only_mark_supported_if_set<BYPASS_UMD_CACHING>(_config, _properties, ov::intel_npu::bypass_umd_caching.name());
    register_property_as_read_only_mark_supported_if_set<DEFER_WEIGHTS_LOAD>(_config, _properties, ov::intel_npu::defer_weights_load.name());
    register_property_as_read_only_mark_supported_if_set<COMPILER_DYNAMIC_QUANTIZATION>(_config, _properties, ov::intel_npu::compiler_dynamic_quantization.name());
    register_property_as_read_only_mark_supported_if_set<QDQ_OPTIMIZATION>(_config, _properties, ov::intel_npu::qdq_optimization.name());
    register_property_as_read_only_mark_supported_if_set<QDQ_OPTIMIZATION_AGGRESSIVE>(_config, _properties, ov::intel_npu::qdq_optimization_aggressive.name());
    register_property_as_read_only_mark_supported_if_set<DISABLE_VERSION_CHECK>(_config, _properties, ov::intel_npu::disable_version_check.name());
    register_property_as_read_only_mark_supported_if_set<EXPORT_RAW_BLOB>(_config, _properties, ov::intel_npu::export_raw_blob.name());
    register_property_as_read_only_mark_supported_if_set<IMPORT_RAW_BLOB>(_config, _properties, ov::intel_npu::import_raw_blob.name());
    register_property_as_read_only_mark_supported_if_set<BATCH_COMPILER_MODE_SETTINGS>(_config, _properties, ov::intel_npu::batch_compiler_mode_settings.name());
    register_property_as_read_only_mark_supported_if_set<RUN_INFERENCES_SEQUENTIALLY>(_config, _properties, ov::intel_npu::run_inferences_sequentially.name());
    register_property_as_read_only_mark_supported_if_set<ENABLE_WEIGHTLESS>(_config, _properties, ov::enable_weightless.name());
    register_property_as_read_only_mark_supported_if_set<SEPARATE_WEIGHTS_VERSION>(_config, _properties, ov::intel_npu::separate_weights_version.name());
    register_property_as_read_only_mark_supported_if_set<ENABLE_STRIDES_FOR>(_config, _properties, ov::intel_npu::enable_strides_for.name());
    register_property_as_read_only_mark_supported_if_set<BATCH_MODE>(_config, _properties, ov::intel_npu::batch_mode.name());
    register_property_as_read_only_mark_supported_if_set<SHARED_COMMON_QUEUE>(_config, _properties, ov::intel_npu::shared_common_queue.name());

    register_property_with_custom_function(_config, _properties, ov::cache_encryption_callbacks.name(), [](const FilteredConfig&) {
        return ov::EncryptionCallbacks{nullptr, nullptr};
    });
    register_property_with_custom_function(_properties, ov::hint::model.name(), true, [](const FilteredConfig&) {
        return std::shared_ptr<const ov::Model>(nullptr);
    });
    register_property_with_custom_function(_properties, ov::model_name.name(), true, [this](const FilteredConfig&) {
        OPENVINO_ASSERT(_graph != nullptr, "Missing graph");
        return ov::Any(_graph->get_metadata().name);
    });
    register_property_with_custom_function(_properties, ov::optimal_number_of_infer_requests.name(), true, [](const FilteredConfig& config) {
        return ov::Any(utils::getOptimalNumberOfInferRequestsInParallel(config.get<PLATFORM>(), config.get<PERFORMANCE_HINT>()));
    });
    register_property_with_custom_function(_properties, ov::execution_devices.name(), true, [](const FilteredConfig&) {
        return ov::Any(std::string("NPU"));
    });
    register_property_with_custom_function(_properties, ov::supported_properties.name(), true, [this](const FilteredConfig&) {
        std::vector<ov::PropertyName> supportedProperties;
        for (const auto& property : _properties) {
            if (property.second.isPublic && property.second.isSupported(_config)) {
                supportedProperties.emplace_back(property.first, property.second.mutability);
            }
        }
        return ov::Any(supportedProperties);
    });
    // clang-format on

    register_property_with_support_and_custom_function(
        _properties,
        ov::runtime_requirements.name(),
        [this](const FilteredConfig&) {  // support predicate
            return _graph != nullptr && _graph->get_compatibility_descriptor().has_value();
        },
        true,
        [this](const FilteredConfig&) {  // value getter
            return ov::Any(buildRuntimeRequirements(_graph, _batchSize, _logger));
        });
}

}  // namespace intel_npu
