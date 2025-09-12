// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Plugin
#include "properties.hpp"

#include "compiler_adapter_factory.hpp"
#include "intel_npu/common/device_helpers.hpp"
#include "intel_npu/config/npuw.hpp"
#include "intel_npu/config/options.hpp"

namespace intel_npu {

//
// Helper macro functions to ease registration of properties (both from configs and from metrics)
//

/**
 * @brief Macro for registering simple get<> properties which have everything defined in their optionBase.
 *
 * This macro can be used for registering properties which have entry in the optionsDesc table and
 * requre a simple globalconfig.get<TEMPLATE>() callback for value, without any extra data manipulation
 * It will check if the configuration option is available in the globalconfig and if it exists,
 * then registers it as a property in the internal property map. It handles
 * public/private property visibility and the mutability of the property
 * based on the configuration from the optionBase type.
 *
 * @param OPT_NAME Class/type of the option (will fetch .name() from it)
 * @param OPT_TYPE Type (template) of the option
 *
 * @details
 * - It first checks if the option was registered in the global config. Options not present in the global config
 *   are not supported in the current configuration and were previously filtered out on plugin level.
 * - The property visibility (public/private) and mutability (RO/RW) are read from the base option descriptor
 * (optionBase) the property will map to
 * - COMPILED_MODEL only: In case the property is registered for a compiled_model, mutability will be automatically
 * forced to READ-ONLY
 * - COMPILED_MODEL only: If the configuration has no entry for the option, it means it was not set, which means it will
 * skip registering it
 * - A simple config.get<OPT_TYPE> lambda function is defined as the property's callback function
 *
 * @note The macro ensures that compiled model properties are marked read-only unless the configuration lacks the
 * specified option type.
 */
#define TRY_REGISTER_SIMPLE_PROPERTY(OPT_NAME, OPT_TYPE)                                                \
    do {                                                                                                \
        std::string o_name = OPT_NAME.name();                                                           \
        if (_config.isAvailable(o_name)) {                                                              \
            bool isPublic = _config.getOpt(o_name).isPublic();                                          \
            ov::PropertyMutability isMutable = _config.getOpt(o_name).mutability();                     \
            if (_pType == PropertiesType::COMPILED_MODEL) {                                             \
                isMutable = ov::PropertyMutability::RO;                                                 \
            }                                                                                           \
            _properties.emplace(o_name, std::make_tuple(isPublic, isMutable, [](const Config& config) { \
                                    return config.get<OPT_TYPE>();                                      \
                                }));                                                                    \
        }                                                                                               \
    } while (0)

/**
 * @brief Macro for registering simple get<> properties which have been previously set. For COMPILED_MODEL
 *
 * This macro is the same as TRY_REGISTER_SIMPLE_PROPERTY with the added functionality that it first verifies
 * whether the option was previously set + forces the property to  be public (!!). Basicly it will not register
 * properties which are still on their default values. This is useful for Compiled_model properties to avoid reporting
 * settings which were not set at model compilation. Having properties in compiled_model which report default value can
 * be misleading as those default values might not even haven been used by compiler, or in extreme cornercases be out of
 * sync with the compiler's default values.
 *
 * @param OPT_NAME Class/type of the option (will fetch .name() from it)
 * @param OPT_TYPE Type (template) of the option
 *
 * @details
 * - It first checks if the config has a previusly set value for this key. If there isn't any value set for this key,
 *   it will skip registration
 * - Then checks if the option was registered in the global config. Options not present in the global config
 *   are not supported in the current configuration and were previously filtered out on plugin level.
 * - The property visibility (public/private) and mutability (RO/RW) are read from the base option descriptor
 * (optionBase) the property will map to
 * - mutability will be automatically forced to READ-ONLY
 * - visibility will be automatically forced to PUBLIC
 * - If the configuration has no entry for the option, it means it was not set, which means it will
 * skip registering it
 * - A simple config.get<OPT_TYPE> lambda function is defined as the property's callback function
 *
 * @note The macro ensures that compiled model properties are marked read-only unless the configuration lacks the
 * specified option type.
 * @note ***** TO BE USED FOR COMPILED_MODEL ONLY ***** It forces the property public!
 */
#define TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(OPT_NAME, OPT_TYPE)                                   \
    do {                                                                                                \
        std::string o_name = OPT_NAME.name();                                                           \
        if (!_config.has(o_name)) {                                                                     \
            break;                                                                                      \
        }                                                                                               \
        if (_config.isAvailable(o_name)) {                                                              \
            bool isPublic = _config.getOpt(o_name).isPublic();                                          \
            ov::PropertyMutability isMutable = _config.getOpt(o_name).mutability();                     \
            if (_pType == PropertiesType::COMPILED_MODEL) {                                             \
                isMutable = ov::PropertyMutability::RO;                                                 \
                isPublic = true;                                                                        \
            }                                                                                           \
            _properties.emplace(o_name, std::make_tuple(isPublic, isMutable, [](const Config& config) { \
                                    return config.get<OPT_TYPE>();                                      \
                                }));                                                                    \
        }                                                                                               \
    } while (0)

/**
 * @brief Macro for defining otherwise simple get<> properties but which have variable public/private field
 *
 * This macro offers the same functionality as TRY_REGISTER_SIMPLE_PROPERTY (see its description for more)
 * with the extra feature that enforces plugin property visibility (public/private) given by user.
 * Compiled model properties will still be forced to READ-ONLY
 *
 * @param OPT_NAME Class/type of the option (will fetch .name() from it)
 * @param OPT_TYPE Type (template) of the option
 * @param PROP_VISIBILITY Visibility (true=public, false=private) of the resulting property
 *
 * @details
 * @see TRY_REGISTER_SIMPLE_PROPERTY
 */
#define TRY_REGISTER_VARPUB_PROPERTY(OPT_NAME, OPT_TYPE, PROP_VISIBILITY)                                      \
    do {                                                                                                       \
        std::string o_name = OPT_NAME.name();                                                                  \
        if (_config.isAvailable(o_name)) {                                                                     \
            ov::PropertyMutability isMutable = _config.getOpt(o_name).mutability();                            \
            if (_pType == PropertiesType::COMPILED_MODEL) {                                                    \
                isMutable = ov::PropertyMutability::RO;                                                        \
            }                                                                                                  \
            _properties.emplace(o_name, std::make_tuple(PROP_VISIBILITY, isMutable, [](const Config& config) { \
                                    return config.get<OPT_TYPE>();                                             \
                                }));                                                                           \
        }                                                                                                      \
    } while (0)

/**
 * @brief Macro for registering properties which need a custom return function
 *
 * This macro offers the same functionality as TRY_REGISTER_SIMPLE_PROPERTY (see its description for more)
 * but with a custom callback function.
 *
 * @param OPT_NAME Class/type of the option (will fetch .name() from it)
 * @param OPT_TYPE Type (template) of the option
 * @param PROP_RETFUNC Custom lambda callback function for the resulting property
 *
 * @details
 * @see TRY_REGISTER_SIMPLE_PROPERTY
 */
#define TRY_REGISTER_CUSTOMFUNC_PROPERTY(OPT_NAME, OPT_TYPE, PROP_RETFUNC)                   \
    do {                                                                                     \
        std::string o_name = OPT_NAME.name();                                                \
        if (_config.isAvailable(o_name)) {                                                   \
            bool isPublic = _config.getOpt(o_name).isPublic();                               \
            ov::PropertyMutability isMutable = _config.getOpt(o_name).mutability();          \
            if (_pType == PropertiesType::COMPILED_MODEL) {                                  \
                isMutable = ov::PropertyMutability::RO;                                      \
            }                                                                                \
            _properties.emplace(o_name, std::make_tuple(isPublic, isMutable, PROP_RETFUNC)); \
        }                                                                                    \
    } while (0)

/**
 * @brief Macro for registering a fully custom property, with option entry validation
 *
 * This macro offers the flexibility of registering a fully custom property where all its parameters
 * are user provided: visibility (public/private), mutability and callback function
 *
 * @param OPT_NAME Class/type of the option (will fetch .name() from it)
 * @param OPT_TYPE Type (template) of the option
 * @param PROP_VISIBILITY Visibility (true=public, false=private) of the resulting property
 * @param PROP_MUTABILITY Mutability (RO/RW) of the resulting property
 * @param PROP_RETFUNC Custom lambda callback function for the resulting property
 *
 * @details
 * - It first checks if the option was registered in the global config. Options not present in the global config
 *   are not supported in the current configuration and were previously filtered out on plugin level.
 * - A new entry is added to the properties table with the user provided parameters (visibility,mutability,callback)
 *   without any further checks and validations
 *
 * @note This macro does not offer any compiled-model specific checks, such as
 * if the config options this property maps to has actual value, nor it enforces RO, like previous macros.
 */
#define TRY_REGISTER_CUSTOM_PROPERTY(OPT_NAME, OPT_TYPE, PROP_VISIBILITY, PROP_MUTABILITY, PROP_RETFUNC)  \
    do {                                                                                                  \
        std::string o_name = OPT_NAME.name();                                                             \
        if (_config.isAvailable(o_name)) {                                                                \
            _properties.emplace(o_name, std::make_tuple(PROP_VISIBILITY, PROP_MUTABILITY, PROP_RETFUNC)); \
        }                                                                                                 \
    } while (0)

/**
 * @brief Macro for force registering a fully custom property (no option entry validation.
 *
 * Same as TRY_REGISTER_CUSTOM_PROPERTY but without any checks. It will force register the property.
 *
 * @param OPT_NAME Class/type of the option (will fetch .name() from it)
 * @param OPT_TYPE Type (template) of the option
 * @param PROP_VISIBILITY Visibility (true=public, false=private) of the resulting property
 * @param PROP_MUTABILITY Mutability (RO/RW) of the resulting property
 * @param PROP_RETFUNC Custom lambda callback function for the resulting property
 *
 * @details
 * - A new entry is added to the properties table with the user provided parameters (visibility,mutability,callback)
 *   without any checks and validations
 *
 * @note This macro does not offer any compiled-model specific checks, such as
 * if the config options this property maps to has actual value, nor it enforces RO, like previous macros.
 */
#define FORCE_REGISTER_CUSTOM_PROPERTY(OPT_NAME, OPT_TYPE, PROP_VISIBILITY, PROP_MUTABILITY, PROP_RETFUNC)     \
    do {                                                                                                       \
        _properties.emplace(OPT_NAME.name(), std::make_tuple(PROP_VISIBILITY, PROP_MUTABILITY, PROP_RETFUNC)); \
    } while (0)

/**
 * @brief Macro for defining properties which have simmple single value returning metrics
 *
 * The key differentiator for Metrics (from configs) is that they don't have an entry in the config map, nor an
 * OptionBase descriptor. Metrics are static, Read-Only properties returning fixed characteristics of the device, plugin
 * or environment. Since they don't have an entry in the config map, nor an optionBase descriptor, their callback
 * function returns custom data.
 *
 * @param PROP_NAME Class/type of the property (will fetch .name() from it)
 * @param PROP_VISIBILITY Visibility (true=public, false=private) of the resulting property
 * @param PROP_RETVAL Value for the callback function to return
 *
 * @details
 * - A new entry is added to the properties table with the user provided parameters (visibility,mutability,callback val)
 *   without any checks and validations
 *
 * @note This macro does not offer any compiled-model specific checks
 */
#define REGISTER_SIMPLE_METRIC(PROP_NAME, PROP_VISIBILITY, PROP_RETVAL)                                      \
    do {                                                                                                     \
        _properties.emplace(                                                                                 \
            PROP_NAME.name(),                                                                                \
            std::make_tuple(PROP_VISIBILITY, ov::PropertyMutability::RO, [&](const Config& config) -> auto { \
                return PROP_RETVAL;                                                                          \
            }));                                                                                             \
    } while (0)

/**
 * @brief Macro for defining metric properties with full callback lambda functions
 *
 * The difference from REGISTER_SIMPLE_METRIC is that here we can define the whole callback lambdafunction, not just the
 * return value
 *
 * @param PROP_NAME Class/type of the property (will fetch .name() from it)
 * @param PROP_VISIBILITY Visibility (true=public, false=private) of the resulting property
 * @param PROP_RETFUNC Callback lambda function of the resulting property
 *
 * @details
 * - A new entry is added to the properties table with the user provided parameters (visibility,mutability,callback
 * func) without any checks and validations
 *
 * @note This macro does not offer any compiled-model specific checks
 */
// Macro for defining metrics with custom return function
#define REGISTER_CUSTOM_METRIC(PROP_NAME, PROP_VISIBILITY, PROP_RETFUNC)                                 \
    do {                                                                                                 \
        _properties.emplace(PROP_NAME.name(),                                                            \
                            std::make_tuple(PROP_VISIBILITY, ov::PropertyMutability::RO, PROP_RETFUNC)); \
    } while (0)

// Local helper function for appending platform name to the config
static Config add_platform_to_the_config(Config config, const std::string_view platform) {
    config.update({{ov::intel_npu::platform.name(), std::string(platform)}});
    return config;
}

// Local helper function for retrieving the device name
static auto get_specified_device_name(const Config config) {
    if (config.has<DEVICE_ID>()) {
        return config.get<DEVICE_ID>();
    }
    return std::string();
}

// Heuristically obtained number. Varies depending on the values of PLATFORM and PERFORMANCE_HINT
// Note: this is the value provided by the plugin, application should query and consider it, but may supply its own
// preference for number of parallel requests via dedicated configuration
static int64_t getOptimalNumberOfInferRequestsInParallel(const Config& config) {
    const std::string platform = ov::intel_npu::Platform::standardize(config.get<PLATFORM>());

    if (platform == ov::intel_npu::Platform::NPU3720) {
        if (config.get<PERFORMANCE_HINT>() == ov::hint::PerformanceMode::THROUGHPUT) {
            return 4;
        } else {
            return 1;
        }
    } else {
        if (config.get<PERFORMANCE_HINT>() == ov::hint::PerformanceMode::THROUGHPUT) {
            return 8;
        } else {
            return 1;
        }
    }
}

Properties::Properties(const PropertiesType pType,
                       FilteredConfig& config,
                       const std::shared_ptr<Metrics>& metrics,
                       const ov::SoPtr<IEngineBackend>& backend)
    : _pType(pType),
      _config(config),
      _metrics(metrics),
      _backend(backend) {}

void Properties::registerProperties() {
    // Reset
    _properties.clear();

    switch (_pType) {
    case PropertiesType::PLUGIN:
        registerPluginProperties();
        break;
    case PropertiesType::COMPILED_MODEL:
        registerCompiledModelProperties();
        break;
    default:
        OPENVINO_THROW("Invalid plugin configuration!");
        break;
    }

    // 2.3. Common metrics (exposed same way by both Plugin and CompiledModel)
    REGISTER_SIMPLE_METRIC(ov::supported_properties, true, _supportedProperties);

    // 3. Populate supported properties list
    // ========
    for (auto& property : _properties) {
        if (std::get<0>(property.second)) {
            _supportedProperties.emplace_back(ov::PropertyName(property.first, std::get<1>(property.second)));
        }
    }
}

void Properties::registerPluginProperties() {
    // 1. Configs
    // ========
    // 1.1 simple configs which only return value
    // TRY_REGISTER_SIMPLE_PROPERTY format: (property, config_to_return)
    // TRY_REGISTER_VARPUB_PROPERTY format: (property, config_to_return, dynamic public/private value)
    // TRY_REGISTER_CUSTOMFUNC_PROPERTY format: (property, custom_return_lambda_function)
    // TRY_REGISTER_CUSTOM_PROPERTY format: (property, visibility, mutability, custom_return_lambda_function)
    // FORCE_REGISTER_CUSTOM_PROPERTY format: (property, visibility, mutability, custom_return_lambda_function)
    TRY_REGISTER_SIMPLE_PROPERTY(ov::enable_profiling, PERF_COUNT);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::hint::performance_mode, PERFORMANCE_HINT);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::hint::execution_mode, EXECUTION_MODE_HINT);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::hint::num_requests, PERFORMANCE_HINT_NUM_REQUESTS);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::compilation_num_threads, COMPILATION_NUM_THREADS);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::hint::inference_precision, INFERENCE_PRECISION_HINT);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::log::level, LOG_LEVEL);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::cache_dir, CACHE_DIR);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::cache_mode, CACHE_MODE);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::hint::compiled_blob, COMPILED_BLOB);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::device::id, DEVICE_ID);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::num_streams, NUM_STREAMS);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::weights_path, WEIGHTS_PATH);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::internal::exclusive_async_requests, EXCLUSIVE_ASYNC_REQUESTS);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::compilation_mode_params, COMPILATION_MODE_PARAMS);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::dma_engines, DMA_ENGINES);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::tiles, TILES);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::compilation_mode, COMPILATION_MODE);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::compiler_type, COMPILER_TYPE);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::platform, PLATFORM);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::create_executor, CREATE_EXECUTOR);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::dynamic_shape_to_static, DYNAMIC_SHAPE_TO_STATIC);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::profiling_type, PROFILING_TYPE);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::backend_compilation_params, BACKEND_COMPILATION_PARAMS);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::batch_mode, BATCH_MODE);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::turbo, TURBO);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::hint::model_priority, MODEL_PRIORITY);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::bypass_umd_caching, BYPASS_UMD_CACHING);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::defer_weights_load, DEFER_WEIGHTS_LOAD);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::compiler_dynamic_quantization, COMPILER_DYNAMIC_QUANTIZATION);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::qdq_optimization, QDQ_OPTIMIZATION);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::qdq_optimization_aggressive, QDQ_OPTIMIZATION_AGGRESSIVE);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::disable_version_check, DISABLE_VERSION_CHECK);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::batch_compiler_mode_settings, BATCH_COMPILER_MODE_SETTINGS);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::hint::enable_cpu_pinning, ENABLE_CPU_PINNING);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::workload_type, WORKLOAD_TYPE);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::weightless_blob, WEIGHTLESS_BLOB);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::separate_weights_version, SEPARATE_WEIGHTS_VERSION);

    TRY_REGISTER_CUSTOMFUNC_PROPERTY(ov::intel_npu::stepping, STEPPING, [&](const Config& config) {
        if (!config.has<STEPPING>()) {
            const auto specifiedDeviceName = get_specified_device_name(config);
            return static_cast<int64_t>(_metrics->GetSteppingNumber(specifiedDeviceName));
        } else {
            return config.get<STEPPING>();
        }
    });
    TRY_REGISTER_CUSTOMFUNC_PROPERTY(ov::intel_npu::max_tiles, MAX_TILES, [&](const Config& config) {
        if (!config.has<MAX_TILES>()) {
            const auto specifiedDeviceName = get_specified_device_name(config);
            return static_cast<int64_t>(_metrics->GetMaxTiles(specifiedDeviceName));
        } else {
            return config.get<MAX_TILES>();
        }
    });

    TRY_REGISTER_VARPUB_PROPERTY(ov::intel_npu::run_inferences_sequentially, RUN_INFERENCES_SEQUENTIALLY, [&] {
        if (_backend && _backend->getInitStructs()) {
            if (_backend->getInitStructs()->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 1)) {
                return true;
            }
        }
        return false;
    }());
    TRY_REGISTER_SIMPLE_PROPERTY(ov::hint::enable_cpu_pinning, ENABLE_CPU_PINNING);

    // NPUW properties are requested by OV Core during caching and have no effect on the NPU plugin. But we still need
    // to enable those for OV Core to query.
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::use_npuw, NPU_USE_NPUW);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::devices, NPUW_DEVICES);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::submodel_device, NPUW_SUBMODEL_DEVICE);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::weights_bank, NPUW_WEIGHTS_BANK);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::weights_bank_alloc, NPUW_WEIGHTS_BANK_ALLOC);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::partitioning::online::pipeline, NPUW_ONLINE_PIPELINE);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::partitioning::online::avoid, NPUW_ONLINE_AVOID);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::partitioning::online::isolate, NPUW_ONLINE_ISOLATE);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::partitioning::online::nofold, NPUW_ONLINE_NO_FOLD);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::partitioning::online::min_size, NPUW_ONLINE_MIN_SIZE);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::partitioning::online::keep_blocks, NPUW_ONLINE_KEEP_BLOCKS);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::partitioning::online::keep_block_size,
                                 NPUW_ONLINE_KEEP_BLOCK_SIZE);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::partitioning::fold, NPUW_FOLD);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::partitioning::cwai, NPUW_CWAI);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::partitioning::dyn_quant, NPUW_DQ);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::partitioning::dyn_quant_full, NPUW_DQ_FULL);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::partitioning::par_matmul_merge_dims, NPUW_PMM);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::partitioning::slice_out, NPUW_SLICE_OUT);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::partitioning::spatial, NPUW_SPATIAL);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::partitioning::spatial_nway, NPUW_SPATIAL_NWAY);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::partitioning::spatial_dyn, NPUW_SPATIAL_DYN);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::partitioning::f16_interconnect, NPUW_F16IC);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::partitioning::host_gather, NPUW_HOST_GATHER);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::partitioning::dcoff_type, NPUW_DCOFF_TYPE);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::partitioning::dcoff_with_scale, NPUW_DCOFF_SCALE);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::partitioning::funcall_for_all, NPUW_FUNCALL_FOR_ALL);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::funcall_async, NPUW_FUNCALL_ASYNC);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::unfold_ireqs, NPUW_UNFOLD_IREQS);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::llm::enabled, NPUW_LLM);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::llm::batch_dim, NPUW_LLM_BATCH_DIM);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::llm::seq_len_dim, NPUW_LLM_SEQ_LEN_DIM);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::llm::max_prompt_len, NPUW_LLM_MAX_PROMPT_LEN);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::llm::max_generation_token_len, NPUW_LLM_MAX_GENERATION_TOKEN_LEN);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::llm::min_response_len, NPUW_LLM_MIN_RESPONSE_LEN);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::llm::optimize_v_tensors, NPUW_LLM_OPTIMIZE_V_TENSORS);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::llm::cache_rope, NPUW_LLM_CACHE_ROPE);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::llm::prefill_chunk_size, NPUW_LLM_PREFILL_CHUNK_SIZE);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::llm::shared_lm_head, NPUW_LLM_SHARED_HEAD);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::llm::max_lora_rank, NPUW_LLM_MAX_LORA_RANK);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::llm::prefill_hint, NPUW_LLM_PREFILL_HINT);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::llm::prefill_config, NPUW_LLM_PREFILL_CONFIG);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::llm::additional_prefill_config,
                                 NPUW_LLM_ADDITIONAL_PREFILL_CONFIG);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::llm::generate_hint, NPUW_LLM_GENERATE_HINT);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::llm::generate_config, NPUW_LLM_GENERATE_CONFIG);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::llm::additional_generate_config,
                                 NPUW_LLM_ADDITIONAL_GENERATE_CONFIG);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::llm::shared_lm_head_config, NPUW_LLM_SHARED_LM_HEAD_CONFIG);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::npuw::llm::additional_shared_lm_head_config,
                                 NPUW_LLM_ADDITIONAL_SHARED_LM_HEAD_CONFIG);

    // 2. Metrics (static device and enviroment properties)
    // ========
    // REGISTER_SIMPLE_METRIC format: (property, public true/false, return value)
    // REGISTER_CUSTOM_METRIC format: (property, public true/false, return value function)
    if (_metrics != nullptr) {
        REGISTER_SIMPLE_METRIC(ov::available_devices, true, _metrics->GetAvailableDevicesNames());
        REGISTER_SIMPLE_METRIC(ov::device::capabilities, true, _metrics->GetOptimizationCapabilities());
        REGISTER_SIMPLE_METRIC(
            ov::optimal_number_of_infer_requests,
            true,
            static_cast<uint32_t>(getOptimalNumberOfInferRequestsInParallel(add_platform_to_the_config(
                config,
                utils::getCompilationPlatform(
                    config.get<PLATFORM>(),
                    config.get<DEVICE_ID>(),
                    _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames())))));
        REGISTER_SIMPLE_METRIC(ov::range_for_async_infer_requests, true, _metrics->GetRangeForAsyncInferRequest());
        REGISTER_SIMPLE_METRIC(ov::range_for_streams, true, _metrics->GetRangeForStreams());
        REGISTER_SIMPLE_METRIC(ov::device::pci_info, true, _metrics->GetPciInfo(get_specified_device_name(config)));
        REGISTER_SIMPLE_METRIC(ov::device::gops, true, _metrics->GetGops(get_specified_device_name(config)));
        REGISTER_SIMPLE_METRIC(ov::device::type, true, _metrics->GetDeviceType(get_specified_device_name(config)));
        REGISTER_SIMPLE_METRIC(ov::internal::supported_properties, false, _internalSupportedProperties);
        REGISTER_SIMPLE_METRIC(ov::intel_npu::device_alloc_mem_size,
                               true,
                               _metrics->GetDeviceAllocMemSize(get_specified_device_name(config)));
        REGISTER_SIMPLE_METRIC(ov::intel_npu::device_total_mem_size,
                               true,
                               _metrics->GetDeviceTotalMemSize(get_specified_device_name(config)));
        REGISTER_SIMPLE_METRIC(ov::intel_npu::driver_version, true, _metrics->GetDriverVersion());
        REGISTER_SIMPLE_METRIC(ov::intel_npu::backend_name, false, _metrics->GetBackendName());
        REGISTER_SIMPLE_METRIC(ov::intel_npu::batch_mode, false, _metrics->GetDriverVersion());
        REGISTER_CUSTOM_METRIC(ov::device::architecture,
                               !_metrics->GetAvailableDevicesNames().empty(),
                               [&](const Config& config) {
                                   const auto specifiedDeviceName = get_specified_device_name(config);
                                   return _metrics->GetDeviceArchitecture(specifiedDeviceName);
                               });
        REGISTER_CUSTOM_METRIC(ov::device::full_name,
                               !_metrics->GetAvailableDevicesNames().empty(),
                               [&](const Config& config) {
                                   const auto specifiedDeviceName = get_specified_device_name(config);
                                   return _metrics->GetFullDeviceName(specifiedDeviceName);
                               });
        REGISTER_CUSTOM_METRIC(ov::device::luid,
                               _backend == nullptr ? false : _backend->isLUIDExtSupported(),
                               [&](const Config& config) {
                                   const auto specifiedDeviceName = get_specified_device_name(config);
                                   return _metrics->GetDeviceLUID(specifiedDeviceName);
                               });
        REGISTER_CUSTOM_METRIC(ov::device::uuid, true, [&](const Config& config) {
            const auto specifiedDeviceName = get_specified_device_name(config);
            auto devUuid = _metrics->GetDeviceUuid(specifiedDeviceName);
            return decltype(ov::device::uuid)::value_type{devUuid};
        });
        REGISTER_CUSTOM_METRIC(ov::execution_devices, true, [&](const Config& config) {
            if (_metrics->GetAvailableDevicesNames().size() > 1) {
                return std::string("NPU." + config.get<DEVICE_ID>());
            } else {
                return std::string("NPU");
            }
        });
        REGISTER_CUSTOM_METRIC(ov::intel_npu::compiler_version, true, [&](const Config& config) {
            /// create dummy compiler
            CompilerAdapterFactory compilerAdapterFactory;
            auto dummyCompiler = compilerAdapterFactory.getCompiler(_backend, config.get<COMPILER_TYPE>());
            return dummyCompiler->get_version();
        });
        REGISTER_CUSTOM_METRIC(ov::internal::caching_properties, false, [&](const Config& config) {
            // return a dynamically created list based on what is supported in current configuration
            std::vector<ov::PropertyName> caching_props{};
            // walk the static caching properties, add only what is supported now
            for (auto prop : _cachingProperties) {
                if (_config.isAvailable(prop)) {
                    caching_props.emplace_back(prop);
                }
            }
            // NPUW properties are requested by OV Core during caching and have no effect on the NPU plugin. But we
            // still need to enable those for OV Core to query. add all NPUW properties
            for (auto prop : _cachingProperties) {
                if (prop.find("NPUW") != prop.npos) {
                    caching_props.emplace_back(prop);
                }
            }
            return caching_props;
        });
    }
}

void Properties::registerCompiledModelProperties() {
    // 1. Configs
    // ========
    // 1.1 simple configs which only return value
    // TRY_REGISTER_SIMPLE_PROPERTY format: (property, config_to_return)
    // TRY_REGISTER_VARPUB_PROPERTY format: (property, config_to_return, dynamic public/private value)
    // TRY_REGISTER_CUSTOMFUNC_PROPERTY format: (property, custom_return_lambda_function)
    // TRY_REGISTER_CUSTOM_PROPERTY format: (property, visibility, mutability, custom_return_lambda_function)
    // FORCE_REGISTER_CUSTOM_PROPERTY format: (property, visibility, mutability, custom_return_lambda_function)

    // Permanent properties
    TRY_REGISTER_SIMPLE_PROPERTY(ov::hint::enable_cpu_pinning, ENABLE_CPU_PINNING);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::log::level, LOG_LEVEL);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::loaded_from_cache, LOADED_FROM_CACHE);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::hint::model_priority, MODEL_PRIORITY);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::hint::performance_mode, PERFORMANCE_HINT);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::hint::execution_mode, EXECUTION_MODE_HINT);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::hint::num_requests, PERFORMANCE_HINT_NUM_REQUESTS);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::compilation_num_threads, COMPILATION_NUM_THREADS);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::hint::inference_precision, INFERENCE_PRECISION_HINT);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::cache_mode, CACHE_MODE);

    // Properties we shall only enable if they were set prior-to-compilation
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::weights_path, WEIGHTS_PATH);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::cache_dir, CACHE_DIR);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::enable_profiling, PERF_COUNT);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::profiling_type, PROFILING_TYPE);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::turbo, TURBO);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::compilation_mode_params, COMPILATION_MODE_PARAMS);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::dma_engines, DMA_ENGINES);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::tiles, TILES);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::compilation_mode, COMPILATION_MODE);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::compiler_type, COMPILER_TYPE);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::platform, PLATFORM);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::dynamic_shape_to_static, DYNAMIC_SHAPE_TO_STATIC);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::backend_compilation_params, BACKEND_COMPILATION_PARAMS);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::bypass_umd_caching, BYPASS_UMD_CACHING);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::defer_weights_load, DEFER_WEIGHTS_LOAD);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::compiler_dynamic_quantization,
                                              COMPILER_DYNAMIC_QUANTIZATION);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::qdq_optimization, QDQ_OPTIMIZATION);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::qdq_optimization_aggressive, QDQ_OPTIMIZATION_AGGRESSIVE);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::disable_version_check, DISABLE_VERSION_CHECK);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::batch_compiler_mode_settings,
                                              BATCH_COMPILER_MODE_SETTINGS);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::run_inferences_sequentially, RUN_INFERENCES_SEQUENTIALLY);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::weightless_blob, WEIGHTLESS_BLOB);
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::separate_weights_version, SEPARATE_WEIGHTS_VERSION);

    TRY_REGISTER_VARPUB_PROPERTY(ov::intel_npu::batch_mode, BATCH_MODE, false);

    TRY_REGISTER_CUSTOM_PROPERTY(ov::workload_type,
                                 WORKLOAD_TYPE,
                                 true,
                                 ov::PropertyMutability::RW,
                                 [](const Config& config) {
                                     return config.get<WORKLOAD_TYPE>();
                                 });

    // 2. Metrics (static device and enviroment properties)
    // ========
    // REGISTER_SIMPLE_METRIC format: (property, public true/false, return value)
    // REGISTER_CUSTOM_METRIC format: (property, public true/false, return value function)

    REGISTER_CUSTOM_METRIC(ov::model_name, true, [](const Config&) {
        // TODO: log an error here as the code shouldn't have gotten here
        // this property is implemented in compiled model directly
        // this implementation here servers only to publish it in supported_properties
        return std::string("invalid");
    });
    REGISTER_SIMPLE_METRIC(ov::optimal_number_of_infer_requests,
                           true,
                           static_cast<uint32_t>(getOptimalNumberOfInferRequestsInParallel(config)));
    REGISTER_CUSTOM_METRIC(ov::internal::supported_properties, false, [&](const Config&) {
        static const std::vector<ov::PropertyName> supportedProperty{
            ov::PropertyName(ov::internal::caching_properties.name(), ov::PropertyMutability::RO)};
        return supportedProperty;
    });
    REGISTER_CUSTOM_METRIC(ov::execution_devices, true, [](const Config&) {
        // TODO: log an error here as the code shouldn't have gotten here
        // this property is implemented in compiled model directly
        // this implementation here servers only to publish it in supported_properties
        return std::string("NPU");
    });
}

ov::Any Properties::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    std::map<std::string, std::string> amends;
    for (auto&& value : arguments) {
        amends.emplace(value.first, value.second.as<std::string>());
    }
    FilteredConfig amendedConfig = _config;
    amendedConfig.update(amends, OptionMode::Both);

    auto&& configIterator = _properties.find(name);
    if (configIterator != _properties.cend()) {
        return std::get<2>(configIterator->second)(amendedConfig);
    }
    try {
        return amendedConfig.getInternal(name);
    } catch (...) {
        OPENVINO_THROW("Unsupported configuration key: ", name);
    }
}

void Properties::set_property(const ov::AnyMap& properties) {
    std::map<std::string, std::string> cfgs_to_set;

    std::unique_ptr<ICompilerAdapter> compiler = nullptr;
    if (_pType == PropertiesType::PLUGIN) {
        try {
            // Only accepting unknown config keys in plugin
            CompilerAdapterFactory compilerAdapterFactory;
            compiler = compilerAdapterFactory.getCompiler(_backend, _config.get<COMPILER_TYPE>());
        } catch (...) {
            // nothing to do here. we will just throw exception bellow in case unknown property check is called
            // if its not called, nothing to do
        }
    }

    for (auto&& value : properties) {
        if (_properties.find(value.first) == _properties.end()) {
            // property doesn't exist
            // checking as internal now
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
            if (std::get<1>(_properties[value.first]) == ov::PropertyMutability::RO) {
                OPENVINO_THROW("READ-ONLY configuration key: ", value.first);
            } else {
                cfgs_to_set.emplace(value.first, value.second.as<std::string>());
            }
        }
    }

    if (!cfgs_to_set.empty()) {
        _config.update(cfgs_to_set);
    }
}

}  // namespace intel_npu
