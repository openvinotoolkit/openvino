// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Plugin
#include "properties.hpp"

#include "intel_npu/al/config/options.hpp"

namespace intel_npu {
// namespace intel_npu

// Macro for registering simple get<> properties which have everything defined in their optionBase
#define TRY_REGISTER_SIMPLE_PROPERTY(OPT_NAME, OPT_TYPE)                                                \
    do {                                                                                                \
        std::string o_name = OPT_NAME.name();                                                           \
        if (_config.hasOpt(o_name)) {                                                                   \
            bool isPublic = _config.getOpt(o_name).isPublic();                                          \
            ov::PropertyMutability isMutable = _config.getOpt(o_name).mutability();                     \
            if (_pType == PropertiesType::COMPILED_MODEL) {                                             \
                if (_config.has<OPT_TYPE>()) {                                                          \
                    isMutable = ov::PropertyMutability::RO;                                             \
                } else {                                                                                \
                    break;                                                                              \
                }                                                                                       \
            }                                                                                           \
            _properties.emplace(o_name, std::make_tuple(isPublic, isMutable, [](const Config& config) { \
                                    return config.get<OPT_TYPE>();                                      \
                                }));                                                                    \
        }                                                                                               \
    } while (0)

// Macro for defining otherwise simple get<> properties but which have variable public/private field
#define TRY_REGISTER_VARPUB_PROPERTY(OPT_NAME, OPT_TYPE, PROP_VISIBILITY)                                      \
    do {                                                                                                       \
        std::string o_name = OPT_NAME.name();                                                                  \
        if (_config.hasOpt(o_name)) {                                                                          \
            ov::PropertyMutability isMutable = _config.getOpt(o_name).mutability();                            \
            if (_pType == PropertiesType::COMPILED_MODEL) {                                                    \
                if (_config.has<OPT_TYPE>()) {                                                                 \
                    isMutable = ov::PropertyMutability::RO;                                                    \
                } else {                                                                                       \
                    break;                                                                                     \
                }                                                                                              \
            }                                                                                                  \
            _properties.emplace(o_name, std::make_tuple(PROP_VISIBILITY, isMutable, [](const Config& config) { \
                                    return config.get<OPT_TYPE>();                                             \
                                }));                                                                           \
        }                                                                                                      \
    } while (0)

// Macro for registering config properties which have custom return function
#define TRY_REGISTER_CUSTOMFUNC_PROPERTY(OPT_NAME, OPT_TYPE, PROP_RETFUNC)                   \
    do {                                                                                     \
        std::string o_name = OPT_NAME.name();                                                \
        if (_config.hasOpt(o_name)) {                                                        \
            bool isPublic = _config.getOpt(o_name).isPublic();                               \
            ov::PropertyMutability isMutable = _config.getOpt(o_name).mutability();          \
            if (_pType == PropertiesType::COMPILED_MODEL) {                                  \
                if (_config.has<OPT_TYPE>()) {                                               \
                    isMutable = ov::PropertyMutability::RO;                                  \
                } else {                                                                     \
                    break;                                                                   \
                }                                                                            \
            }                                                                                \
            _properties.emplace(o_name, std::make_tuple(isPublic, isMutable, PROP_RETFUNC)); \
        }                                                                                    \
    } while (0)

// Macro for registering fully custom property, with option entry validation
#define TRY_REGISTER_CUSTOM_PROPERTY(OPT_NAME, OPT_TYPE, PROP_VISIBILITY, PROP_MUTABILITY, PROP_RETFUNC)  \
    do {                                                                                                  \
        std::string o_name = OPT_NAME.name();                                                             \
        if (_config.hasOpt(o_name)) {                                                                     \
            _properties.emplace(o_name, std::make_tuple(PROP_VISIBILITY, PROP_MUTABILITY, PROP_RETFUNC)); \
        }                                                                                                 \
    } while (0)

// Macro for force registering a fully custom property (no option entry validation)
#define FORCE_REGISTER_CUSTOM_PROPERTY(OPT_NAME, OPT_TYPE, PROP_VISIBILITY, PROP_MUTABILITY, PROP_RETFUNC)     \
    do {                                                                                                       \
        _properties.emplace(OPT_NAME.name(), std::make_tuple(PROP_VISIBILITY, PROP_MUTABILITY, PROP_RETFUNC)); \
    } while (0)

// Macro for defining simple single-function-call value returning metrics
#define REGISTER_SIMPLE_METRIC(PROP_NAME, PROP_VISIBILITY, PROP_RETVAL)                                              \
    do {                                                                                                             \
        _properties.emplace(PROP_NAME.name(),                                                                        \
                            std::make_tuple(PROP_VISIBILITY, ov::PropertyMutability::RO, [&](const Config& config) { \
                                return PROP_RETVAL;                                                                  \
                            }));                                                                                     \
    } while (0)

// Macro for defining metrics with custom return function
#define REGISTER_CUSTOM_METRIC(PROP_NAME, PROP_VISIBILITY, PROP_RETFUNC)                                 \
    do {                                                                                                 \
        _properties.emplace(PROP_NAME.name(),                                                            \
                            std::make_tuple(PROP_VISIBILITY, ov::PropertyMutability::RO, PROP_RETFUNC)); \
    } while (0)

static Config add_platform_to_the_config(Config config, const std::string_view platform) {
    config.update({{ov::intel_npu::platform.name(), std::string(platform)}});
    return config;
}

static auto get_specified_device_name(const Config config) {
    if (config.has<DEVICE_ID>()) {
        return config.get<DEVICE_ID>();
    }
    return std::string();
}

Properties::Properties(const PropertiesType pType, Config& config, const std::shared_ptr<Metrics>& metrics)
    : _pType(pType),
      _config(config),
      _metrics(metrics) {}

void Properties::registerProperties() {
    // Reset
    _properties.clear();

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
    TRY_REGISTER_SIMPLE_PROPERTY(ov::hint::enable_cpu_pinning, ENABLE_CPU_PINNING);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::log::level, LOG_LEVEL);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::cache_dir, CACHE_DIR);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::device::id, DEVICE_ID);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::num_streams, NUM_STREAMS);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::hint::model_priority, MODEL_PRIORITY);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::internal::exclusive_async_requests, EXCLUSIVE_ASYNC_REQUESTS);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::compilation_mode_params, COMPILATION_MODE_PARAMS);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::dma_engines, DMA_ENGINES);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::tiles, TILES);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::dpu_groups, DPU_GROUPS);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::compilation_mode, COMPILATION_MODE);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::compiler_type, COMPILER_TYPE);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::platform, PLATFORM);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::use_elf_compiler_backend, USE_ELF_COMPILER_BACKEND);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::create_executor, CREATE_EXECUTOR);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::dynamic_shape_to_static, DYNAMIC_SHAPE_TO_STATIC);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::profiling_type, PROFILING_TYPE);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::backend_compilation_params, BACKEND_COMPILATION_PARAMS);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::batch_mode, BATCH_MODE);
    TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::turbo, TURBO);

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
    // 1.2. Special cases
    // ==================
    if (_pType == PropertiesType::PLUGIN && _metrics != nullptr) {
        // These properties require different handling in plugin vs compiled_model
        TRY_REGISTER_SIMPLE_PROPERTY(ov::workload_type, WORKLOAD_TYPE);
    } else if (_pType == PropertiesType::COMPILED_MODEL) {
        // These properties require different handling in plugin vs compiled_model
        TRY_REGISTER_CUSTOM_PROPERTY(ov::workload_type,
                                     WORKLOAD_TYPE,
                                     true,
                                     ov::PropertyMutability::RW,
                                     [](const Config& config) {
                                         return config.get<WORKLOAD_TYPE>();
                                     });
    }

    // 2. Metrics (static device and enviroment properties)
    // ========
    // REGISTER_SIMPLE_METRIC format: (property, public true/false, return value)
    // REGISTER_CUSTOM_METRIC format: (property, public true/false, return value function)

    // 2.1 Metrics for Plugin-only (or those which need to be handled differently)
    if (_pType == PropertiesType::PLUGIN && _metrics != nullptr) {
        REGISTER_SIMPLE_METRIC(ov::available_devices, true, _metrics->GetAvailableDevicesNames());
        REGISTER_SIMPLE_METRIC(ov::device::capabilities, true, _metrics->GetOptimizationCapabilities());
        REGISTER_SIMPLE_METRIC(
            ov::optimal_number_of_infer_requests,
            true,
            static_cast<uint32_t>(getOptimalNumberOfInferRequestsInParallel(add_platform_to_the_config(
                config,
                _metrics->GetCompilationPlatform(config.get<PLATFORM>(), config.get<DEVICE_ID>())))));
        REGISTER_SIMPLE_METRIC(ov::range_for_async_infer_requests, true, _metrics->GetRangeForAsyncInferRequest());
        REGISTER_SIMPLE_METRIC(ov::range_for_streams, true, _metrics->GetRangeForStreams());
        REGISTER_SIMPLE_METRIC(ov::device::pci_info, true, _metrics->GetPciInfo(get_specified_device_name(config)));
        REGISTER_SIMPLE_METRIC(ov::device::gops, true, _metrics->GetGops(get_specified_device_name(config)));
        REGISTER_SIMPLE_METRIC(ov::device::type, true, _metrics->GetDeviceType(get_specified_device_name(config)));
        REGISTER_SIMPLE_METRIC(ov::internal::caching_properties, true, _metrics->GetCachingProperties());
        REGISTER_SIMPLE_METRIC(ov::internal::supported_properties, true, _metrics->GetInternalSupportedProperties());
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
    } else if (_pType == PropertiesType::COMPILED_MODEL) {
        /// 2.2 Metrics for CompiledModel-only (or those which need to be handled differently)
        REGISTER_CUSTOM_METRIC(ov::model_name, true, [](const Config&) {
            // TODO: log an error here as the code shouldn't have gotten here
            // this property is implemented in compiled model directly
            // this implementation here servers only to publish it in supported_properties
            return std::string("invalid");
        });
        REGISTER_SIMPLE_METRIC(ov::optimal_number_of_infer_requests,
                               true,
                               static_cast<uint32_t>(getOptimalNumberOfInferRequestsInParallel(config)));
        REGISTER_CUSTOM_METRIC(ov::internal::supported_properties, true, [&](const Config&) {
            static const std::vector<ov::PropertyName> supportedProperty{
                ov::PropertyName(ov::internal::caching_properties.name(), ov::PropertyMutability::RO)};
            return supportedProperty;
        });
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
    // DEBUG
    std::cout << "[CSOKADBG] Registered properties (all): " << std::endl;
    for (const auto& prop : _properties) {
        std::cout << "Key: " << prop.first << std::endl;
    }
    std::cout << "[CSOKADBG] Registered properties END: " << std::endl;
}

ov::Any Properties::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    std::map<std::string, std::string> amends;
    for (auto&& value : arguments) {
        amends.emplace(value.first, value.second.as<std::string>());
    }
    Config amendedConfig = _config;
    amendedConfig.update(amends, OptionMode::Both);

    auto&& configIterator = _properties.find(name);
    if (configIterator != _properties.cend()) {
        return std::get<2>(configIterator->second)(amendedConfig);
    }

    OPENVINO_THROW("Unsupported configuration key: ", name);
}

void Properties::set_property(const ov::AnyMap& properties) {
    std::map<std::string, std::string> cfgs_to_set;
    for (auto&& value : properties) {
        cfgs_to_set.emplace(value.first, value.second.as<std::string>());
    }
    for (const auto& configEntry : cfgs_to_set) {
        if (_properties.find(configEntry.first) == _properties.end()) {
            OPENVINO_THROW("Unsupported configuration key: ", configEntry.first);
        } else {
            if (std::get<1>(_properties[configEntry.first]) == ov::PropertyMutability::RO) {
                OPENVINO_THROW("READ-ONLY configuration key: ", configEntry.first);
            }
        }
    }

    _config.update(cfgs_to_set);
}

}  // namespace intel_npu