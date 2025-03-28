// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <fstream>

#include "compiled_model.hpp"
#include "compiler_adapter_factory.hpp"
#include "driver_compiler_adapter.hpp"
#include "intel_npu/common/device_helpers.hpp"
#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/common/igraph.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/common.hpp"
#include "intel_npu/config/compiler.hpp"
#include "intel_npu/config/npuw.hpp"
#include "intel_npu/config/runtime.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "metadata.hpp"
#include "npuw/compiled_model.hpp"
#include "npuw/llm_compiled_model.hpp"
#include "npuw/serialization.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "remote_context.hpp"

using namespace intel_npu;

namespace {

const std::vector<size_t> CONSTANT_NODE_DUMMY_SHAPE{1};

const char* NPU_PLUGIN_LIB_NAME = "openvino_intel_npu_plugin";

/**
 * @brief Creates an "ov::Model" object which contains only the given "parameter" and "result" nodes.
 * @details Using an "ov::Model" object to create the "CompiledModel" is the preferred way of using the OV API.
 * This path allows making use of the already written funtions/attributes for handling the I/O infromation.
 *
 * Note that a stored compiled model does not hold the original IR model within it. The only related information
 * which may be extracted is the original model's "parameter"/"result" nodes. Thus, we need to build a dummy model
 * starting from these fields in order to satisfy the API.
 *
 * @param inputDescriptors Describes the input nodes.
 * @param outputDescriptors Describes the output nodes.
 * @returns The dummy "ov::Model" composed of "parameter" and "result" nodes built using the given descriptors.
 */
std::shared_ptr<ov::Model> create_dummy_model(const std::vector<IODescriptor>& inputDescriptors,
                                              const std::vector<IODescriptor>& outputDescriptors) {
    ov::ParameterVector parameters;
    ov::NodeVector results;

    for (const IODescriptor& inputDescriptor : inputDescriptors) {
        if (inputDescriptor.isStateInput || inputDescriptor.isStateOutput || inputDescriptor.isShapeTensor) {
            continue;
        }

        std::shared_ptr<ov::op::v0::Parameter> parameter = std::make_shared<ov::op::v0::Parameter>(
            inputDescriptor.precision,
            inputDescriptor.shapeFromIRModel.has_value() ? *inputDescriptor.shapeFromIRModel
                                                         : inputDescriptor.shapeFromCompiler);

        parameter->set_friendly_name(inputDescriptor.nodeFriendlyName);
        parameter->output(0).get_tensor().set_names(inputDescriptor.outputTensorNames);
        parameters.push_back(std::move(parameter));
    }

    // The "result" nodes require a parent node in order to satisfy the API conventions. Additionally, a dummy shape for
    // the "Constant" node was required since the specific constructor does not accept "ov::PartialShape" values (a
    // constant can't have dynamic shape). The dummy tensor was also brought in order to register the correct,
    // potentially dynamic, output shape.
    for (const IODescriptor& outputDescriptor : outputDescriptors) {
        if (outputDescriptor.isStateInput || outputDescriptor.isStateOutput || outputDescriptor.isShapeTensor) {
            continue;
        }

        std::shared_ptr<ov::Node> constantDummy =
            std::make_shared<ov::op::v0::Constant>(outputDescriptor.precision, CONSTANT_NODE_DUMMY_SHAPE);

        const std::shared_ptr<ov::descriptor::Tensor>& tensorDummy = std::make_shared<ov::descriptor::Tensor>(
            outputDescriptor.precision,
            outputDescriptor.shapeFromIRModel.has_value() ? *outputDescriptor.shapeFromIRModel
                                                          : outputDescriptor.shapeFromCompiler,
            outputDescriptor.outputTensorNames);

        std::shared_ptr<ov::Node> result = std::make_shared<ov::op::v0::Result>(constantDummy);
        result->output(0).set_tensor_ptr(tensorDummy);
        result->set_friendly_name(outputDescriptor.nodeFriendlyName);
        results.push_back(std::move(result));
    }

    return std::make_shared<ov::Model>(results, parameters);
}

/**
 * @brief Setting batching mode
 * @details  In the case of older drivers, we force batching to compiler mode since it is not
 * supported. Othwersie set it tu AUTO if this wasn't set by the user
 * @param isBatchingSupported  Newer driver versions support batching mode on the plugin.
 * @param config A configuration map.
 */
void set_batch_config(bool isBatchingSupported, Config& config) {
    if (!isBatchingSupported) {
        if (config.has<BATCH_MODE>()) {
            if (config.get<BATCH_MODE>() == ov::intel_npu::BatchMode::PLUGIN) {
                OPENVINO_THROW("Batching on plugin is not supported with this driver version");
            }
        }

        std::stringstream strStream;
        strStream << ov::intel_npu::BatchMode::COMPILER;
        config.update({{ov::intel_npu::batch_mode.name(), strStream.str()}});
    }

    if (!config.has<BATCH_MODE>()) {
        std::stringstream strStream;
        strStream << ov::intel_npu::BatchMode::AUTO;
        config.update({{ov::intel_npu::batch_mode.name(), strStream.str()}});
    }
}

std::map<std::string, std::string> any_copy(const ov::AnyMap& params) {
    std::map<std::string, std::string> result;
    for (auto&& value : params) {
        // The value of cache_encryption_callbacks cannot be converted to std::string
        if (value.first == ov::cache_encryption_callbacks.name()) {
            continue;
        }
        result.emplace(value.first, value.second.as<std::string>());
    }
    return result;
}

void update_log_level(const std::map<std::string, std::string>& propertiesMap) {
    auto it = propertiesMap.find(std::string(LOG_LEVEL::key()));
    if (it != propertiesMap.end()) {
        std::istringstream is(it->second);
        ov::log::Level level;
        is >> level;
        Logger::global().setLevel(level);
    }
}

}  // namespace

namespace intel_npu {

static Config merge_configs(const Config& globalConfig,
                            const std::map<std::string, std::string>& rawConfig,
                            OptionMode mode = OptionMode::Both) {
    update_log_level(rawConfig);
    Config localConfig = globalConfig;
    localConfig.update(rawConfig, mode);
    return localConfig;
}

static auto get_specified_device_name(const Config config) {
    if (config.has<DEVICE_ID>()) {
        return config.get<DEVICE_ID>();
    }
    return std::string();
}

static Config add_platform_to_the_config(Config config, const std::string_view platform) {
    config.update({{ov::intel_npu::platform.name(), std::string(platform)}});
    return config;
}

Plugin::Plugin()
    : _options(std::make_shared<OptionsDesc>()),
      _globalConfig(_options),
      _logger("NPUPlugin", Logger::global().level()) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "Plugin::Plugin");
    set_device_name("NPU");

    registerCommonOptions(*_options);
    registerCompilerOptions(*_options);
    registerRunTimeOptions(*_options);

    // parse env_variables to get LOG_LEVEL if needed
    _globalConfig.parseEnvVars();
    Logger::global().setLevel(_globalConfig.get<LOG_LEVEL>());

    OV_ITT_TASK_CHAIN(PLUGIN, itt::domains::NPUPlugin, "Plugin::Plugin", "GetBackend");
    // backend registry shall be created after configs are updated
    _backendsRegistry = std::make_unique<BackendsRegistry>();
    _backend = _backendsRegistry->getEngineBackend();

    if (_backend) {
        OV_ITT_TASK_NEXT(PLUGIN, "registerBackendOptions");
        _backend->registerOptions(*_options);
    }

    OV_ITT_TASK_NEXT(PLUGIN, "createMetrics");
    _metrics = std::make_unique<Metrics>(_backend);

    // parse again env_variables after backend is initialized to get backend proprieties
    _globalConfig.parseEnvVars();

    // Map from name to function {Config -> ov::Any}
    // Note that some properties are RW before network is loaded, and become RO after network is loaded
    _properties = {
        // OV Public
        // =========
        {ov::supported_properties.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config&) {
              return _supportedProperties;
          }}},
        {ov::enable_profiling.name(),
         {true,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<PERF_COUNT>();
          }}},
        {ov::hint::performance_mode.name(),
         {true,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<PERFORMANCE_HINT>();
          }}},
        {ov::hint::execution_mode.name(),
         {true,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<EXECUTION_MODE_HINT>();
          }}},
        {ov::hint::num_requests.name(),
         {true,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<PERFORMANCE_HINT_NUM_REQUESTS>();
          }}},
        {ov::hint::inference_precision.name(),
         {true,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<INFERENCE_PRECISION_HINT>();
          }}},
        {ov::hint::enable_cpu_pinning.name(),
         {true,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<ENABLE_CPU_PINNING>();
          }}},
        {ov::log::level.name(),
         {true,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<LOG_LEVEL>();
          }}},
        {ov::cache_dir.name(),
         {true,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<CACHE_DIR>();
          }}},
        {ov::device::id.name(),
         {true,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<DEVICE_ID>();
          }}},
        {ov::compilation_num_threads.name(),
         {true,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.getString<COMPILATION_NUM_THREADS>();
          }}},
        {ov::available_devices.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config&) {
              return _metrics->GetAvailableDevicesNames();
          }}},
        {ov::workload_type.name(),
         {_backend == nullptr ? false : _backend->isCommandQueueExtSupported(),
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<WORKLOAD_TYPE>();
          }}},
        {ov::device::capabilities.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config&) {
              return _metrics->GetOptimizationCapabilities();
          }}},
        {ov::optimal_number_of_infer_requests.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config& config) {
              return static_cast<uint32_t>(getOptimalNumberOfInferRequestsInParallel(add_platform_to_the_config(
                  config,
                  utils::getCompilationPlatform(
                      config.get<PLATFORM>(),
                      config.get<DEVICE_ID>(),
                      _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames()))));
          }}},
        {ov::range_for_async_infer_requests.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config&) {
              return _metrics->GetRangeForAsyncInferRequest();
          }}},
        {ov::range_for_streams.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config&) {
              return _metrics->GetRangeForStreams();
          }}},
        {ov::num_streams.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<NUM_STREAMS>();
          }}},
        {ov::weights_path.name(),
         {true,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<WEIGHTS_PATH>();
          }}},
        {ov::device::uuid.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config& config) {
              const auto specifiedDeviceName = get_specified_device_name(config);
              auto devUuid = _metrics->GetDeviceUuid(specifiedDeviceName);
              return decltype(ov::device::uuid)::value_type{devUuid};
          }}},
        {ov::device::luid.name(),
         {_backend == nullptr ? false : _backend->isLUIDExtSupported(),
          ov::PropertyMutability::RO,
          [&](const Config& config) {
              const auto specifiedDeviceName = get_specified_device_name(config);
              return _metrics->GetDeviceLUID(specifiedDeviceName);
          }}},
        // Add FULL_DEVICE_NAME and DEVICE_ARCHITECTURE in supported
        // properties list only in case of non-empty device list (#1424144d)
        {ov::device::architecture.name(),
         {!_metrics->GetAvailableDevicesNames().empty(),
          ov::PropertyMutability::RO,
          [&](const Config& config) {
              const auto specifiedDeviceName = get_specified_device_name(config);
              return _metrics->GetDeviceArchitecture(specifiedDeviceName);
          }}},
        {ov::device::full_name.name(),
         {!_metrics->GetAvailableDevicesNames().empty(),
          ov::PropertyMutability::RO,
          [&](const Config& config) {
              const auto specifiedDeviceName = get_specified_device_name(config);
              return _metrics->GetFullDeviceName(specifiedDeviceName);
          }}},
        {ov::hint::model_priority.name(),
         {true,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<MODEL_PRIORITY>();
          }}},
        {ov::device::pci_info.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config& config) {
              return _metrics->GetPciInfo(get_specified_device_name(config));
          }}},
        {ov::device::gops.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config& config) {
              return _metrics->GetGops(get_specified_device_name(config));
          }}},
        {ov::device::type.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config& config) {
              return _metrics->GetDeviceType(get_specified_device_name(config));
          }}},
        {ov::execution_devices.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config& config) {
              if (_metrics->GetAvailableDevicesNames().size() > 1) {
                  return std::string("NPU." + config.get<DEVICE_ID>());
              } else {
                  return std::string("NPU");
              }
          }}},
        // OV Internals
        // =========
        {ov::internal::caching_properties.name(),
         {false,
          ov::PropertyMutability::RO,
          [&](const Config&) {
              return _metrics->GetCachingProperties();
          }}},
        {ov::internal::exclusive_async_requests.name(),
         {false,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<EXCLUSIVE_ASYNC_REQUESTS>();
          }}},
        {ov::internal::supported_properties.name(),
         {false,
          ov::PropertyMutability::RO,
          [&](const Config&) {
              return _metrics->GetInternalSupportedProperties();
          }}},
        // NPU Public
        // =========
        {ov::intel_npu::device_alloc_mem_size.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config& config) {
              return _metrics->GetDeviceAllocMemSize(get_specified_device_name(config));
          }}},
        {ov::intel_npu::device_total_mem_size.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config& config) {
              return _metrics->GetDeviceTotalMemSize(get_specified_device_name(config));
          }}},
        {ov::intel_npu::driver_version.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config& config) {
              return _metrics->GetDriverVersion();
          }}},
        {ov::intel_npu::compiler_version.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config& config) {
              /// create dummy compiler
              CompilerAdapterFactory compilerAdapterFactory;
              auto dummyCompiler = compilerAdapterFactory.getCompiler(_backend, config);
              return dummyCompiler->get_version();
          }}},
        {ov::intel_npu::compilation_mode_params.name(),
         {true,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<COMPILATION_MODE_PARAMS>();
          }}},
        {ov::intel_npu::compiler_dynamic_quantization.name(),
         {false,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<COMPILER_DYNAMIC_QUANTIZATION>();
          }}},
        {ov::intel_npu::qdq_optimization.name(),
         {false,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<QDQ_OPTIMIZATION>();
          }}},
        {ov::intel_npu::turbo.name(),
         {_backend == nullptr ? false : _backend->isCommandQueueExtSupported(),
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<TURBO>();
          }}},
        {ov::intel_npu::tiles.name(),
         {true,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<TILES>();
          }}},
        {ov::intel_npu::max_tiles.name(),
         {true,
          ov::PropertyMutability::RW,
          [&](const Config& config) {
              if (!config.has<MAX_TILES>()) {
                  const auto specifiedDeviceName = get_specified_device_name(config);
                  return static_cast<int64_t>(_metrics->GetMaxTiles(specifiedDeviceName));
              }
              return config.get<MAX_TILES>();
          }}},
        {ov::intel_npu::bypass_umd_caching.name(),
         {true,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<BYPASS_UMD_CACHING>();
          }}},
        {ov::intel_npu::defer_weights_load.name(),
         {true,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<DEFER_WEIGHTS_LOAD>();
          }}},
        // NPU Private
        // =========
        {ov::intel_npu::dma_engines.name(),
         {false,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<DMA_ENGINES>();
          }}},
        {ov::intel_npu::dpu_groups.name(),
         {false,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<DPU_GROUPS>();
          }}},
        {ov::intel_npu::stepping.name(),
         {false,
          ov::PropertyMutability::RW,
          [&](const Config& config) {
              if (!config.has<STEPPING>()) {
                  const auto specifiedDeviceName = get_specified_device_name(config);
                  return static_cast<int64_t>(_metrics->GetSteppingNumber(specifiedDeviceName));
              } else {
                  return config.get<STEPPING>();
              }
          }}},
        {ov::intel_npu::compilation_mode.name(),
         {false,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<COMPILATION_MODE>();
          }}},
        {ov::intel_npu::compiler_type.name(),
         {false,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.getString<COMPILER_TYPE>();
          }}},
        {ov::intel_npu::platform.name(),
         {false,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<PLATFORM>();
          }}},
        {ov::intel_npu::backend_name.name(),
         {false,
          ov::PropertyMutability::RO,
          [&](const Config&) {
              return _metrics->GetBackendName();
          }}},
        {ov::intel_npu::create_executor.name(),
         {false,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<CREATE_EXECUTOR>();
          }}},
        {ov::intel_npu::dynamic_shape_to_static.name(),
         {false,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<DYNAMIC_SHAPE_TO_STATIC>();
          }}},
        {ov::intel_npu::profiling_type.name(),
         {false,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<PROFILING_TYPE>();
          }}},
        {ov::intel_npu::backend_compilation_params.name(),
         {false,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.getString<BACKEND_COMPILATION_PARAMS>();
          }}},
        {ov::intel_npu::run_inferences_sequentially.name(),
         {false,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<RUN_INFERENCES_SEQUENTIALLY>();
          }}},
        {ov::intel_npu::batch_mode.name(),
         {false,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.getString<BATCH_MODE>();
          }}},
        {ov::intel_npu::disable_version_check.name(),
         {false,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.getString<DISABLE_VERSION_CHECK>();
          }}},
        {ov::intel_npu::batch_compiler_mode_settings.name(),
         {false, ov::PropertyMutability::RW, [](const Config& config) {
              return config.get<BATCH_COMPILER_MODE_SETTINGS>();
          }}}};
}

void Plugin::reset_supported_properties() const {
    /// reset first
    _supportedProperties.clear();  /// Mutable member
    /// populate
    for (auto& property : _properties) {
        if (std::get<0>(property.second)) {
            _supportedProperties.emplace_back(ov::PropertyName(property.first, std::get<1>(property.second)));
        }
    }
}

void Plugin::reset_compiler_dependent_properties() const {
    uint32_t active_compiler_version = 0;
    // get active compiler version
    try {
        CompilerAdapterFactory compilerAdapterFactory;
        auto dummyCompiler = compilerAdapterFactory.getCompiler(_backend, _globalConfig);
        active_compiler_version = dummyCompiler->get_version();
    } catch (...) {
        _logger.warning(
            "No available compiler. Can not determine version > compiler dependent properties remain hidden");
        return;
    }

    // NPU_COMPILER_DYNAMIC_QUANTIZATION
    // unpublish if compiler version requirement is not met
    if (_properties.find(ov::intel_npu::compiler_dynamic_quantization.name()) != _properties.end()) {
        if (active_compiler_version >= ICOMPILER_MAKE_VERSION(7, 1)) {
            std::get<0>(_properties[ov::intel_npu::compiler_dynamic_quantization.name()]) = true;  /// mark supported
        } else {
            std::get<0>(_properties[ov::intel_npu::compiler_dynamic_quantization.name()]) = false;  // mark unsupported
        }
    }

    // NPU_QDQ_OPTIMIZATION
    // unpublish if compiler version requirement is not met
    if (_properties.find(ov::intel_npu::qdq_optimization.name()) != _properties.end()) {
        if (active_compiler_version >= ICOMPILER_MAKE_VERSION(7, 20)) {
            std::get<0>(_properties[ov::intel_npu::qdq_optimization.name()]) = true;  /// mark supported
        } else {
            std::get<0>(_properties[ov::intel_npu::qdq_optimization.name()]) = false;  // mark unsupported
        }
    }
}

void Plugin::set_property(const ov::AnyMap& properties) {
    const std::map<std::string, std::string> config = any_copy(properties);
    update_log_level(config);
    bool compiler_type_change = false;
    for (const auto& configEntry : config) {
        if (_properties.find(configEntry.first) == _properties.end()) {
            OPENVINO_THROW("Unsupported configuration key: ", configEntry.first);
        } else {
            if (std::get<1>(_properties[configEntry.first]) == ov::PropertyMutability::RO) {
                OPENVINO_THROW("READ-ONLY configuration key: ", configEntry.first);
            }
            if (configEntry.first == ov::intel_npu::compiler_type.name()) {
                // we just assume its a change, not compare against old value
                compiler_type_change = true;
            }
        }
    }

    _globalConfig.update(config);
    if (_backend != nullptr) {
        _backend->updateInfo(_globalConfig);
    }

    for (const auto& entry : config) {
        _config[entry.first] = entry.second;
    }

    if (compiler_type_change) {
        // if compiler type was changed > need to reset properties to match the new compiler
        // since properties have changed > need to reset supported_properties as well
        reset_compiler_dependent_properties();
        reset_supported_properties();
    }
}

ov::Any Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    const std::map<std::string, std::string>& amends = any_copy(arguments);
    const Config amendedConfig = merge_configs(_globalConfig, amends);

    /// Special case for supportedProperties
    /// populate it at first get
    if (name == ov::supported_properties.name() && _supportedProperties.size() < 1) {
        reset_compiler_dependent_properties();
        reset_supported_properties();
    }

    auto&& configIterator = _properties.find(name);
    if (configIterator != _properties.cend()) {
        return std::get<2>(configIterator->second)(amendedConfig);
    }

    OPENVINO_THROW("Unsupported configuration key: ", name);
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "Plugin::compile_model");
    OV_ITT_TASK_CHAIN(PLUGIN_COMPILE_MODEL, itt::domains::NPUPlugin, "Plugin::compile_model", "merge_configs");

    // Before going any further: if
    // ... 1 - NPUW mode is activated
    // ... 2 - this request is NOT coming from NPUW,
    // activate the NPUW path
    auto useNpuwKey = ov::intel_npu::use_npuw.name();
    ov::AnyMap localProperties = properties;
    if (localProperties.count(useNpuwKey)) {
        if (localProperties.at(useNpuwKey).as<bool>() == true) {
            return ov::npuw::ICompiledModel::create(model->clone(), shared_from_this(), localProperties);
        } else {
            // NPUW is disabled, remove the key from the properties
            localProperties.erase(useNpuwKey);
        }
    }

    const std::map<std::string, std::string> localPropertiesMap = any_copy(localProperties);
    auto localConfig = merge_configs(_globalConfig, localPropertiesMap);
    update_log_level(localPropertiesMap);

    const auto set_cache_dir = localConfig.get<CACHE_DIR>();
    if (!set_cache_dir.empty()) {
        const auto compilerType = localConfig.get<COMPILER_TYPE>();
        if (compilerType == ov::intel_npu::CompilerType::MLIR) {
            OPENVINO_THROW("Option 'CACHE_DIR' is not supported with MLIR compiler type");
        }
    }

    const auto platform =
        utils::getCompilationPlatform(localConfig.get<PLATFORM>(),
                                      localConfig.get<DEVICE_ID>(),
                                      _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());
    auto device = _backend == nullptr ? nullptr : _backend->getDevice(localConfig.get<DEVICE_ID>());
    localConfig.update({{ov::intel_npu::platform.name(), platform}});

    set_batch_config(_backend == nullptr ? false : _backend->isBatchingSupported(), localConfig);

    if (!model->get_variables().empty()) {
        if (localConfig.get<BATCH_MODE>() == ov::intel_npu::BatchMode::PLUGIN) {
            OPENVINO_THROW("This model contains states, thus it is not supported when handling batching on the plugin");
        }

        _logger.info("The batching will be handled by the compiler due to states found inside the IR");

        std::stringstream strStream;
        strStream << ov::intel_npu::BatchMode::COMPILER;
        localConfig.update({{ov::intel_npu::batch_mode.name(), strStream.str()}});
    }

    // Update stepping w/ information from driver, unless provided by user or we are off-device
    // Ignore, if compilation was requested for platform, different from current
    if (!localConfig.has<STEPPING>() && device != nullptr &&
        device->getName() == ov::intel_npu::Platform::standardize(platform) &&
        _metrics->GetBackendName() == "level_zero") {
        try {
            localConfig.update({{ov::intel_npu::stepping.name(), std::to_string(device->getSubDevId())}});
        } catch (...) {
            _logger.warning("Stepping information not implemented by selected backend. Skipping. Please provide "
                            "NPU_STEPPING if required.");
        }
    }
    // Update max_tiles w/ information from driver, unless provided by user or we are off-device
    // Ignore, if compilation was requested for platform, different from current
    if (!localConfig.has<MAX_TILES>() && device != nullptr &&
        device->getName() == ov::intel_npu::Platform::standardize(platform) &&
        _metrics->GetBackendName() == "level_zero") {
        try {
            localConfig.update({{ov::intel_npu::max_tiles.name(), std::to_string(device->getMaxNumSlices())}});
        } catch (...) {
            _logger.warning("Max tiles information not implemented by selected backend. Skipping. Please provide "
                            "NPU_MAX_TILES if required.");
        }
    }

    CompilerAdapterFactory compilerAdapterFactory;
    auto compiler = compilerAdapterFactory.getCompiler(_backend, localConfig);

    OV_ITT_TASK_NEXT(PLUGIN_COMPILE_MODEL, "compile");
    std::shared_ptr<intel_npu::IGraph> graph;
    try {
        _logger.debug("performing compile");
        graph = compiler->compile(model->clone(), localConfig);
    } catch (const std::exception& ex) {
        OPENVINO_THROW(ex.what());
    } catch (...) {
        _logger.error("Unexpected exception");
        OPENVINO_THROW("NPU plugin: got an unexpected exception from compiler");
    }

    std::shared_ptr<ov::ICompiledModel> compiledModel;
    try {
        compiledModel = std::make_shared<CompiledModel>(model, shared_from_this(), device, graph, localConfig);
    } catch (const std::exception& ex) {
        OPENVINO_THROW(ex.what());
    } catch (...) {
        OPENVINO_THROW("Unexpected exception thrown upon attempting to create the \"CompiledModel\" object");
    }

    ++_compiledModelLoadCounter;
    OV_ITT_TASK_SKIP(PLUGIN_COMPILE_MODEL);

    return compiledModel;
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties,
                                                          const ov::SoPtr<ov::IRemoteContext>& context) const {
    auto casted = std::dynamic_pointer_cast<RemoteContextImpl>(context._ptr);
    if (casted == nullptr) {
        OPENVINO_THROW("Invalid remote context type. Can't cast to ov::intel_npu::RemoteContext type");
    }

    return compile_model(model, properties);
}

ov::SoPtr<ov::IRemoteContext> Plugin::create_context(const ov::AnyMap& remoteProperties) const {
    return std::make_shared<RemoteContextImpl>(_backend, _globalConfig, remoteProperties);
}

ov::SoPtr<ov::IRemoteContext> Plugin::get_default_context(const ov::AnyMap&) const {
    return std::make_shared<RemoteContextImpl>(_backend, _globalConfig);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& stream, const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "Plugin::import_model");
    OV_ITT_TASK_CHAIN(PLUGIN_IMPORT_MODEL, itt::domains::NPUPlugin, "Plugin::import_model", "merge_configs");

    // If was exported via NPUW
    auto stream_start_pos = stream.tellg();
    ov::npuw::s11n::IndicatorType serialization_indicator;
    ov::npuw::s11n::read(stream, serialization_indicator);
    if (serialization_indicator == NPUW_SERIALIZATION_INDICATOR) {
        stream.seekg(-stream.tellg() + stream_start_pos, std::ios::cur);
        // Properties are required for ov::weights_path
        return ov::npuw::LLMCompiledModel::import_model(stream, shared_from_this(), properties);
    }
    stream.seekg(-stream.tellg() + stream_start_pos, std::ios::cur);

    // Drop NPUW properties if there are any
    ov::AnyMap npu_plugin_properties;
    for (auto it = properties.begin(); it != properties.end(); ++it) {
        if (it->first.find("NPUW") == it->first.npos) {
            npu_plugin_properties.insert(*it);
        }
    }

    std::shared_ptr<ov::AlignedBuffer> modelBuffer;
    // ov::hint::compiled_blob has no corresponding "Config" implementation thus we need to remove it from the
    // list of properties
    if (auto blob_it = npu_plugin_properties.find(ov::hint::compiled_blob.name());
        blob_it != npu_plugin_properties.end()) {
        auto compiled_blob = blob_it->second.as<ov::Tensor>();
        modelBuffer = std::make_shared<ov::SharedBuffer<ov::Tensor>>(reinterpret_cast<char*>(compiled_blob.data()),
                                                                     compiled_blob.get_byte_size(),
                                                                     compiled_blob);
        npu_plugin_properties.erase(blob_it);
    }

    const auto propertiesMap = any_copy(npu_plugin_properties);

    auto localConfig = merge_configs(_globalConfig, propertiesMap, OptionMode::RunTime);
    _logger.setLevel(localConfig.get<LOG_LEVEL>());
    const auto platform =
        utils::getCompilationPlatform(localConfig.get<PLATFORM>(),
                                      localConfig.get<DEVICE_ID>(),
                                      _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());
    localConfig.update({{ov::intel_npu::platform.name(), platform}});
    auto device = _backend == nullptr ? nullptr : _backend->getDevice(localConfig.get<DEVICE_ID>());

    set_batch_config(_backend == nullptr ? false : _backend->isBatchingSupported(), localConfig);

    const auto loadedFromCache = localConfig.get<LOADED_FROM_CACHE>();
    if (!loadedFromCache) {
        _logger.warning(
            "The usage of a compiled model can lead to undefined behavior. Please use OpenVINO IR instead!");
    }

    OV_ITT_TASK_NEXT(PLUGIN_IMPORT_MODEL, "parse");

    std::shared_ptr<ov::ICompiledModel> compiledModel;

    try {
        CompilerAdapterFactory compilerAdapterFactory;
        auto compiler = compilerAdapterFactory.getCompiler(_backend, localConfig);

        uint64_t graphSize;
        const bool skipCompatibility = localConfig.get<DISABLE_VERSION_CHECK>();
        if (!skipCompatibility) {
            auto storedMeta = read_metadata_from(stream);
            if (!storedMeta->is_compatible()) {
                OPENVINO_THROW("Incompatible blob version!");
            }
            graphSize = storedMeta->get_blob_size();
        } else {
            _logger.info("Blob compatibility check skipped.");
            graphSize = MetadataBase::getFileSize(stream);
        }

        std::unique_ptr<BlobContainer> blobPtr;

        if (modelBuffer == nullptr) {
            std::vector<uint8_t> blob(graphSize);
            stream.read(reinterpret_cast<char*>(blob.data()), graphSize);
            if (!stream) {
                OPENVINO_THROW("Failed to read data from stream!");
            }
            _logger.debug("Successfully read %zu bytes into blob.", graphSize);

            blobPtr = std::make_unique<BlobContainerVector>(std::move(blob));
        } else {
            blobPtr = std::make_unique<BlobContainerAlignedBuffer>(modelBuffer, stream.tellg(), graphSize);
        }

        auto graph = compiler->parse(std::move(blobPtr), localConfig);
        graph->update_network_name("net" + std::to_string(_compiledModelLoadCounter++));

        const std::shared_ptr<ov::Model> modelDummy =
            create_dummy_model(graph->get_metadata().inputs, graph->get_metadata().outputs);

        compiledModel = std::make_shared<CompiledModel>(modelDummy, shared_from_this(), device, graph, localConfig);
    } catch (const std::exception& ex) {
        OPENVINO_THROW("Can't import network: ", ex.what());
    } catch (...) {
        OPENVINO_THROW("NPU import_model got unexpected exception from CompiledModel");
    }

    OV_ITT_TASK_SKIP(PLUGIN_IMPORT_MODEL);

    return compiledModel;
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& stream,
                                                         const ov::SoPtr<ov::IRemoteContext>& context,
                                                         const ov::AnyMap& properties) const {
    auto casted = std::dynamic_pointer_cast<RemoteContextImpl>(context._ptr);
    if (casted == nullptr) {
        OPENVINO_THROW("Invalid remote context type. Can't cast to ov::intel_npu::RemoteContext type");
    }

    return import_model(stream, properties);
}

ov::SupportedOpsMap Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                        const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "Plugin::query_model");
    const std::map<std::string, std::string> propertiesMap = any_copy(properties);
    auto localConfig = merge_configs(_globalConfig, propertiesMap, OptionMode::CompileTime);
    _logger.setLevel(localConfig.get<LOG_LEVEL>());
    const auto platform =
        utils::getCompilationPlatform(localConfig.get<PLATFORM>(),
                                      localConfig.get<DEVICE_ID>(),
                                      _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());
    localConfig.update({{ov::intel_npu::platform.name(), platform}});

    CompilerAdapterFactory compilerAdapterFactory;
    auto compiler = compilerAdapterFactory.getCompiler(_backend, localConfig);
    ov::SupportedOpsMap supportedOpsMap;
    try {
        supportedOpsMap = compiler->query(model, localConfig);
    } catch (const std::runtime_error& e) {
        OPENVINO_THROW(e.what());
    } catch (...) {
        OPENVINO_THROW("NPU query_model got unexpected error from compiler");
    }

    return supportedOpsMap;
}

std::atomic<int> Plugin::_compiledModelLoadCounter{1};

static const ov::Version version = {CI_BUILD_NUMBER, NPU_PLUGIN_LIB_NAME};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(Plugin, version)

}  // namespace intel_npu
