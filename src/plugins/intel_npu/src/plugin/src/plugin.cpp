// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <fstream>

#include "compiled_model.hpp"
#include "compiler.hpp"
#include "device_helpers.hpp"
#include "intel_npu/al/config/common.hpp"
#include "intel_npu/al/config/compiler.hpp"
#include "intel_npu/al/config/runtime.hpp"
#include "intel_npu/al/itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

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
 * @param parameterDescriptors Describes the input nodes.
 * @param resultDescriptors Describes the output nodes.
 * @param inputNames The names of the inputs registered in the order given by the model.
 * @param outputNames The names of the outputs registered in the order given by the model.
 */
std::shared_ptr<ov::Model> create_dummy_model(const IONodeDescriptorMap& parameterDescriptors,
                                              const IONodeDescriptorMap& resultDescriptors,
                                              const std::vector<std::string>& inputNames,
                                              const std::vector<std::string>& outputNames) {
    ov::ParameterVector parameters;
    ov::NodeVector results;

    for (const std::string& inputName : inputNames) {
        const IONodeDescriptor& parameterDescriptor = parameterDescriptors.at(inputName);
        std::shared_ptr<ov::op::v0::Parameter> parameter =
            std::make_shared<ov::op::v0::Parameter>(parameterDescriptor.precision, parameterDescriptor.transposedShape);
        parameter->set_friendly_name(parameterDescriptor.currentNodeName);
        parameter->output(0).get_tensor().set_names(parameterDescriptor.outputTensorNames);
        parameters.push_back(parameter);
    }

    // The "result" nodes require a parent node in order to satisfy the legacy API naming conventions as well (in
    // the 1.0 API, the name of an output is given by the parent of the "result" node). Additionally, a dummy shape for
    // the "Constant" node was required since the specific constructor does not accept "ov::PartialShape" values (a
    // constant can't have dynamic shape). The dummy tensor was also brought in order to register the correct,
    // potentially dynamic, output shape.
    for (const std::string& outputName : outputNames) {
        const IONodeDescriptor& resultDescriptor = resultDescriptors.at(outputName);
        std::shared_ptr<ov::Node> constantDummy =
            std::make_shared<ov::op::v0::Constant>(resultDescriptor.precision, CONSTANT_NODE_DUMMY_SHAPE);
        constantDummy->set_friendly_name(resultDescriptor.legacyName);

        const std::shared_ptr<ov::descriptor::Tensor>& tensorDummy =
            std::make_shared<ov::descriptor::Tensor>(resultDescriptor.precision,
                                                     resultDescriptor.transposedShape,
                                                     resultDescriptor.outputTensorNames);

        std::shared_ptr<ov::Node> result = std::make_shared<ov::op::v0::Result>(constantDummy);
        result->output(0).set_tensor_ptr(tensorDummy);
        result->set_friendly_name(resultDescriptor.currentNodeName);
        results.push_back(result);
    }

    return std::make_shared<ov::Model>(results, parameters);
}

std::map<std::string, std::string> any_copy(const ov::AnyMap& params) {
    std::map<std::string, std::string> result;
    for (auto&& value : params) {
        result.emplace(value.first, value.second.as<std::string>());
    }
    return result;
}

size_t getFileSize(std::istream& stream) {
    const size_t streamStart = stream.tellg();
    stream.seekg(0, std::ios_base::end);
    const size_t streamEnd = stream.tellg();
    stream.seekg(streamStart, std::ios_base::beg);
    return streamEnd - streamStart;
}

}  // namespace

namespace intel_npu {

static Config merge_configs(const Config& globalConfig,
                            const std::map<std::string, std::string>& rawConfig,
                            OptionMode mode = OptionMode::Both) {
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

    // TODO: generation of available backends list can be done during execution of CMake scripts
    std::vector<AvailableBackends> backendRegistry;

#if defined(OPENVINO_STATIC_LIBRARY)
    backendRegistry.push_back(AvailableBackends::LEVEL_ZERO);
#else
#    if defined(ENABLE_IMD_BACKEND)
    if (const auto* envVar = std::getenv("IE_NPU_USE_IMD_BACKEND")) {
        if (envVarStrToBool("IE_NPU_USE_IMD_BACKEND", envVar)) {
            backendRegistry.push_back(AvailableBackends::IMD);
        }
    }
#    endif

#    if defined(_WIN32) || defined(_WIN64) || (defined(__linux__) && defined(__x86_64__))
    backendRegistry.push_back(AvailableBackends::LEVEL_ZERO);
#    endif
#endif

    OV_ITT_TASK_CHAIN(PLUGIN, itt::domains::NPUPlugin, "Plugin::Plugin", "NPUBackends");
    _backends = std::make_shared<NPUBackends>(backendRegistry, _globalConfig);
    OV_ITT_TASK_NEXT(PLUGIN, "registerOptions");
    _backends->registerOptions(*_options);

    OV_ITT_TASK_NEXT(PLUGIN, "Metrics");
    _metrics = std::make_unique<Metrics>(_backends);

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
                  _backends->getCompilationPlatform(config.get<PLATFORM>(), config.get<DEVICE_ID>()))));
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
        {ov::device::uuid.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config& config) {
              const auto specifiedDeviceName = get_specified_device_name(config);
              auto devUuid = _metrics->GetDeviceUuid(specifiedDeviceName);
              return decltype(ov::device::uuid)::value_type{devUuid};
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
        // OV Internals
        // =========
        {ov::internal::caching_properties.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config&) {
              return _metrics->GetCachingProperties();
          }}},
        {ov::internal::exclusive_async_requests.name(),
         {true,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<EXCLUSIVE_ASYNC_REQUESTS>();
          }}},
        {ov::internal::supported_properties.name(),
         {true,
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
        // NPU Private
        // =========
        {ov::hint::model_priority.name(),
         {false,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<MODEL_PRIORITY>();
          }}},
        {ov::intel_npu::dma_engines.name(),
         {false,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<DMA_ENGINES>();
          }}},
        {ov::intel_npu::tiles.name(),
         {false,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<TILES>();
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
          [](const Config& config) {
              return config.get<STEPPING>();
          }}},
        {ov::intel_npu::max_tiles.name(),
         {false,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<MAX_TILES>();
          }}},
        {ov::intel_npu::compilation_mode.name(),
         {false,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<COMPILATION_MODE>();
          }}},
        {ov::intel_npu::compilation_mode_params.name(),
         {false,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<COMPILATION_MODE_PARAMS>();
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
        {ov::intel_npu::use_elf_compiler_backend.name(),
         {false,
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.getString<USE_ELF_COMPILER_BACKEND>();
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
    };

    for (auto& property : _properties) {
        if (std::get<0>(property.second)) {
            _supportedProperties.emplace_back(ov::PropertyName(property.first, std::get<1>(property.second)));
        }
    }
}

void Plugin::set_property(const ov::AnyMap& properties) {
    const std::map<std::string, std::string> config = any_copy(properties);
    for (const auto& configEntry : config) {
        if (_properties.find(configEntry.first) == _properties.end()) {
            OPENVINO_THROW("Unsupported configuration key: ", configEntry.first);
        } else {
            if (std::get<1>(_properties[configEntry.first]) == ov::PropertyMutability::RO) {
                OPENVINO_THROW("READ-ONLY configuration key: ", configEntry.first);
            }
        }
    }

    _globalConfig.update(config);
    Logger::global().setLevel(_globalConfig.get<LOG_LEVEL>());
    if (_backends != nullptr) {
        _backends->setup(_globalConfig);
    }

    for (const auto& entry : config) {
        _config[entry.first] = entry.second;
    }
}

ov::Any Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    const std::map<std::string, std::string>& amends = any_copy(arguments);
    const Config amendedConfig = merge_configs(_globalConfig, amends);

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
    auto localConfig = merge_configs(_globalConfig, any_copy(properties));

    const auto set_cache_dir = localConfig.get<CACHE_DIR>();
    if (!set_cache_dir.empty()) {
        const auto compilerType = localConfig.get<COMPILER_TYPE>();
        if (compilerType == ov::intel_npu::CompilerType::MLIR) {
            _logger.error("Option 'CACHE_DIR' is not supported with MLIR compiler type");
        }
    }

    const auto platform = _backends->getCompilationPlatform(localConfig.get<PLATFORM>(), localConfig.get<DEVICE_ID>());
    auto device = _backends->getDevice(localConfig.get<DEVICE_ID>());
    localConfig.update({{ov::intel_npu::platform.name(), platform}});

    // Update stepping w/ information from driver, unless provided by user or we are off-device
    // Ignore, if compilation was requested for platform, different from current
    if (!localConfig.has<STEPPING>() && device != nullptr && device->getName() == platform) {
        try {
            localConfig.update({{ov::intel_npu::stepping.name(), std::to_string(device->getSubDevId())}});
        } catch (...) {
            _logger.warning("Stepping information not implemented by selected backend. Skipping. Please provide "
                            "NPU_STEPPING if required.");
        }
    }
    // Update max_tiles w/ information from driver, unless provided by user or we are off-device
    // Ignore, if compilation was requested for platform, different from current
    if (!localConfig.has<MAX_TILES>() && device != nullptr && device->getName() == platform) {
        try {
            localConfig.update({{ov::intel_npu::max_tiles.name(), std::to_string(device->getMaxNumSlices())}});
        } catch (...) {
            _logger.warning("Max tiles information not implemented by selected backend. Skipping. Please provide "
                            "NPU_MAX_TILES if required.");
        }
    }

    OV_ITT_TASK_NEXT(PLUGIN_COMPILE_MODEL, "compile");

    std::shared_ptr<const NetworkDescription> networkDescription;
    std::shared_ptr<ov::ICompiledModel> compiledModel;

    ov::SoPtr<ICompiler> compiler;
    try {
        compiler = getCompiler(localConfig);
        networkDescription = std::make_shared<const NetworkDescription>(compiler->compile(model, localConfig));
    } catch (const std::exception& ex) {
        OPENVINO_THROW(ex.what());
    } catch (...) {
        _logger.error("Unexpected exception");
        OPENVINO_THROW("NPU ExecutableNetwork got unexpected exception from compiler");
    }

    try {
        bool profiling = localConfig.get<PERF_COUNT>();

        compiledModel = std::make_shared<CompiledModel>(model,
                                                        shared_from_this(),
                                                        networkDescription,
                                                        device,
                                                        profiling ? std::optional(compiler) : std::nullopt,
                                                        localConfig);
    } catch (const std::exception& ex) {
        OPENVINO_THROW(ex.what());
    } catch (...) {
        OPENVINO_THROW("Unexpected exception thrown upon attempting to create the \"CompiledModel\" object");
    }

    ++_compiledModelLoadCounter;
    OV_ITT_TASK_SKIP(PLUGIN_COMPILE_MODEL);

    return compiledModel;
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& /*model*/,
                                                          const ov::AnyMap& /*properties*/,
                                                          const ov::SoPtr<ov::IRemoteContext>& /*context*/) const {
    OPENVINO_THROW_NOT_IMPLEMENTED("The remote context feature is not supported by the NPU plugin");
}

ov::SoPtr<ov::IRemoteContext> Plugin::create_context(const ov::AnyMap& /*remote_properties*/) const {
    OPENVINO_THROW_NOT_IMPLEMENTED("The remote context feature is not supported by the NPU plugin");
}

ov::SoPtr<ov::IRemoteContext> Plugin::get_default_context(const ov::AnyMap& /*remote_properties*/) const {
    OPENVINO_THROW_NOT_IMPLEMENTED("The remote context feature is not supported by the NPU plugin");
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& stream, const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "Plugin::import_model");

    OV_ITT_TASK_CHAIN(PLUGIN_IMPORT_MODEL, itt::domains::NPUPlugin, "Plugin::import_model", "merge_configs");
    auto localConfig = merge_configs(_globalConfig, any_copy(properties), OptionMode::RunTime);
    const auto platform = _backends->getCompilationPlatform(localConfig.get<PLATFORM>(), localConfig.get<DEVICE_ID>());
    localConfig.update({{ov::intel_npu::platform.name(), platform}});
    auto device = _backends->getDevice(localConfig.get<DEVICE_ID>());

    Logger logger("NPUPlugin", localConfig.get<LOG_LEVEL>());

    const auto loadedFromCache = localConfig.get<LOADED_FROM_CACHE>();
    if (!loadedFromCache) {
        logger.warning("The usage of a compiled model can lead to undefined behavior. Please use OpenVINO IR instead!");
    }

    OV_ITT_TASK_NEXT(PLUGIN_IMPORT_MODEL, "parse");

    std::shared_ptr<ov::ICompiledModel> compiledModel;

    try {
        auto compiler = getCompiler(localConfig);

        auto graphSize = getFileSize(stream);
        if (graphSize == 0) {
            OPENVINO_THROW("Blob is empty");
        }
        std::vector<uint8_t> blob(graphSize);
        stream.read(reinterpret_cast<char*>(blob.data()), graphSize);

        auto meta = compiler->parse(blob, localConfig);
        meta.name = "net" + std::to_string(_compiledModelLoadCounter++);

        const std::shared_ptr<ov::Model> modelDummy =
            create_dummy_model(meta.parameters, meta.results, meta.inputNames, meta.outputNames);

        bool profiling = localConfig.get<PERF_COUNT>();

        auto networkDescription = std::make_shared<const NetworkDescription>(std::move(blob), std::move(meta));

        compiledModel = std::make_shared<CompiledModel>(modelDummy,
                                                        shared_from_this(),
                                                        networkDescription,
                                                        device,
                                                        profiling ? std::optional(compiler) : std::nullopt,
                                                        localConfig);
    } catch (const std::exception& ex) {
        OPENVINO_THROW("Can't import network: ", ex.what());
    } catch (...) {
        OPENVINO_THROW("NPU import_model got unexpected exception from CompiledModel");
    }

    OV_ITT_TASK_SKIP(PLUGIN_IMPORT_MODEL);

    return compiledModel;
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& /*stream*/,
                                                         const ov::SoPtr<ov::IRemoteContext>& /*context*/,
                                                         const ov::AnyMap& /*properties*/) const {
    OPENVINO_THROW_NOT_IMPLEMENTED("The remote context feature is not supported by the NPU plugin");
}

ov::SupportedOpsMap Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                        const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "Plugin::query_model");

    auto localConfig = merge_configs(_globalConfig, any_copy(properties), OptionMode::CompileTime);
    const auto platform = _backends->getCompilationPlatform(localConfig.get<PLATFORM>(), localConfig.get<DEVICE_ID>());
    localConfig.update({{ov::intel_npu::platform.name(), platform}});

    auto compiler = getCompiler(localConfig);
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

ov::SoPtr<ICompiler> Plugin::getCompiler(const Config& config) const {
    auto compilerType = config.get<COMPILER_TYPE>();
    return createCompiler(compilerType, _logger);
}

std::atomic<int> Plugin::_compiledModelLoadCounter{1};

static const ov::Version version = {CI_BUILD_NUMBER, NPU_PLUGIN_LIB_NAME};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(Plugin, version)

}  // namespace intel_npu
