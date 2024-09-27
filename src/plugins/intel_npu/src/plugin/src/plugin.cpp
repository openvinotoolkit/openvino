// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <fstream>

#include "compiled_model.hpp"
#include "compiler.hpp"
#include "device_helpers.hpp"
#include "intel_npu/al/config/npuw.hpp"
#include "intel_npu/al/config/options.hpp"
#include "intel_npu/al/itt.hpp"
#include "npuw/compiled_model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "remote_context.hpp"

using namespace intel_npu;

namespace {

const std::vector<size_t> CONSTANT_NODE_DUMMY_SHAPE{1};

const char* NPU_PLUGIN_LIB_NAME = "openvino_intel_npu_plugin";

// Macro for registering simple get<> properties which have everything defined in their optionBase
#define REGISTER_SIMPLE_PROPERTY(option_name, config_type)                            \
    do {                                                                              \
        std::string o_name = option_name.name();                                      \
        if (_options->has(o_name)) {                                                  \
            _properties.emplace(o_name,                                               \
                                std::make_tuple(_options->get(o_name).isPublic(),     \
                                                _options->get(o_name).mutability(),   \
                                                [](const Config& config) {            \
                                                    return config.get<config_type>(); \
                                                }));                                  \
        }                                                                             \
    } while (0)

// Macro for defining otherwise simple get<> properties but which have variable public/private field
#define REGISTER_VARPUB_PROPERTY(option_name, config_type, __isPublic)                                     \
    do {                                                                                                   \
        std::string o_name = option_name.name();                                                           \
        if (_options->has(o_name)) {                                                                       \
            _properties.emplace(                                                                           \
                o_name,                                                                                    \
                std::make_tuple(__isPublic, _options->get(o_name).mutability(), [](const Config& config) { \
                    return config.get<config_type>();                                                      \
                }));                                                                                       \
        }                                                                                                  \
    } while (0)

// Macro for registering config properties which have custom return function
#define REGISTER_CUSTOM_PROPERTY(option_name, __retfunc)                                                           \
    do {                                                                                                           \
        std::string o_name = option_name.name();                                                                   \
        if (_options->has(o_name)) {                                                                               \
            _properties.emplace(                                                                                   \
                o_name,                                                                                            \
                std::make_tuple(_options->get(o_name).isPublic(), _options->get(o_name).mutability(), __retfunc)); \
        }                                                                                                          \
    } while (0)

// Macro for defining simple single-function-call value returning metrics
#define REGISTER_SIMPLE_METRIC(m_name, public, __retfunc)                                                   \
    do {                                                                                                    \
        _properties.emplace(m_name.name(),                                                                  \
                            std::make_tuple(public, ov::PropertyMutability::RO, [&](const Config& config) { \
                                return __retfunc;                                                           \
                            }));                                                                            \
    } while (0)

// Macro for defining metrics with custom return function
#define REGISTER_CUSTOM_METRIC(m_name, public, __retfunc)                                                   \
    do {                                                                                                    \
        _properties.emplace(m_name.name(), std::make_tuple(public, ov::PropertyMutability::RO, __retfunc)); \
    } while (0)

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
 * @param isBatchingSupported Newer driver versions support batching mode on the plugin.
 */
std::shared_ptr<ov::Model> create_dummy_model(const IONodeDescriptorMap& parameterDescriptors,
                                              const IONodeDescriptorMap& resultDescriptors,
                                              const std::vector<std::string>& inputNames,
                                              const std::vector<std::string>& outputNames,
                                              bool isBatchingSupported) {
    ov::ParameterVector parameters;
    ov::NodeVector results;

    for (const std::string& inputName : inputNames) {
        const IONodeDescriptor& parameterDescriptor = parameterDescriptors.at(inputName);

        std::shared_ptr<ov::op::v0::Parameter> parameter = [&] {
            if (isBatchingSupported) {
                return std::make_shared<ov::op::v0::Parameter>(parameterDescriptor.precision,
                                                               parameterDescriptor.originalShape);
            }
            return std::make_shared<ov::op::v0::Parameter>(parameterDescriptor.precision,
                                                           parameterDescriptor.transposedShape);
        }();

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

        const std::shared_ptr<ov::descriptor::Tensor>& tensorDummy = [&] {
            if (isBatchingSupported) {
                return std::make_shared<ov::descriptor::Tensor>(resultDescriptor.precision,
                                                                resultDescriptor.originalShape,
                                                                resultDescriptor.outputTensorNames);
            }
            return std::make_shared<ov::descriptor::Tensor>(resultDescriptor.precision,
                                                            resultDescriptor.transposedShape,
                                                            resultDescriptor.outputTensorNames);
        }();

        std::shared_ptr<ov::Node> result = std::make_shared<ov::op::v0::Result>(constantDummy);
        result->output(0).set_tensor_ptr(tensorDummy);
        result->set_friendly_name(resultDescriptor.currentNodeName);
        results.push_back(result);
    }

    return std::make_shared<ov::Model>(results, parameters);
}

/**
 * @brief Setting batching mode
 * @details  In the case of older drivers or discrete platforms, we force batching to compiler mode since it is not
 * supported. Othwersie set it tu AUTO if this wasn't set by the user
 * @param isBatchingSupported  Newer driver versions support batching mode on the plugin.
 * @param config A configuration map.
 */
void set_batch_config(bool isBatchingSupported, Config& config) {
    if (!isBatchingSupported || config.get<PLATFORM>() == ov::intel_npu::Platform::NPU3700) {
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

    /// Fetch CID version and create properties list
    OV_ITT_TASK_NEXT(PLUGIN, "FetchCompilerVer");
    compilerVersion cid_ver = {0, 0, 0};
    auto device = _backends->getDevice();
    if (device) {
        cid_ver = device->getCompilerVersion();
    }
    printf("\n\n[CSOKADBG]CID version major %d minor %d \n\n", cid_ver.vclMajor, cid_ver.vclMinor);

    OV_ITT_TASK_NEXT(PLUGIN, "Metrics");
    _metrics = std::make_unique<Metrics>(_backends);

    init_options(cid_ver);
    init_properties();
}

void Plugin::init_options(compilerVersion comp_ver) {
    // TODO: implement reset here

    // Initialize
    OV_ITT_TASK_NEXT(PLUGIN, "initOptions");
    registerOptions(*_options, comp_ver);
    _backends->registerOptions(*_options);

    // parse again env_variables after backend is initialized to get backend proprieties
    _globalConfig.parseEnvVars();

    std::vector<ov::PropertyName> sup_props = _options->getSupportedProperties();
    std::cout << "Registered options: " << std::endl;
    for (const std::string& prop : sup_props) {
        std::cout << prop << std::endl;
    }
    std::cout << "Registered options end;" << std::endl;
}

void Plugin::init_properties() {
    // TODO: implement reset here

    // 1. Configs
    // ========
    // 1.1 simple configs which only return value
    // REGISTER_SIMPLE_PROPERTY format: (property, config_to_return)
    // REGISTER_VARPUB_PROPERTY format: (property, config_to_return, dynamic public/private value)
    // REGISTER_CUSTOM_PROPERTY format: (property, custom_return_lambda_function)
    REGISTER_SIMPLE_PROPERTY(ov::enable_profiling, PERF_COUNT);
    REGISTER_SIMPLE_PROPERTY(ov::hint::performance_mode, PERFORMANCE_HINT);
    REGISTER_SIMPLE_PROPERTY(ov::hint::execution_mode, EXECUTION_MODE_HINT);
    REGISTER_SIMPLE_PROPERTY(ov::hint::num_requests, PERFORMANCE_HINT_NUM_REQUESTS);
    REGISTER_SIMPLE_PROPERTY(ov::compilation_num_threads, COMPILATION_NUM_THREADS);
    REGISTER_SIMPLE_PROPERTY(ov::hint::inference_precision, INFERENCE_PRECISION_HINT);
    REGISTER_SIMPLE_PROPERTY(ov::hint::enable_cpu_pinning, ENABLE_CPU_PINNING);
    REGISTER_SIMPLE_PROPERTY(ov::log::level, LOG_LEVEL);
    REGISTER_SIMPLE_PROPERTY(ov::cache_dir, CACHE_DIR);
    REGISTER_SIMPLE_PROPERTY(ov::device::id, DEVICE_ID);
    REGISTER_SIMPLE_PROPERTY(ov::num_streams, NUM_STREAMS);
    REGISTER_SIMPLE_PROPERTY(ov::hint::model_priority, MODEL_PRIORITY);
    REGISTER_SIMPLE_PROPERTY(ov::internal::exclusive_async_requests, EXCLUSIVE_ASYNC_REQUESTS);
    REGISTER_SIMPLE_PROPERTY(ov::intel_npu::compilation_mode_params, COMPILATION_MODE_PARAMS);
    REGISTER_SIMPLE_PROPERTY(ov::intel_npu::dma_engines, DMA_ENGINES);
    REGISTER_SIMPLE_PROPERTY(ov::intel_npu::tiles, TILES);
    REGISTER_SIMPLE_PROPERTY(ov::intel_npu::dpu_groups, DPU_GROUPS);
    REGISTER_SIMPLE_PROPERTY(ov::intel_npu::compilation_mode, COMPILATION_MODE);
    REGISTER_SIMPLE_PROPERTY(ov::intel_npu::compiler_type, COMPILER_TYPE);
    REGISTER_SIMPLE_PROPERTY(ov::intel_npu::platform, PLATFORM);
    REGISTER_SIMPLE_PROPERTY(ov::intel_npu::use_elf_compiler_backend, USE_ELF_COMPILER_BACKEND);
    REGISTER_SIMPLE_PROPERTY(ov::intel_npu::create_executor, CREATE_EXECUTOR);
    REGISTER_SIMPLE_PROPERTY(ov::intel_npu::dynamic_shape_to_static, DYNAMIC_SHAPE_TO_STATIC);
    REGISTER_SIMPLE_PROPERTY(ov::intel_npu::profiling_type, PROFILING_TYPE);
    REGISTER_SIMPLE_PROPERTY(ov::intel_npu::backend_compilation_params, BACKEND_COMPILATION_PARAMS);
    REGISTER_SIMPLE_PROPERTY(ov::intel_npu::batch_mode, BATCH_MODE);
    REGISTER_VARPUB_PROPERTY(ov::workload_type, WORKLOAD_TYPE, _backends->isCommandQueueExtSupported());
    REGISTER_VARPUB_PROPERTY(ov::intel_npu::turbo, TURBO, _backends->isCommandQueueExtSupported());
    REGISTER_CUSTOM_PROPERTY(ov::intel_npu::stepping, [&](const Config& config) {
        if (!config.has<STEPPING>()) {
            const auto specifiedDeviceName = get_specified_device_name(config);
            return static_cast<int64_t>(_metrics->GetSteppingNumber(specifiedDeviceName));
        } else {
            return config.get<STEPPING>();
        }
    });
    REGISTER_CUSTOM_PROPERTY(ov::intel_npu::max_tiles, [&](const Config& config) {
        if (!config.has<MAX_TILES>()) {
            const auto specifiedDeviceName = get_specified_device_name(config);
            return static_cast<int64_t>(_metrics->GetMaxTiles(specifiedDeviceName));
        } else {
            return config.get<MAX_TILES>();
        }
    });
    // 1.2. Special cases where generic macros don't fit

    // 2. Metrics
    // ========
    // 2.1. simple metrics which only return value
    // REGISTER_SIMPLE_METRIC format: (property, public true/false, return value)
    // REGISTER_CUSTOM_METRIC format: (property, public true/false, return value function)
    REGISTER_SIMPLE_METRIC(ov::available_devices, true, _metrics->GetAvailableDevicesNames());
    REGISTER_SIMPLE_METRIC(ov::device::capabilities, true, _metrics->GetOptimizationCapabilities());
    REGISTER_SIMPLE_METRIC(ov::optimal_number_of_infer_requests,
                           true,
                           static_cast<uint32_t>(getOptimalNumberOfInferRequestsInParallel(add_platform_to_the_config(
                               config,
                               _backends->getCompilationPlatform(config.get<PLATFORM>(), config.get<DEVICE_ID>())))));
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
    REGISTER_SIMPLE_METRIC(ov::supported_properties, true, _supportedProperties);
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
    // 2.2. Special cases where generic macro doesn't fit

    // 3. Populate supported properties list
    // ========
    for (auto& property : _properties) {
        if (std::get<0>(property.second)) {
            _supportedProperties.emplace_back(ov::PropertyName(property.first, std::get<1>(property.second)));
        }
    }
}

void Plugin::set_property(const ov::AnyMap& properties) {
    const std::map<std::string, std::string> config = any_copy(properties);
    update_log_level(config);
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
    if (_backends != nullptr) {
        _backends->setup(_globalConfig);
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

    // Before going any further: if
    // ... 1 - NPUW mode is activated
    // ... 2 - this request is NOT coming from NPUW,
    // activate the NPUW path
    auto useNpuwKey = ov::intel_npu::use_npuw.name();
    if (properties.count(useNpuwKey) && properties.at(useNpuwKey).as<bool>()) {
        // CACHE_DIR isn't supported with NPU_USE_NPUW
        if (properties.count(ov::cache_dir.name()) || !_globalConfig.get<CACHE_DIR>().empty()) {
            OPENVINO_THROW("Option 'CACHE_DIR' is not supported with NPU_USE_NPUW");
        }
        return std::make_shared<ov::npuw::CompiledModel>(model->clone(), shared_from_this(), properties);
    }

    const std::map<std::string, std::string> propertiesMap = any_copy(properties);
    update_log_level(propertiesMap);
    auto localConfig = merge_configs(_globalConfig, propertiesMap);

    const auto set_cache_dir = localConfig.get<CACHE_DIR>();
    if (!set_cache_dir.empty()) {
        const auto compilerType = localConfig.get<COMPILER_TYPE>();
        if (compilerType == ov::intel_npu::CompilerType::MLIR) {
            OPENVINO_THROW("Option 'CACHE_DIR' is not supported with MLIR compiler type");
        }
    }

    const auto platform = _backends->getCompilationPlatform(localConfig.get<PLATFORM>(), localConfig.get<DEVICE_ID>());
    auto device = _backends->getDevice(localConfig.get<DEVICE_ID>());
    localConfig.update({{ov::intel_npu::platform.name(), platform}});

    set_batch_config(_backends->isBatchingSupported(), localConfig);

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

    OV_ITT_TASK_NEXT(PLUGIN_COMPILE_MODEL, "compile");

    std::shared_ptr<ov::ICompiledModel> compiledModel;

    try {
        bool profiling = localConfig.get<PERF_COUNT>();

        compiledModel = std::make_shared<CompiledModel>(model,
                                                        shared_from_this(),
                                                        device,
                                                        getCompiler(localConfig),
                                                        profiling,
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

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties,
                                                          const ov::SoPtr<ov::IRemoteContext>& context) const {
    auto casted = std::dynamic_pointer_cast<RemoteContextImpl>(context._ptr);
    OPENVINO_ASSERT(casted, "Invalid remote context type. Can't cast to ov::intel_npu::RemoteContext type");

    return compile_model(model, properties);
}

ov::SoPtr<ov::IRemoteContext> Plugin::create_context(const ov::AnyMap& remote_properties) const {
    return get_default_context(remote_properties);
}

ov::SoPtr<ov::IRemoteContext> Plugin::get_default_context(const ov::AnyMap&) const {
    return std::make_shared<RemoteContextImpl>(_backends, _globalConfig);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& stream, const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "Plugin::import_model");
    OV_ITT_TASK_CHAIN(PLUGIN_IMPORT_MODEL, itt::domains::NPUPlugin, "Plugin::import_model", "merge_configs");
    const std::map<std::string, std::string> propertiesMap = any_copy(properties);
    update_log_level(propertiesMap);
    auto localConfig = merge_configs(_globalConfig, propertiesMap, OptionMode::RunTime);
    const auto platform = _backends->getCompilationPlatform(localConfig.get<PLATFORM>(), localConfig.get<DEVICE_ID>());
    localConfig.update({{ov::intel_npu::platform.name(), platform}});
    auto device = _backends->getDevice(localConfig.get<DEVICE_ID>());

    set_batch_config(_backends->isBatchingSupported(), localConfig);

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

        const std::shared_ptr<ov::Model> modelDummy = create_dummy_model(meta.parameters,
                                                                         meta.results,
                                                                         meta.inputNames,
                                                                         meta.outputNames,
                                                                         _backends->isBatchingSupported());

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
    const std::map<std::string, std::string> propertiesMap = any_copy(properties);
    update_log_level(propertiesMap);
    auto localConfig = merge_configs(_globalConfig, propertiesMap, OptionMode::CompileTime);
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
