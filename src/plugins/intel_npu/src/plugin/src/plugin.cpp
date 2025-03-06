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
#include "intel_npu/config/npuw.hpp"
#include "intel_npu/config/options.hpp"
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

Plugin::Plugin()
    : _options(std::make_shared<OptionsDesc>()),
      _globalConfig(_options),
      _logger("NPUPlugin", Logger::global().level()) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "Plugin::Plugin");
    set_device_name("NPU");

    // parse env_variables to get LOG_LEVEL if needed
    _options->add<LOG_LEVEL>();
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

    OV_ITT_TASK_NEXT(PLUGIN, "InitOptions");
    init_options();

    /// Init and register properties
    OV_ITT_TASK_NEXT(PLUGIN, "RegisterProperties");
    _properties = std::make_unique<Properties>(PropertiesType::PLUGIN, _globalConfig, _metrics, _backend);
    _properties->registerProperties();
}

void Plugin::init_options() {
    // Initialize (note: it will reset registered options)
    _options->reset();

#define REGISTER_OPTION(OPT_TYPE)                             \
    do {                                                      \
        auto dummyopt = details::makeOptionModel<OPT_TYPE>(); \
        std::string o_name = dummyopt.key().data();           \
        _options->add<OPT_TYPE>();                            \
        _globalConfig.enable(o_name, false);                  \
    } while (0)

    REGISTER_OPTION(LOG_LEVEL);
    REGISTER_OPTION(CACHE_DIR);
    REGISTER_OPTION(DEVICE_ID);
    REGISTER_OPTION(NUM_STREAMS);
    REGISTER_OPTION(PERF_COUNT);
    REGISTER_OPTION(LOADED_FROM_CACHE);
    REGISTER_OPTION(COMPILATION_NUM_THREADS);
    REGISTER_OPTION(PERFORMANCE_HINT);
    REGISTER_OPTION(EXECUTION_MODE_HINT);
    REGISTER_OPTION(PERFORMANCE_HINT_NUM_REQUESTS);
    REGISTER_OPTION(ENABLE_CPU_PINNING);
    REGISTER_OPTION(INFERENCE_PRECISION_HINT);
    REGISTER_OPTION(MODEL_PRIORITY);
    REGISTER_OPTION(EXCLUSIVE_ASYNC_REQUESTS);
    REGISTER_OPTION(COMPILATION_MODE_PARAMS);
    REGISTER_OPTION(DMA_ENGINES);
    REGISTER_OPTION(TILES);
    REGISTER_OPTION(DPU_GROUPS);
    REGISTER_OPTION(COMPILATION_MODE);
    REGISTER_OPTION(COMPILER_TYPE);
    REGISTER_OPTION(PLATFORM);
    REGISTER_OPTION(CREATE_EXECUTOR);
    REGISTER_OPTION(DYNAMIC_SHAPE_TO_STATIC);
    REGISTER_OPTION(PROFILING_TYPE);
    REGISTER_OPTION(BACKEND_COMPILATION_PARAMS);
    REGISTER_OPTION(BATCH_MODE);
    REGISTER_OPTION(BYPASS_UMD_CACHING);
    REGISTER_OPTION(DEFER_WEIGHTS_LOAD);
    REGISTER_OPTION(WEIGHTS_PATH);
    REGISTER_OPTION(RUN_INFERENCES_SEQUENTIALLY);
    REGISTER_OPTION(COMPILER_DYNAMIC_QUANTIZATION);
    REGISTER_OPTION(QDQ_OPTIMIZATION);
    REGISTER_OPTION(STEPPING);
    REGISTER_OPTION(MAX_TILES);
    REGISTER_OPTION(DISABLE_VERSION_CHECK);
    if (_backend) {
        if (_backend->isCommandQueueExtSupported()) {
            REGISTER_OPTION(TURBO);
            REGISTER_OPTION(WORKLOAD_TYPE);
        }
    }

    recheck_compiler_support(_globalConfig);

    if (_backend) {
        _backend->registerOptions(*_options);
    }

    // parse again env_variables after backend is initialized to get backend proprieties
    _globalConfig.parseEnvVars();
}

void Plugin::recheck_compiler_support(Config& cfg) const {
    bool legacy = false;
    CompilerAdapterFactory compilerAdapterFactory;
    std::vector<std::string> compiler_support_list{};
    uint32_t compiler_version = 0;
    // create a dummy compiler to fetch version and supported options

    try {
        auto dummyCompiler = compilerAdapterFactory.getCompiler(_backend, cfg);
        compiler_version = dummyCompiler->get_version();
        compiler_support_list = dummyCompiler->get_supported_options();
        if (compiler_support_list.size() == 0) {
            _logger.warning("No compiler support options list received! Fallback to version-based option registration");
            legacy = true;
        }
    } catch (...) {
        // assuming getCompiler failed, meaning we are offline
        _logger.warning("No available or legacy compiler. Registering only legacy options with no compiler "
                        "version requirement");
        legacy = true;
    }

    // Logs
    _logger.debug("Compiler version: %ld", compiler_version);
    _logger.debug("Compiler supported options list (%ld): ", compiler_support_list.size());
    for (const auto& str : compiler_support_list) {
        _logger.debug("    %s ", str.c_str());
    }
    _logger.debug("Legacy registration: %s", legacy ? "true" : "false");

    // Parse enables
    cfg.walkEnables([&](const std::string& key) {
        bool isEnabled = false;
        auto opt = cfg.getOpt(key);
        if (opt.mode() == OptionMode::RunTime) {
            isEnabled = true;
        } else {
            if (legacy) {
                if (compiler_version >= opt.compilerSupportVersion()) {
                    isEnabled = true;
                }
            } else {
                auto it = std::find(compiler_support_list.begin(), compiler_support_list.end(), key);
                if (it != compiler_support_list.end()) {
                    isEnabled = true;
                } else {
                    try {
                        auto compiler = compilerAdapterFactory.getCompiler(_backend, cfg);
                        isEnabled = compiler->is_option_supported(key);
                    } catch (...) {
                        _logger.debug("Could not determine if option %s is supported!", key.c_str());
                        isEnabled = false;
                    }
                }
            }
        }
        if (!isEnabled) {
            _logger.debug("Config option %s not supported! Requirements not met.", key.c_str());
        } else {
            _logger.debug("Enabled config option %s", key.c_str());
        }
        // update enable flag
        cfg.enable(key, isEnabled);
    });
}

Config Plugin::fork_local_config(const std::map<std::string, std::string>& rawConfig, OptionMode mode) const {
    update_log_level(rawConfig);
    // create a copy of the global config
    Config localConfig = _globalConfig;

    // Check if compiler was changed
    // 1. Check for compiler change
    auto it = rawConfig.find(std::string(COMPILER_TYPE::key()));
    if (it != rawConfig.end()) {
        if (localConfig.getString<COMPILER_TYPE>() != it->second) {
            // Compiler type has changed!
            // Set new compiler type
            localConfig.update({{std::string(COMPILER_TYPE::key()), it->second}});
            // enable/disable config keys based on what the new compiler supports
            recheck_compiler_support(localConfig);
        }
    }

    localConfig.update(rawConfig, mode);
    return localConfig;
}

void Plugin::set_property(const ov::AnyMap& properties) {
    // 1. Check for compiler change
    if (properties.count(std::string(COMPILER_TYPE::key())) != 0) {
        // Compiler change detected
        // Set new compiler in _globalConfig
        auto it = properties.find(std::string(COMPILER_TYPE::key()));
        if (it != properties.end()) {
            _globalConfig.update({{std::string(COMPILER_TYPE::key()), it->second.as<std::string>()}});
            // enable/disable config keys based on what the new compiler supports
            recheck_compiler_support(_globalConfig);
            // 2. Reset properties for the new options
            _properties->registerProperties();
        }
    }

    // 2. Set the property via Properties interface
    _properties->set_property(properties);

    // 3. Extra hooks
    // Update log level if it was provided
    if (properties.count(std::string(LOG_LEVEL::key())) != 0) {
        Logger::global().setLevel(_globalConfig.get<LOG_LEVEL>());
    }
    // Init backends if needed
    if (_backend != nullptr) {
        _backend->setup(_globalConfig);
    }
}

ov::Any Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    return _properties->get_property(name, arguments);
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "Plugin::compile_model");
    OV_ITT_TASK_CHAIN(PLUGIN_COMPILE_MODEL, itt::domains::NPUPlugin, "Plugin::compile_model", "fork_local_config");

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

    std::map<std::string, std::string> localPropertiesMap = any_copy(localProperties);
    auto localConfig = fork_local_config(localPropertiesMap);
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
    OV_ITT_TASK_CHAIN(PLUGIN_IMPORT_MODEL, itt::domains::NPUPlugin, "Plugin::import_model", "fork_local_config");

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

    auto localConfig = fork_local_config(propertiesMap, OptionMode::RunTime);
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
    std::map<std::string, std::string> propertiesMap = any_copy(properties);
    auto localConfig = fork_local_config(propertiesMap, OptionMode::CompileTime);
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
