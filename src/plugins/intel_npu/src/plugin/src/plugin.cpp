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
        parameters.push_back(parameter);
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
        results.push_back(result);
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

size_t getFileSize(std::istream& stream) {
    auto log = Logger::global().clone("getFileSize");
    if (!stream) {
        OPENVINO_THROW("Stream is in bad status! Please check the passed stream status!");
    }

    const size_t streamStart = stream.tellg();
    stream.seekg(0, std::ios_base::end);
    const size_t streamEnd = stream.tellg();
    stream.seekg(streamStart, std::ios_base::beg);

    log.debug("Read blob size: streamStart=%zu, streamEnd=%zu", streamStart, streamEnd);

    if (streamEnd < streamStart) {
        OPENVINO_THROW("Invalid stream size: streamEnd (",
                       streamEnd,
                       ") is not larger than streamStart (",
                       streamStart,
                       ")!");
    }

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
    update_log_level(rawConfig);
    Config localConfig = globalConfig;
    localConfig.update(rawConfig, mode);
    return localConfig;
}

Plugin::Plugin()
    : _options(std::make_shared<OptionsDesc>()),
      _globalConfig(_options),
      _logger("NPUPlugin", Logger::global().level()) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "Plugin::Plugin");
    set_device_name("NPU");

    // parse env_variables to get LOG_LEVEL if needed
    _globalConfig.parseEnvVars();

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

    OV_ITT_TASK_NEXT(PLUGIN, "Metrics");
    _metrics = std::make_shared<Metrics>(_backends);

    init_options(cid_ver);
    /// Init and register properties
    _properties = std::make_unique<Properties>(PropertiesType::PLUGIN, _globalConfig, _metrics);
    _properties->registerProperties();
}

void Plugin::init_options(compilerVersion comp_ver) {
    // Initialize (note: it will reset registered options)
    OV_ITT_TASK_NEXT(PLUGIN, "initOptions");
    registerOptions(*_options, comp_ver);
    _backends->registerOptions(*_options);

    // Extras
    // Additional filtering of options, based on other criteria (other than compiler version)
    if (!_backends->isCommandQueueExtSupported()) {
        // Remove options which are tied to CommandQueueExtension
        _options->remove(ov::workload_type.name());
        _options->remove(ov::intel_npu::turbo.name());
    }

    // parse again env_variables after backend is initialized to get backend proprieties
    _globalConfig.parseEnvVars();
}

void Plugin::set_property(const ov::AnyMap& properties) {
    // 1. Set the property via Properties interface
    _properties->set_property(properties);

    // 2. Extra hooks
    // Update log level if it was provided
    if (properties.count(std::string(LOG_LEVEL::key())) != 0) {
        Logger::global().setLevel(_globalConfig.get<LOG_LEVEL>());
    }
    // Init backends if needed
    if (_backends != nullptr) {
        _backends->setup(_globalConfig);
    }
}

ov::Any Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    return _properties->get_property(name, arguments);
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
            // CACHE_DIR isn't supported with NPU_USE_NPUW
            if (localProperties.count(ov::cache_dir.name()) || !_globalConfig.get<CACHE_DIR>().empty()) {
                OPENVINO_THROW("Option 'CACHE_DIR' is not supported with NPU_USE_NPUW");
            }
            return std::make_shared<ov::npuw::CompiledModel>(model->clone(), shared_from_this(), localProperties);
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
    if (casted == nullptr) {
        OPENVINO_THROW("Invalid remote context type. Can't cast to ov::intel_npu::RemoteContext type");
    }

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
    auto localConfig = merge_configs(_globalConfig, propertiesMap, OptionMode::RunTime);
    _logger.setLevel(localConfig.get<LOG_LEVEL>());
    const auto platform = _backends->getCompilationPlatform(localConfig.get<PLATFORM>(), localConfig.get<DEVICE_ID>());
    localConfig.update({{ov::intel_npu::platform.name(), platform}});
    auto device = _backends->getDevice(localConfig.get<DEVICE_ID>());

    set_batch_config(_backends->isBatchingSupported(), localConfig);

    const auto loadedFromCache = localConfig.get<LOADED_FROM_CACHE>();
    if (!loadedFromCache) {
        _logger.warning(
            "The usage of a compiled model can lead to undefined behavior. Please use OpenVINO IR instead!");
    }

    OV_ITT_TASK_NEXT(PLUGIN_IMPORT_MODEL, "parse");

    std::shared_ptr<ov::ICompiledModel> compiledModel;

    try {
        auto compiler = getCompiler(localConfig);

        auto graphSize = getFileSize(stream);

        std::vector<uint8_t> blob(graphSize);
        stream.read(reinterpret_cast<char*>(blob.data()), graphSize);
        if (!stream) {
            OPENVINO_THROW("Failed to read data from stream!");
        }
        _logger.debug("Successfully read %zu bytes into blob.", graphSize);

        auto meta = compiler->parse(blob, localConfig);
        meta.name = "net" + std::to_string(_compiledModelLoadCounter++);

        const std::shared_ptr<ov::Model> modelDummy = create_dummy_model(meta.inputs, meta.outputs);

        auto networkDescription = std::make_shared<const NetworkDescription>(std::move(blob), std::move(meta));

        compiledModel = std::make_shared<CompiledModel>(modelDummy,
                                                        shared_from_this(),
                                                        networkDescription,
                                                        device,
                                                        compiler,
                                                        localConfig);
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
    _logger.debug("performing createCompiler");
    return createCompiler(_backends, compilerType);
}

std::atomic<int> Plugin::_compiledModelLoadCounter{1};

static const ov::Version version = {CI_BUILD_NUMBER, NPU_PLUGIN_LIB_NAME};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(Plugin, version)

}  // namespace intel_npu
