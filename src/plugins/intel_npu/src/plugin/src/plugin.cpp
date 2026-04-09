// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <fstream>
#include <numeric>

#include "compiled_model.hpp"
#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/device_helpers.hpp"
#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/common/igraph.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/common/parser_factory.hpp"
#include "intel_npu/config/npuw.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/utils.hpp"
#include "metrics.hpp"
#include "npuw/compiled_model.hpp"
#include "npuw/llm_compiled_model.hpp"
#include "npuw/serialization.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/file_util.hpp"
#include "remote_context.hpp"
#include "transformations.hpp"

namespace {
using namespace intel_npu;

const std::vector<size_t> CONSTANT_NODE_DUMMY_SHAPE{1};

const char* NPU_PLUGIN_LIB_NAME = "openvino_intel_npu_plugin";
constexpr std::string_view WEIGHTS_EXTENSION = ".bin";
constexpr std::string_view XML_EXTENSION = ".xml";
constexpr std::string_view ONNX_EXTENSION = ".onnx";

/**
 * @brief Creates an "ov::Model" object which contains only the given "parameter" and "result" nodes.
 * @details Using an "ov::Model" object to create the "CompiledModel" is the preferred way of using the OV API.
 * This path allows making use of the already written functions/attributes for handling the I/O information.
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
                                              const std::vector<IODescriptor>& outputDescriptors,
                                              const std::optional<int64_t> batchSize,
                                              const std::optional<std::vector<ov::Layout>>& inputLayouts,
                                              const std::optional<std::vector<ov::Layout>>& outputLayouts) {
    ov::ParameterVector parameters;
    ov::ResultVector results;

    for (size_t inputIndex = 0; inputIndex < inputDescriptors.size(); ++inputIndex) {
        const IODescriptor& inputDescriptor = inputDescriptors.at(inputIndex);
        if (inputDescriptor.isStateInput || inputDescriptor.isStateOutput || inputDescriptor.isShapeTensor ||
            inputDescriptor.isInitInputWeights || inputDescriptor.isMainInputWeights) {
            continue;
        }

        auto shape = inputDescriptor.shapeFromIRModel.has_value() ? *inputDescriptor.shapeFromIRModel
                                                                  : inputDescriptor.shapeFromCompiler;

        if (batchSize.has_value()) {
            shape[intel_npu::utils::BATCH_AXIS] = ov::Dimension(batchSize.value());
        }

        std::shared_ptr<ov::op::v0::Parameter> parameter =
            std::make_shared<ov::op::v0::Parameter>(inputDescriptor.precision, shape);

        parameter->set_friendly_name(inputDescriptor.nodeFriendlyName);
        parameter->output(0).get_tensor().set_names(inputDescriptor.outputTensorNames);
        if (inputLayouts.has_value()) {
            parameter->set_layout(inputLayouts->at(inputIndex));
        }
        parameters.push_back(std::move(parameter));
    }

    // The "result" nodes require a parent node in order to satisfy the API conventions. Additionally, a dummy shape for
    // the "Constant" node was required since the specific constructor does not accept "ov::PartialShape" values (a
    // constant can't have dynamic shape). The dummy tensor was also brought in order to register the correct,
    // potentially dynamic, output shape.
    for (size_t outputIndex = 0; outputIndex < outputDescriptors.size(); ++outputIndex) {
        const IODescriptor& outputDescriptor = outputDescriptors.at(outputIndex);
        if (outputDescriptor.isStateInput || outputDescriptor.isStateOutput || outputDescriptor.isShapeTensor ||
            outputDescriptor.isInitOutputWeights) {
            continue;
        }

        std::shared_ptr<ov::Node> constantDummy =
            std::make_shared<ov::op::v0::Constant>(outputDescriptor.precision, CONSTANT_NODE_DUMMY_SHAPE);

        auto shape = outputDescriptor.shapeFromIRModel.has_value() ? *outputDescriptor.shapeFromIRModel
                                                                   : outputDescriptor.shapeFromCompiler;

        if (batchSize.has_value()) {
            shape[intel_npu::utils::BATCH_AXIS] = ov::Dimension(batchSize.value());
        }

        const std::shared_ptr<ov::descriptor::Tensor>& tensorDummy =
            std::make_shared<ov::descriptor::Tensor>(outputDescriptor.precision,
                                                     shape,
                                                     outputDescriptor.outputTensorNames);

        auto& result = results.emplace_back(std::make_shared<ov::op::v0::Result>(constantDummy));
        result->output(0).set_tensor_ptr(tensorDummy);
        if (outputLayouts.has_value()) {
            result->set_layout(outputLayouts->at(outputIndex));
        }
        result->set_friendly_name(outputDescriptor.nodeFriendlyName);
    }

    return std::make_shared<ov::Model>(results, parameters);
}

/**
 * @brief Just checks if there is any "WeightlessCacheAttribute" present in the model. In the negative case, an error is
 * thrown. The weights separation flow in its current state cannot work without this attribuite.
 */
void check_weightless_cache_attribute_occurrence(const std::shared_ptr<const ov::Model>& model) {
    for (const auto& ov_node : model->get_ordered_ops()) {
        if (!ov::is_type<ov::op::v0::Constant>(ov_node)) {
            continue;
        }

        if (auto it = ov_node->get_rt_info().find(ov::WeightlessCacheAttribute::get_type_info_static());
            it != ov_node->get_rt_info().end()) {
            return;
        }
    }

    OPENVINO_THROW("No \"WeightlessCacheAttribute\" has been found in any of the model's Constant nodes. This "
                   "attribute is required for running the \"weights separation\" flow.");
}

std::shared_ptr<ov::ICompiledModel> import_model_npuw(std::istream& stream,
                                                      ov::AnyMap& properties,
                                                      std::shared_ptr<const ov::IPlugin> pluginSO) {
    // If was exported via NPUW
    auto stream_start_pos = stream.tellg();
    ov::npuw::s11n::IndicatorType serialization_indicator;
    ov::npuw::s11n::read(stream, serialization_indicator);
    if (serialization_indicator == NPUW_SERIALIZATION_INDICATOR) {
        ov::npuw::s11n::IndicatorType compiled_model_indicator;
        ov::npuw::s11n::read(stream, compiled_model_indicator);
        stream.seekg(-stream.tellg() + stream_start_pos, std::ios::cur);

        if (compiled_model_indicator == NPUW_LLM_COMPILED_MODEL_INDICATOR) {
            // Properties are required for ov::weights_path
            return ov::npuw::LLMCompiledModel::import_model(stream, pluginSO, properties);
        } else if (compiled_model_indicator == NPUW_COMPILED_MODEL_INDICATOR) {
            // Properties are required for ov::weights_path
            return ov::npuw::CompiledModel::import_model(stream, pluginSO, properties);
        } else {
            OPENVINO_THROW("Couldn't deserialize NPUW blob - fatal error!");
        }
    }
    stream.seekg(-stream.tellg() + stream_start_pos, std::ios::cur);

    // Drop NPUW properties if there are any
    for (auto it = properties.begin(); it != properties.end(); ++it) {
        if (it->first.find("NPUW") != it->first.npos) {
            properties.erase(it->first);
        }
    }
    return nullptr;
}

std::shared_ptr<const ov::Model> exclude_model_ptr_from_map(ov::AnyMap& properties) {
    std::shared_ptr<const ov::Model> modelPtr = nullptr;
    if (properties.count(ov::hint::model.name())) {
        try {
            modelPtr = properties.at(ov::hint::model.name()).as<std::shared_ptr<const ov::Model>>();
        } catch (const ov::Exception&) {
            try {
                modelPtr = std::const_pointer_cast<const ov::Model>(
                    properties.at(ov::hint::model.name()).as<std::shared_ptr<ov::Model>>());
            } catch (const ov::Exception&) {
                OPENVINO_THROW("The value of the \"ov::hint::model\" configuration option (\"MODEL_PTR\") has the "
                               "wrong data type. Expected: std::shared_ptr<const ov::Model>.");
            }
        }
        properties.erase(ov::hint::model.name());
    }
    return modelPtr;
}

std::optional<ov::EncryptionCallbacks> exclude_cache_encryption_callbacks_from_map(ov::AnyMap& properties) {
    std::optional<ov::EncryptionCallbacks> encryptionCallbacksOpt = std::nullopt;
    if (properties.count(ov::cache_encryption_callbacks.name())) {
        try {
            encryptionCallbacksOpt = properties.at(ov::cache_encryption_callbacks.name()).as<ov::EncryptionCallbacks>();
        } catch (const ov::Exception&) {
            OPENVINO_THROW("The value of the \"ov::cache_encryption_callbacks\" configuration option "
                           "(\"CACHE_ENCRYPTION_CALLBACKS\") has the "
                           "wrong data type. Expected: ov::EncryptionCallbacks.");
        }
        properties.erase(ov::cache_encryption_callbacks.name());
    }
    return encryptionCallbacksOpt;
}

void init_config(const IEngineBackend* backend, OptionsDesc& options, FilteredConfig& config) {
    // Initialize (note: it will reset registered options)
    options.reset();

#define REGISTER_OPTION(OPT_TYPE)                             \
    do {                                                      \
        auto dummyopt = details::makeOptionModel<OPT_TYPE>(); \
        std::string o_name = dummyopt.key().data();           \
        options.add<OPT_TYPE>();                              \
        config.enable(std::move(o_name), false);              \
    } while (0)

    REGISTER_OPTION(LOG_LEVEL);
    REGISTER_OPTION(CACHE_DIR);
    REGISTER_OPTION(CACHE_MODE);
    REGISTER_OPTION(COMPILED_BLOB);
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
    REGISTER_OPTION(COMPILATION_MODE);
    REGISTER_OPTION(COMPILER_TYPE);
    REGISTER_OPTION(COMPILER_VERSION);
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
    REGISTER_OPTION(QDQ_OPTIMIZATION_AGGRESSIVE);
    REGISTER_OPTION(STEPPING);
    REGISTER_OPTION(DISABLE_VERSION_CHECK);
    REGISTER_OPTION(EXPORT_RAW_BLOB);
    REGISTER_OPTION(IMPORT_RAW_BLOB);
    REGISTER_OPTION(BATCH_COMPILER_MODE_SETTINGS);
    REGISTER_OPTION(TURBO);
    REGISTER_OPTION(ENABLE_WEIGHTLESS);
    REGISTER_OPTION(SEPARATE_WEIGHTS_VERSION);
    REGISTER_OPTION(WS_COMPILE_CALL_NUMBER);
    REGISTER_OPTION(MODEL_SERIALIZER_VERSION);
    REGISTER_OPTION(ENABLE_STRIDES_FOR);
    REGISTER_OPTION(SHARED_COMMON_QUEUE);

    if (backend) {
        // Options registered only if drivers is present and supports the corresponding extension
        REGISTER_OPTION(MAX_TILES);

        if (backend->isCommandQueueExtSupported()) {
            REGISTER_OPTION(WORKLOAD_TYPE);
        }
        if (backend->isContextExtSupported()) {
            REGISTER_OPTION(DISABLE_IDLE_MEMORY_PRUNING);
        }
    }

    // parse again env_variables to update registered configs which have env vars set
    config.parseEnvVars();

    // NPUW properties are requested by OV Core during caching and have no effect on the NPU plugin. But we still need
    // to enable those for OV Core to query. Note: do this last to not filter them out. register npuw caching properties
    for_each_exposed_npuw_option([&](auto tag) {
        using Opt = typename decltype(tag)::type;
        REGISTER_OPTION(Opt);
    });

    config.enableRuntimeOptions();

    // Special cases - options with OptionMode::Both must be enabled for the plugin even if the compiler does not
    // support them, because they may be used by the plugin itself or by the driver.
    // We still check compiler support to decide whether these options should be removed from the config string.

    // NPU_TURBO might be supported by the driver
    if (backend && backend->isCommandQueueExtSupported()) {
        config.enable(ov::intel_npu::turbo.name(), true);
    }

    // LOG_LEVEL, PERFORMANCE_HINT and PERF_COUNT are needed by runtime options
    config.enable(ov::log::level.name(), true);
    config.enable(ov::hint::performance_mode.name(), true);
    config.enable(ov::enable_profiling.name(), true);

    if (config.get<COMPILER_TYPE>() == ov::intel_npu::CompilerType::PREFER_PLUGIN && backend != nullptr) {
        auto device = backend->getDevice();
        if (device) {
            auto platformName = device->getName();
            CompilerAdapterFactory compilerFactory;
            auto compileType = compilerFactory.determineAppropriateCompilerTypeBasedOnPlatform(platformName);
            if (compileType == ov::intel_npu::CompilerType::DRIVER) {
                config.update({{ov::intel_npu::compiler_type.name(), COMPILER_TYPE::toString(compileType)}});
            }
        }
    }
}

}  // namespace

namespace intel_npu {

Plugin::Plugin() : _logger("NPUPlugin", Logger::global().level()) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "Plugin::Plugin");
    set_device_name("NPU");

    std::shared_ptr<OptionsDesc> options = std::make_shared<OptionsDesc>();
    // parse env_variables to get LOG_LEVEL if needed
    options->add<LOG_LEVEL>();

    FilteredConfig config(options);
    config.parseEnvVars();
    Logger::global().setLevel(config.get<LOG_LEVEL>());
    _logger.setLevel(config.get<LOG_LEVEL>());

    OV_ITT_TASK_CHAIN(PLUGIN, itt::domains::NPUPlugin, "Plugin::Plugin", "GetBackend");
    // backend registry shall be created after configs are updated
    _backendsRegistry = std::make_unique<BackendsRegistry>();
    _backend = _backendsRegistry->getEngineBackend();

    OV_ITT_TASK_NEXT(PLUGIN, "InitConfig");
    init_config(_backend._ptr.get(), *options, config);

    if (_backend) {
        OV_ITT_TASK_NEXT(PLUGIN, "RegisterBackendOptions");
        _backend->registerOptions(*options);
    }

    OV_ITT_TASK_NEXT(PLUGIN, "CreateMetrics");
    auto metrics = std::make_shared<Metrics>(_backend);

    /// Init and register properties
    OV_ITT_TASK_NEXT(PLUGIN, "RegisterProperties");
    _propertiesManager = std::make_unique<Properties>(PropertiesType::PLUGIN, config, metrics, _backend);
    _encryptionCallbacksOpt = std::nullopt;
}

void Plugin::set_property(const ov::AnyMap& properties) {
    if (properties.empty()) {
        return;
    }

    auto npuPluginProperties = properties;
    update_log_level(npuPluginProperties);

    if (_backend != nullptr) {
        _backend->updateInfo(npuPluginProperties);
    }

    auto encryptionCallbacksOpt = exclude_cache_encryption_callbacks_from_map(npuPluginProperties);
    if (encryptionCallbacksOpt.has_value()) {
        std::lock_guard<std::mutex> encryptionCallbacksLock(_encryptionCallbacksMutex);
        _encryptionCallbacksOpt = encryptionCallbacksOpt;
    }

    _propertiesManager->setProperty(npuPluginProperties);
}

ov::Any Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    if (!arguments.empty()) {
        auto npuPluginArguments = arguments;
        exclude_model_ptr_from_map(npuPluginArguments);
        exclude_cache_encryption_callbacks_from_map(npuPluginArguments);

        // Need to create a temporary copy of the properties manager. The set of arguments we get might change the list
        // of supported properties, but we cannot alter the global state
        auto copyPropertiesManager = std::make_unique<Properties>(*_propertiesManager);
        copyPropertiesManager->setProperty(npuPluginArguments);

        return copyPropertiesManager->getProperty(name);
    }

    return _propertiesManager->getProperty(name);
}

bool Plugin::is_property_supported(const std::string& name, const ov::AnyMap& arguments) const {
    if (!arguments.empty()) {
        auto npuPluginArguments = arguments;
        exclude_model_ptr_from_map(npuPluginArguments);
        exclude_cache_encryption_callbacks_from_map(npuPluginArguments);

        // Need to create a temporary copy of the properties manager. The set of arguments we get might change the list
        // of supported properties, but we cannot alter the global state
        auto copyPropertiesManager = std::make_unique<Properties>(*_propertiesManager);

        try {
            copyPropertiesManager->setProperty(npuPluginArguments);
        } catch (...) {
            // In case of a failure during property setting, we assume the arguments are not valid and thus the
            // supported properties cannot be reliably determined - return false in this case
            return false;
        }

        return copyPropertiesManager->isPropertySupported(name);
    }

    return _propertiesManager->isPropertySupported(name);
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "Plugin::compile_model");
    update_log_level(properties);

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

    // ov::hint::model and ov::cache_encryption_callbacks have no corresponding "Config" implementation thus we need to
    // remove them from the list of properties
    if (exclude_model_ptr_from_map(localProperties)) {
        _logger.warning("Model received in config will be ignored as it was already provided by parameter.");
    }
    auto encryptionCallbacksOpt = exclude_cache_encryption_callbacks_from_map(localProperties);
    if (!encryptionCallbacksOpt.has_value()) {
        std::lock_guard<std::mutex> encryptionCallbacksLock(_encryptionCallbacksMutex);
        encryptionCallbacksOpt = _encryptionCallbacksOpt;
    }

    if (_backend != nullptr) {
        _backend->updateInfo(localProperties);
    }

    // Resolving the requested compiler type based on local and global properties.
    // It can still remain PREFER_PLUGIN even after this point
    ov::intel_npu::CompilerType compilerType = _propertiesManager->determineCompilerType(localProperties);

    auto deviceId = _propertiesManager->determineDeviceId(localProperties);
    // DEVICE_ID can be passed both as an index and as a platform name.
    // Identify the right device object to be taken into account when the target compilation platform is determined
    std::shared_ptr<IDevice> device = utils::getDeviceById(_backend, deviceId);

    // Determine the final compilation target based on NPU_PLATFORM, determined device name (if any) and the list of
    // available devices (if any)
    const auto compilationPlatform =
        utils::getCompilationPlatform(_propertiesManager->determinePlatform(localProperties),
                                      device == nullptr ? std::move(deviceId) : device->getName(),
                                      _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());

    CompilerAdapterFactory factory;
    auto compiler = factory.getCompiler(_backend, compilerType, compilationPlatform);

    localProperties[ov::intel_npu::compiler_type.name()] = compilerType;
    if (!compilationPlatform.empty()) {
        localProperties[ov::intel_npu::platform.name()] = compilationPlatform;
    }

    OV_ITT_TASK_CHAIN(PLUGIN_COMPILE_MODEL, itt::domains::NPUPlugin, "Plugin::compile_model", "fork_local_config");
    FilteredConfig localConfig = _propertiesManager->getConfigForSpecificCompiler(localProperties, compiler.get());
    localConfig.update({{ov::intel_npu::compiler_version.name(), std::to_string(compiler->get_version())}});

    auto updateBatchMode = [&](ov::intel_npu::BatchMode mode) {
        std::stringstream strStream;
        strStream << mode;
        _logger.info("Setting batching mode to %s.", strStream.str().c_str());
        localConfig.update({{ov::intel_npu::batch_mode.name(), strStream.str()}});
    };

    // Handle batch mode configuration
    std::optional<ov::Dimension> originalBatch = std::nullopt;
    std::shared_ptr<ov::Model> batchedModel;

    bool shouldHandleBatching = false;
    bool successfullyDebatched = false;

    if (localConfig.isAvailable(ov::intel_npu::batch_mode.name())) {
        // Set default batch mode if not configured
        if (!localConfig.has(ov::intel_npu::batch_mode.name())) {
            updateBatchMode(ov::intel_npu::BatchMode::AUTO);
        }

        // Handle models with variables (states)
        if (!model->get_variables().empty()) {
            if (localConfig.get<BATCH_MODE>() == ov::intel_npu::BatchMode::PLUGIN) {
                OPENVINO_THROW(
                    "This model contains states, thus it is not supported when handling batching on the plugin");
            }
            updateBatchMode(ov::intel_npu::BatchMode::COMPILER);
        }
        shouldHandleBatching = true;
    } else {
        // If the model contains states, it is not supported when handling batching on the plugin
        shouldHandleBatching = model->get_variables().empty();
    }

    if (shouldHandleBatching) {
        // Process batching
        std::tie(batchedModel, successfullyDebatched) =
            intel_npu::batch_helpers::handlePluginBatching(model, localConfig, updateBatchMode, originalBatch, _logger);
    }

    if (localConfig.has(ov::intel_npu::enable_strides_for.name())) {
        if (model->is_dynamic()) {
            OPENVINO_ASSERT(
                !intel_npu::batch_helpers::checkModelDynamicDims(model),
                "Dynamic shape tensors are not supported with the dynamic strides feature (ENABLE_STRIDES_FOR).");

            OPENVINO_ASSERT(successfullyDebatched || !localConfig.isAvailable(ov::intel_npu::batch_mode.name()) ||
                                localConfig.get<BATCH_MODE>() != ov::intel_npu::BatchMode::COMPILER,
                            "Dynamic batching is not supported with the dynamic strides feature (ENABLE_STRIDES_FOR).");
        }
    }

    // Update stepping w/ information from driver, unless provided by user or we are off-device
    // Ignore if compilation was requested for a platform that is different from the current one
    if (!localConfig.has<STEPPING>() && device != nullptr && device->getName() == compilationPlatform) {
        try {
            localConfig.update({{ov::intel_npu::stepping.name(), std::to_string(device->getSubDevId())}});
        } catch (...) {
            _logger.warning("Stepping information not implemented by selected backend. Skipping. Please provide "
                            "NPU_STEPPING if required.");
        }
    }
    // Update max_tiles w/ information from driver, unless provided by user or we are off-device
    // Ignore if compilation was requested for a platform that is different from the current one
    if (!localConfig.has<MAX_TILES>() && device != nullptr && device->getName() == compilationPlatform) {
        try {
            localConfig.update({{ov::intel_npu::max_tiles.name(), std::to_string(device->getMaxNumSlices())}});
        } catch (...) {
            _logger.warning("Max tiles information not implemented by selected backend. Default value will be used.");
        }
    }

    OV_ITT_TASK_NEXT(PLUGIN_COMPILE_MODEL, "compile");

    if (localConfig.isAvailable(ov::enable_weightless.name()) && !localConfig.get<CACHE_DIR>().empty()) {
        // If OV caching is enabled, then weights separation is performed only if the user opted for optimizing the
        // size of the binary object
        const bool cacheModeOptimizeSize = (localConfig.get<CACHE_MODE>() == ov::CacheMode::OPTIMIZE_SIZE);
        if (localConfig.get<ENABLE_WEIGHTLESS>() && !cacheModeOptimizeSize) {
            _logger.warning(
                "The cache mode was not set to \"optimize size\" but the \"ENABLE_WEIGHTLESS\" configuration option "
                "was set to true. Weights separation WILL NOT be performed in this case.");
        } else if (!localConfig.get<ENABLE_WEIGHTLESS>() && cacheModeOptimizeSize) {
            _logger.warning(
                "The cache mode was set to \"optimize size\" but the \"ENABLE_WEIGHTLESS\" configuration option "
                "was set to false. Weights separation WILL be performed in this case.");
        }

        localConfig.update({{ov::enable_weightless.name(), cacheModeOptimizeSize ? "YES" : "NO"}});
    }

    std::shared_ptr<intel_npu::IGraph> graph;

    auto compileWithConfig = [&](auto&& modelToCompile, const auto& config) {
        if (!localConfig.get<ENABLE_WEIGHTLESS>()) {
            return compiler->compile(modelToCompile, config);
        } else {
            check_weightless_cache_attribute_occurrence(model);
            return compiler->compileWS(std::move(modelToCompile), config);
        }
    };

    try {
        _logger.debug("performing compile");

        // Determine which model to use
        auto modelToCompile = successfullyDebatched ? std::move(batchedModel) : model->clone();

        const bool performanceHintSetByUser = localConfig.has(ov::hint::performance_mode.name());
        const bool shouldForceThroughput = successfullyDebatched && !performanceHintSetByUser;
        const bool shouldWarnAboutLatency = successfullyDebatched && performanceHintSetByUser &&
                                            localConfig.get<PERFORMANCE_HINT>() == ov::hint::PerformanceMode::LATENCY;

        if (shouldWarnAboutLatency) {
            _logger.warning("PERFORMANCE_HINT is explicitly set to LATENCY mode, but batch dimension (N) is "
                            "detected in the model. The NPU Plugin will reshape the model to batch size 1 and "
                            "process each batch slice separately.");
            _logger.warning("For optimal performance with batched models, THROUGHPUT mode is highly recommended, "
                            "as LATENCY mode prevents parallel batch processing.");
            _logger.warning("If batch detection appears incorrect, verify that the input and output layouts are "
                            "configured properly.");
        }

        if (shouldForceThroughput) {
            _logger.info("Setting performance mode to THROUGHPUT for batched model compilation.");

            auto modifiedConfig = localConfig;  // Copy only when needed
            std::stringstream strStream;
            strStream << ov::hint::PerformanceMode::THROUGHPUT;
            modifiedConfig.update({{ov::hint::performance_mode.name(), strStream.str()}});
            graph = compileWithConfig(std::move(modelToCompile), modifiedConfig);
        } else {
            graph = compileWithConfig(std::move(modelToCompile), localConfig);
        }
    } catch (const std::exception& ex) {
        OPENVINO_THROW(ex.what());
    } catch (...) {
        _logger.error("Unexpected exception");
        OPENVINO_THROW("NPU plugin: got an unexpected exception from compiler");
    }

    std::optional<int64_t> batch = std::nullopt;
    if (originalBatch.has_value() && successfullyDebatched) {
        batch = originalBatch.value().is_static() ? originalBatch.value().get_length() : -1;
        if (batch > 0) {
            // Initial batch setup for static cases
            graph->set_batch_size(batch.value());
        }
    }

    std::optional<decltype(std::declval<ov::EncryptionCallbacks>().encrypt)> encryptionCallback = std::nullopt;
    if (encryptionCallbacksOpt.has_value()) {
        if (encryptionCallbacksOpt->encrypt) {
            encryptionCallback = encryptionCallbacksOpt->encrypt;
        } else if (encryptionCallbacksOpt->decrypt) {
            _logger.warning("Encryption callbacks were provided for compiled model creation, but the encrypt "
                            "callback is null. Proceeding with unencrypted compilation; encrypted blob export "
                            "will be disabled.");
        }
    }

    std::shared_ptr<ov::ICompiledModel> compiledModel;
    try {
        compiledModel = std::make_shared<CompiledModel>(model,
                                                        shared_from_this(),
                                                        device,
                                                        graph,
                                                        localConfig,
                                                        batch,
                                                        encryptionCallback);
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
    auto npuPluginProperties = remoteProperties;
    exclude_model_ptr_from_map(npuPluginProperties);
    exclude_cache_encryption_callbacks_from_map(npuPluginProperties);
    return std::make_shared<RemoteContextImpl>(_backend, npuPluginProperties);
}

ov::SoPtr<ov::IRemoteContext> Plugin::get_default_context(const ov::AnyMap& remoteProperties) const {
    auto npuPluginProperties = remoteProperties;
    exclude_model_ptr_from_map(npuPluginProperties);
    exclude_cache_encryption_callbacks_from_map(npuPluginProperties);
    return std::make_shared<RemoteContextImpl>(_backend);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& stream, const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "Plugin::import_model");
    update_log_level(properties);

    if (properties.find(ov::hint::compiled_blob.name()) != properties.end()) {
        _logger.warning("ov::hint::compiled_blob is no longer supported for import_model(stream) API! Please use new "
                        "import_model(tensor) API instead.");
    }

    auto npuPluginProperties = properties;
    // NPUW properties from npuPluginProperties will be erased if import_model_npuw returns nullptr
    auto compiledModel = import_model_npuw(stream, npuPluginProperties, shared_from_this());
    if (compiledModel) {
        return compiledModel;
    }

    std::optional<ov::EncryptionCallbacks> encryptionCallbacksOpt = std::nullopt;
    try {
        encryptionCallbacksOpt =
            npuPluginProperties.count(ov::cache_encryption_callbacks.name())
                ? std::make_optional(
                      npuPluginProperties.at(ov::cache_encryption_callbacks.name()).as<ov::EncryptionCallbacks>())
                : std::nullopt;
    } catch (const ov::Exception&) {
        OPENVINO_THROW("The value of the \"ov::cache_encryption_callbacks\" configuration option "
                       "(\"CACHE_ENCRYPTION_CALLBACKS\") has the "
                       "wrong data type. Expected: ov::EncryptionCallbacks.");
    }
    if (!encryptionCallbacksOpt.has_value()) {
        std::lock_guard<std::mutex> encryptionCallbacksLock(_encryptionCallbacksMutex);
        encryptionCallbacksOpt = _encryptionCallbacksOpt;
    }

    if (_backend != nullptr) {
        _backend->updateInfo(npuPluginProperties);
    }

    try {
        const bool skipCompatibility =
            (npuPluginProperties.find(DISABLE_VERSION_CHECK::key().data()) != npuPluginProperties.end())
                ? npuPluginProperties[DISABLE_VERSION_CHECK::key().data()].as<bool>()
                : _propertiesManager->getConfig().get<DISABLE_VERSION_CHECK>();
        const bool importRawBlob =
            (npuPluginProperties.find(IMPORT_RAW_BLOB::key().data()) != npuPluginProperties.end())
                ? npuPluginProperties[IMPORT_RAW_BLOB::key().data()].as<bool>()
                : _propertiesManager->getConfig().get<IMPORT_RAW_BLOB>();
        std::unique_ptr<MetadataBase> metadata = nullptr;
        size_t blobSize = MetadataBase::getFileSize(stream);

        const bool isNotNullDecryption = (encryptionCallbacksOpt.has_value() && encryptionCallbacksOpt->decrypt);
        if (!importRawBlob && !skipCompatibility) {
            // Read only metadata from the stream and check if blob is compatible. Load blob into memory only in case it
            // passes compatibility checks.
            metadata = read_metadata_from(stream);
            blobSize = metadata->get_blob_size();
        } else {
            _logger.info("Blob compatibility check skipped.");
            if (isNotNullDecryption) {
                _logger.warning("Received decryption callback, but metadata parsing is skipped and cannot determine if "
                                "blob was encrypted or not.");
            }
        }
        OPENVINO_ASSERT(blobSize > 0, "Parsed blob size is empty from the given stream!");

        ov::Allocator customAllocator{utils::AlignedAllocator{utils::STANDARD_PAGE_SIZE}};
        ov::Tensor tensor;

        const bool isEncryptedBlobOrNullMetadata =
            (metadata == nullptr || (metadata != nullptr && metadata->is_encrypted_blob()));
        if (isNotNullDecryption && isEncryptedBlobOrNullMetadata) {
            std::string decryptedMainBlobStr;
            std::vector<std::string> decryptedInitBlobsStr;
            std::optional<std::vector<uint64_t>> initSizes;
            size_t mainSize = blobSize;
            if (metadata) {
                initSizes = metadata->get_init_sizes();
                if (initSizes.has_value()) {
                    const uint64_t sumInitSizes = std::accumulate(initSizes->begin(), initSizes->end(), uint64_t{0});
                    if (sumInitSizes > blobSize) {
                        OPENVINO_THROW("Malformed blob metadata: sum of init blob sizes (",
                                       sumInitSizes,
                                       ") exceeds total blob size (",
                                       blobSize,
                                       ")");
                    }
                    mainSize -= sumInitSizes;
                    decryptedInitBlobsStr.reserve(initSizes->size());
                }
            }

            OPENVINO_ASSERT(mainSize > 0, "Invalid main blob size!");
            if (mainSize > static_cast<decltype(mainSize)>(std::numeric_limits<std::streamsize>::max())) {
                OPENVINO_THROW("Main blob size is too large to be represented on a std::streamsize!");
            }

            {
                std::string encryptedMainBlobStr;
                encryptedMainBlobStr.resize(mainSize);  // + 1x main blob size
                const auto expectedReadSize = static_cast<std::streamsize>(mainSize);
                stream.read(&encryptedMainBlobStr.at(0), static_cast<std::streamsize>(mainSize));
                if (stream.gcount() != expectedReadSize) {
                    OPENVINO_THROW("Failed to read the full encrypted blob from stream: expected ",
                                   mainSize,
                                   " bytes, got ",
                                   stream.gcount(),
                                   " bytes");
                }
                decryptedMainBlobStr = encryptionCallbacksOpt->decrypt(encryptedMainBlobStr);  // + 2x main blob size
            }  // -1x main blob size when exiting this scope, but still additional one in ov::Tensor below
            size_t alignedSizeDecryptedMainBlob = utils::align_size_to_standard_page_size(decryptedMainBlobStr.size());
            size_t paddingSizeDecryptedMainBlob = alignedSizeDecryptedMainBlob - decryptedMainBlobStr.size();
            if (paddingSizeDecryptedMainBlob > 0) {
                _logger.warning(
                    "Decrypted main blob size was not page aligned, additional %zu bytes padding will be added",
                    paddingSizeDecryptedMainBlob);
                std::fill_n(std::back_inserter(decryptedMainBlobStr), paddingSizeDecryptedMainBlob, 0);
            }

            size_t totalDecryptedInitAlignedSizes = 0;
            if (initSizes.has_value()) {
                for (auto& initSize : initSizes.value()) {  // will also change initSizes to decrypted init blobs sizes
                    OPENVINO_ASSERT(initSize > 0, "Invalid init blob size!");
                    if (initSize > static_cast<std::remove_reference_t<decltype(initSize)>>(
                                       std::numeric_limits<std::streamsize>::max())) {
                        OPENVINO_THROW("Init size is too large to be represented on a std::streamsize!");
                    }

                    {
                        std::string encryptedInitBlobStr;
                        encryptedInitBlobStr.resize(initSize);  // + 1x init blob size
                        const auto expectedReadSize = static_cast<std::streamsize>(initSize);
                        stream.read(&encryptedInitBlobStr.at(0), static_cast<std::streamsize>(initSize));
                        if (stream.gcount() != expectedReadSize) {
                            OPENVINO_THROW("Failed to read the full encrypted init blob from stream: expected ",
                                           initSize,
                                           " bytes, got ",
                                           stream.gcount(),
                                           " bytes");
                        }
                        decryptedInitBlobsStr.push_back(
                            encryptionCallbacksOpt->decrypt(encryptedInitBlobStr));  // + 2x init blob size
                    }  // -1x init blob size when exiting this scope, but still additional one in ov::Tensor below
                    auto& decryptedInitBlobStr = decryptedInitBlobsStr.back();
                    size_t alignedSizeDecryptedInitBlob =
                        utils::align_size_to_standard_page_size(decryptedInitBlobStr.size());
                    size_t paddingSizeDecryptedInitBlob = alignedSizeDecryptedInitBlob - decryptedInitBlobStr.size();
                    if (paddingSizeDecryptedInitBlob > 0) {
                        _logger.warning(
                            "Decrypted init blob size was not page aligned, additional %zu bytes padding will be added",
                            paddingSizeDecryptedInitBlob);
                        std::fill_n(std::back_inserter(decryptedInitBlobStr), paddingSizeDecryptedInitBlob, 0);
                    }
                    initSize = alignedSizeDecryptedInitBlob;
                    totalDecryptedInitAlignedSizes += alignedSizeDecryptedInitBlob;
                }
            }

            OPENVINO_ASSERT(
                ((alignedSizeDecryptedMainBlob + totalDecryptedInitAlignedSizes) % utils::STANDARD_PAGE_SIZE) == 0,
                "Sum of decrypted main blob size and size of decrypted init blobs should be page aligned!");
            tensor = ov::Tensor(ov::element::u8,
                                ov::Shape{alignedSizeDecryptedMainBlob + totalDecryptedInitAlignedSizes},
                                customAllocator);  // + 1x main blob size + init blobs sizes
            std::memcpy(tensor.data<char>(), decryptedMainBlobStr.c_str(), decryptedMainBlobStr.size());
            ov::Tensor roiInitBlobTensor(tensor,
                                         ov::Coordinate{decryptedMainBlobStr.size()},
                                         ov::Coordinate{tensor.get_byte_size()});
            for (const auto& decryptedInitBlobStr : decryptedInitBlobsStr) {
                std::memcpy(roiInitBlobTensor.data<char>(), decryptedInitBlobStr.c_str(), decryptedInitBlobStr.size());
                roiInitBlobTensor = ov::Tensor(roiInitBlobTensor,
                                               ov::Coordinate{decryptedInitBlobStr.size()},
                                               ov::Coordinate{roiInitBlobTensor.get_byte_size()});
            }

            // parsed metadata will contain sizes of encrypted blobs, need to create an updated version of metadata with
            // sizes of decrypted blobs instead
            auto updatedMetadata = metadata != nullptr
                                       ? std::make_unique<Metadata<CURRENT_METADATA_VERSION>>(
                                             alignedSizeDecryptedMainBlob + totalDecryptedInitAlignedSizes,
                                             CURRENT_OPENVINO_VERSION,
                                             initSizes,
                                             metadata->get_batch_size(),
                                             metadata->get_input_layouts(),
                                             metadata->get_output_layouts(),
                                             /* encryptionCallbacksOpt = */ std::nullopt)
                                       : nullptr;  // no need to pass encryption information at this point
            return parse(tensor, std::move(updatedMetadata), npuPluginProperties);
        }
        if (metadata) {
            OPENVINO_ASSERT(!metadata->is_encrypted_blob(), "Cannot parse encrypted blob!");
        }

        tensor = ov::Tensor(ov::element::u8, ov::Shape{blobSize}, customAllocator);
        if (blobSize > static_cast<decltype(blobSize)>(std::numeric_limits<std::streamsize>::max())) {
            OPENVINO_THROW("Blob size is too large to be represented on a std::streamsize!");
        }
        stream.read(tensor.data<char>(), static_cast<std::streamsize>(blobSize));
        return parse(tensor, std::move(metadata), npuPluginProperties);
    } catch (const std::exception& ex) {
        OPENVINO_THROW("Can't import network: ", ex.what());
    } catch (...) {
        OPENVINO_THROW("NPU import_model got unexpected exception from CompiledModel");
    }
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

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& compiledBlob,
                                                         const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "Plugin::import_model");
    update_log_level(properties);

    // Need to create intermediate istream for NPUW
    ov::SharedStreamBuffer buffer{compiledBlob.data(), compiledBlob.get_byte_size()};
    std::istream stream{&buffer};

    auto npuPluginProperties = properties;
    // NPUW properties from npuPluginProperties will be erased if import_model_npuw returns nullptr
    auto compiledModel = import_model_npuw(stream, npuPluginProperties, shared_from_this());
    if (compiledModel) {
        return compiledModel;
    }

    std::optional<ov::EncryptionCallbacks> encryptionCallbacksOpt = std::nullopt;
    try {
        encryptionCallbacksOpt =
            npuPluginProperties.count(ov::cache_encryption_callbacks.name())
                ? std::make_optional(
                      npuPluginProperties.at(ov::cache_encryption_callbacks.name()).as<ov::EncryptionCallbacks>())
                : std::nullopt;
    } catch (const ov::Exception&) {
        OPENVINO_THROW("The value of the \"ov::cache_encryption_callbacks\" configuration option "
                       "(\"CACHE_ENCRYPTION_CALLBACKS\") has the "
                       "wrong data type. Expected: ov::EncryptionCallbacks.");
    }
    if (!encryptionCallbacksOpt.has_value()) {
        std::lock_guard<std::mutex> encryptionCallbacksLock(_encryptionCallbacksMutex);
        encryptionCallbacksOpt = _encryptionCallbacksOpt;
    }

    if (_backend != nullptr) {
        _backend->updateInfo(npuPluginProperties);
    }

    try {
        const bool skipCompatibility =
            (npuPluginProperties.find(DISABLE_VERSION_CHECK::key().data()) != npuPluginProperties.end())
                ? npuPluginProperties[DISABLE_VERSION_CHECK::key().data()].as<bool>()
                : _propertiesManager->getConfig().get<DISABLE_VERSION_CHECK>();
        const bool importRawBlob =
            (npuPluginProperties.find(IMPORT_RAW_BLOB::key().data()) != npuPluginProperties.end())
                ? npuPluginProperties[IMPORT_RAW_BLOB::key().data()].as<bool>()
                : _propertiesManager->getConfig().get<IMPORT_RAW_BLOB>();
        std::unique_ptr<MetadataBase> metadata = nullptr;
        size_t blobSize = compiledBlob.get_byte_size();

        const bool isNotNullDecryption = (encryptionCallbacksOpt.has_value() && encryptionCallbacksOpt->decrypt);
        if (!importRawBlob && !skipCompatibility) {
            metadata = read_metadata_from(compiledBlob);
            blobSize = metadata->get_blob_size();
        } else {
            _logger.info("Blob compatibility check skipped.");
            if (isNotNullDecryption) {
                _logger.warning("Received decryption callback, but metadata parsing is skipped and cannot determine if "
                                "blob was encrypted or not.");
            }
        }
        OPENVINO_ASSERT(blobSize > 0, "Parsed blob size is empty from the given buffer!");
        const ov::Tensor roiTensor(compiledBlob,
                                   ov::Coordinate{0},
                                   ov::Coordinate{blobSize});  // ROI tensor to skip NPU plugin metadata

        const bool isEncryptedBlobOrNullMetadata =
            (metadata == nullptr || (metadata != nullptr && metadata->is_encrypted_blob()));
        if (isNotNullDecryption && isEncryptedBlobOrNullMetadata) {
            std::string decryptedMainBlobStr;
            std::vector<std::string> decryptedInitBlobsStr;
            std::optional<std::vector<uint64_t>> initSizes;
            size_t mainSize = blobSize;
            if (metadata) {
                initSizes = metadata->get_init_sizes();
                if (initSizes.has_value()) {
                    const uint64_t sumInitSizes = std::accumulate(initSizes->begin(), initSizes->end(), uint64_t{0});
                    if (sumInitSizes > blobSize) {
                        OPENVINO_THROW("Malformed blob metadata: sum of init blob sizes (",
                                       sumInitSizes,
                                       ") exceeds total blob size (",
                                       blobSize,
                                       ")");
                    }
                    mainSize -= sumInitSizes;
                    decryptedInitBlobsStr.reserve(initSizes->size());
                }
            }

            OPENVINO_ASSERT(mainSize > 0, "Invalid main blob size!");
            OPENVINO_ASSERT(mainSize <= roiTensor.get_byte_size(),
                            "Cannot read ",
                            mainSize,
                            " bytes from available ",
                            roiTensor.get_byte_size(),
                            " of the given buffer!");

            {
                std::string encryptedMainBlobStr(roiTensor.data<const char>(), mainSize);      // + 1x main blob size
                decryptedMainBlobStr = encryptionCallbacksOpt->decrypt(encryptedMainBlobStr);  // + 2x main blob size
            }  // -1x main blob size when exiting this scope, but still additional one in ov::Tensor below
            size_t alignedSizeDecryptedMainBlob = utils::align_size_to_standard_page_size(decryptedMainBlobStr.size());
            size_t paddingSizeDecryptedMainBlob = alignedSizeDecryptedMainBlob - decryptedMainBlobStr.size();
            if (paddingSizeDecryptedMainBlob > 0) {
                _logger.warning(
                    "Decrypted main blob size was not page aligned, additional %zu bytes padding will be added",
                    paddingSizeDecryptedMainBlob);
                std::fill_n(std::back_inserter(decryptedMainBlobStr), paddingSizeDecryptedMainBlob, 0);
            }

            size_t totalDecryptedInitAlignedSizes = 0;
            if (initSizes.has_value()) {
                ov::Tensor roiInitBlobTensor(roiTensor,
                                             ov::Coordinate{mainSize},
                                             ov::Coordinate{roiTensor.get_byte_size()});
                for (auto& initSize : initSizes.value()) {  // will also change initSizes to decrypted init blobs sizes
                    OPENVINO_ASSERT(initSize > 0, "Invalid init blob size!");
                    OPENVINO_ASSERT(initSize <= roiInitBlobTensor.get_byte_size(),
                                    "Cannot read ",
                                    initSize,
                                    " bytes from available ",
                                    roiInitBlobTensor.get_byte_size(),
                                    " of the given buffer!");

                    {
                        std::string encryptedInitBlobStr(roiInitBlobTensor.data<const char>(),
                                                         initSize);  // + 1x init blob size
                        decryptedInitBlobsStr.push_back(
                            encryptionCallbacksOpt->decrypt(encryptedInitBlobStr));  // + 2x init blob size
                    }  // -1x init blob size when exiting this scope, but still additional one in ov::Tensor below
                    auto& decryptedInitBlobStr = decryptedInitBlobsStr.back();
                    size_t alignedSizeDecryptedInitBlob =
                        utils::align_size_to_standard_page_size(decryptedInitBlobStr.size());
                    size_t paddingSizeDecryptedInitBlob = alignedSizeDecryptedInitBlob - decryptedInitBlobStr.size();
                    if (paddingSizeDecryptedInitBlob > 0) {
                        _logger.warning(
                            "Decrypted init blob size was not page aligned, additional %zu bytes padding will be added",
                            paddingSizeDecryptedInitBlob);
                        std::fill_n(std::back_inserter(decryptedInitBlobStr), paddingSizeDecryptedInitBlob, 0);
                    }
                    roiInitBlobTensor = ov::Tensor(roiInitBlobTensor,
                                                   ov::Coordinate{initSize},
                                                   ov::Coordinate{roiInitBlobTensor.get_byte_size()});
                    initSize = alignedSizeDecryptedInitBlob;
                    totalDecryptedInitAlignedSizes += alignedSizeDecryptedInitBlob;
                }
            }

            OPENVINO_ASSERT(
                ((alignedSizeDecryptedMainBlob + totalDecryptedInitAlignedSizes) % utils::STANDARD_PAGE_SIZE) == 0,
                "Sum of decrypted main blob size and size of decrypted init blobs should be page aligned!");
            ov::Allocator customAllocator{utils::AlignedAllocator{utils::STANDARD_PAGE_SIZE}};
            ov::Tensor tensor(ov::element::u8,
                              ov::Shape{alignedSizeDecryptedMainBlob + totalDecryptedInitAlignedSizes},
                              customAllocator);  // + 1x main blob size + init blobs sizes
            std::memcpy(tensor.data<char>(), decryptedMainBlobStr.c_str(), decryptedMainBlobStr.size());
            ov::Tensor roiInitBlobTensor(tensor,
                                         ov::Coordinate{decryptedMainBlobStr.size()},
                                         ov::Coordinate{tensor.get_byte_size()});
            for (const auto& decryptedInitBlobStr : decryptedInitBlobsStr) {
                std::memcpy(roiInitBlobTensor.data<char>(), decryptedInitBlobStr.c_str(), decryptedInitBlobStr.size());
                roiInitBlobTensor = ov::Tensor(roiInitBlobTensor,
                                               ov::Coordinate{decryptedInitBlobStr.size()},
                                               ov::Coordinate{roiInitBlobTensor.get_byte_size()});
            }

            // parsed metadata will contain sizes of encrypted blobs, need to create an updated version of metadata with
            // sizes of decrypted blobs instead
            auto updatedMetadata = metadata != nullptr
                                       ? std::make_unique<Metadata<CURRENT_METADATA_VERSION>>(
                                             alignedSizeDecryptedMainBlob + totalDecryptedInitAlignedSizes,
                                             CURRENT_OPENVINO_VERSION,
                                             initSizes,
                                             metadata->get_batch_size(),
                                             metadata->get_input_layouts(),
                                             metadata->get_output_layouts(),
                                             /* encryptionCallbacksOpt = */ std::nullopt)
                                       : nullptr;  // no need to pass encryption information at this point
            return parse(tensor, std::move(updatedMetadata), npuPluginProperties);
        }
        if (metadata) {
            OPENVINO_ASSERT(!metadata->is_encrypted_blob(), "Cannot parse encrypted blob!");
        }

        return parse(roiTensor, std::move(metadata), npuPluginProperties);
    } catch (const std::exception& ex) {
        OPENVINO_THROW("Can't import network: ", ex.what());
    } catch (...) {
        OPENVINO_THROW("NPU import_model got unexpected exception from CompiledModel");
    }
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& compiledBlob,
                                                         const ov::SoPtr<ov::IRemoteContext>& context,
                                                         const ov::AnyMap& properties) const {
    auto casted = std::dynamic_pointer_cast<RemoteContextImpl>(context._ptr);
    if (casted == nullptr) {
        OPENVINO_THROW("Invalid remote context type. Can't cast to ov::intel_npu::RemoteContext type");
    }
    return import_model(compiledBlob, properties);
}

ov::SupportedOpsMap Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                        const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "Plugin::query_model");
    update_log_level(properties);

    auto localProperties = properties;
    exclude_model_ptr_from_map(localProperties);
    exclude_cache_encryption_callbacks_from_map(localProperties);

    if (_backend != nullptr) {
        _backend->updateInfo(localProperties);
    }

    ov::intel_npu::CompilerType compilerType = _propertiesManager->determineCompilerType(localProperties);
    auto deviceId = _propertiesManager->determineDeviceId(localProperties);

    std::shared_ptr<IDevice> device = utils::getDeviceById(_backend, deviceId);

    const auto compilationPlatform =
        utils::getCompilationPlatform(_propertiesManager->determinePlatform(localProperties),
                                      device == nullptr ? std::move(deviceId) : device->getName(),
                                      _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());

    CompilerAdapterFactory factory;
    auto compiler = factory.getCompiler(_backend, compilerType, compilationPlatform);

    localProperties[ov::intel_npu::compiler_type.name()] = compilerType;
    if (!compilationPlatform.empty()) {
        localProperties[ov::intel_npu::platform.name()] = compilationPlatform;
    }

    FilteredConfig localConfig = _propertiesManager->getConfigForSpecificCompiler(localProperties, compiler.get());
    ov::SupportedOpsMap supportedOpsMap;
    try {
        supportedOpsMap = compiler->query(model->clone(), localConfig);
    } catch (const std::runtime_error& e) {
        OPENVINO_THROW(e.what());
    } catch (...) {
        OPENVINO_THROW("NPU query_model got unexpected error from compiler");
    }

    return supportedOpsMap;
}

std::shared_ptr<ov::ICompiledModel> Plugin::parse(const ov::Tensor& tensorBig,
                                                  std::unique_ptr<MetadataBase> metadata,
                                                  const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "Plugin::parse");

    auto localProperties = properties;

    // ov::hint::model and ov::cache_encryption_callbacks have no corresponding "Config" implementation thus we need to
    // remove them from the list of properties
    auto originalModel = exclude_model_ptr_from_map(localProperties);
    auto encryptionCallbacksOpt = exclude_cache_encryption_callbacks_from_map(localProperties);
    if (!encryptionCallbacksOpt.has_value()) {
        std::lock_guard<std::mutex> encryptionCallbacksLock(_encryptionCallbacksMutex);
        encryptionCallbacksOpt = _encryptionCallbacksOpt;
    }

    std::shared_ptr<IDevice> device =
        utils::getDeviceById(_backend, _propertiesManager->determineDeviceId(localProperties));

    if (_backend == nullptr || device == nullptr) {
        OPENVINO_THROW("Device not found.");
    }

    OV_ITT_TASK_CHAIN(PLUGIN_PARSE_MODEL, itt::domains::NPUPlugin, "Plugin::parse", "fork_local_config");
    FilteredConfig localConfig = _propertiesManager->getConfigWithCompilerPropertiesDisabled(localProperties);

    const auto loadedFromCache = localConfig.get<LOADED_FROM_CACHE>();
    if (!loadedFromCache) {
        _logger.warning(
            "The usage of a compiled model can lead to undefined behavior. Please use OpenVINO IR instead!");
    }

    uint64_t mainSize = tensorBig.get_byte_size();
    std::optional<std::vector<uint64_t>> initSizes;
    std::optional<int64_t> batchSize = std::nullopt;

    if (metadata) {
        size_t accumulator = 0;
        initSizes = metadata->get_init_sizes();
        mainSize = initSizes.has_value()
                       ? metadata->get_blob_size() - std::accumulate(initSizes->begin(), initSizes->end(), accumulator)
                       : metadata->get_blob_size();
        batchSize = metadata->get_batch_size();

        std::optional<uint32_t> compilerVersion = metadata->get_compiler_version();
        if (compilerVersion.has_value()) {
            localConfig.update({{ov::intel_npu::compiler_version.name(), std::to_string(compilerVersion.value())}});
            _logger.debug("Imported model was compiled with compiler version: %u.%u",
                          ONEAPI_VERSION_MAJOR(compilerVersion.value()),
                          ONEAPI_VERSION_MINOR(compilerVersion.value()));
        }
    } else {
        _logger.warning(
            "Metadata parsing is skipped, if this is a weightless blob, init schedules cannot be parsed from it!");
    }

    const ov::Tensor tensorMain(tensorBig,
                                ov::Coordinate{0},
                                ov::Coordinate{mainSize});  // ROI tensor to skip NPU plugin metadata

    std::vector<ov::Tensor> tensorsInits;
    const bool weightsSeparationEnabled = initSizes.has_value();

    if (weightsSeparationEnabled) {
        // Read the init compiled models as well
        size_t cursorPosition = mainSize;
        for (uint64_t initSize : initSizes.value()) {
            const ov::Tensor tensorInit(tensorBig,
                                        ov::Coordinate{cursorPosition},
                                        ov::Coordinate{cursorPosition + initSize});
            tensorsInits.push_back(tensorInit);
            cursorPosition += initSize;
        }

        // Retrieve the ov::Model used for compilation. This is required for extracting and matching the weights
        if (!originalModel) {
            if (!localConfig.get<WEIGHTS_PATH>().empty()) {
                const std::string weightsPath = localConfig.get<WEIGHTS_PATH>();
                const size_t weightsPathLength = weightsPath.length();
                std::string xmlPath = weightsPath;

                if (weightsPathLength > WEIGHTS_EXTENSION.length() &&
                    weightsPath.compare(weightsPathLength - WEIGHTS_EXTENSION.length(),
                                        WEIGHTS_EXTENSION.length(),
                                        WEIGHTS_EXTENSION) == 0) {
                    xmlPath.replace(weightsPathLength - WEIGHTS_EXTENSION.length(),
                                    WEIGHTS_EXTENSION.length(),
                                    XML_EXTENSION);
                } else if (weightsPathLength <= ONNX_EXTENSION.length() ||
                           weightsPath.compare(weightsPathLength - ONNX_EXTENSION.length(),
                                               ONNX_EXTENSION.length(),
                                               ONNX_EXTENSION)) {
                    OPENVINO_THROW("Invalid path to the weights: ",
                                   weightsPath,
                                   ". A \".bin\" or \".onnx\" extension was expected.");
                }

                originalModel =
                    get_core()->read_model(ov::util::make_path(xmlPath), ov::util::make_path(weightsPath), properties);
            } else {
                OPENVINO_THROW("Attempted to load a weightless compiled model, but no weights have been provided");
            }
        }

        check_weightless_cache_attribute_occurrence(originalModel);
    }

    const std::optional<std::vector<ov::Tensor>> initBlobs =
        weightsSeparationEnabled ? std::make_optional(std::move(tensorsInits)) : std::nullopt;

    // Special case for PERF_COUNT as it requires compiler_type detection in case it is still set to PREFER_PLUGIN
    if (localConfig.has<PERF_COUNT>() && localConfig.get<PERF_COUNT>() &&
        localConfig.get<COMPILER_TYPE>() == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
        ov::intel_npu::CompilerType compilerType = localConfig.get<COMPILER_TYPE>();
        CompilerAdapterFactory factory;
        (void)factory.getCompiler(_backend, compilerType, device->getName());

        localConfig.update({{ov::intel_npu::compiler_type.name(), COMPILER_TYPE::toString(compilerType)}});
    }

    ParserFactory parserFactory;
    auto parser = parserFactory.getParser(_backend->getInitStructs());
    auto graph = parser->parse(tensorMain,
                               localConfig,
                               initBlobs,
                               weightsSeparationEnabled ? std::make_optional(std::move(originalModel)) : std::nullopt);

    graph->update_network_name("net" + std::to_string(_compiledModelLoadCounter++));
    const std::shared_ptr<ov::Model> modelDummy =
        create_dummy_model(graph->get_metadata().inputs,
                           graph->get_metadata().outputs,
                           batchSize,
                           metadata ? metadata->get_input_layouts() : std::nullopt,
                           metadata ? metadata->get_output_layouts() : std::nullopt);

    if (batchSize.has_value()) {
        if (batchSize.value() > 0) {
            // Initial batch setup for static cases
            graph->set_batch_size(batchSize.value());
        }
    }

    std::optional<decltype(std::declval<ov::EncryptionCallbacks>().encrypt)> encryptionCallback = std::nullopt;
    if (encryptionCallbacksOpt.has_value()) {
        encryptionCallback = encryptionCallbacksOpt->encrypt;
        if (!encryptionCallback.value()) {
            // User might give nullptr for encryption callback when importing
            encryptionCallback = std::nullopt;
        }
    }

    OV_ITT_TASK_NEXT(PLUGIN_PARSE_MODEL, "parse");

    return std::make_shared<CompiledModel>(modelDummy,
                                           shared_from_this(),
                                           device,
                                           graph,
                                           localConfig,
                                           batchSize,
                                           encryptionCallback);
}

void Plugin::update_log_level(const ov::AnyMap& properties) const {
    if (properties.count(ov::log::level.name()) != 0) {
        Logger::global().setLevel(properties.at(ov::log::level.name()).as<ov::log::Level>());
        _logger.setLevel(properties.at(ov::log::level.name()).as<ov::log::Level>());
    }
}

std::atomic<int> Plugin::_compiledModelLoadCounter{1};

static const ov::Version version = {CI_BUILD_NUMBER, NPU_PLUGIN_LIB_NAME};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(Plugin, version)

}  // namespace intel_npu
