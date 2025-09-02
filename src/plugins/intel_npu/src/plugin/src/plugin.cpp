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
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "npuw/compiled_model.hpp"
#include "npuw/llm_compiled_model.hpp"
#include "npuw/serialization.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
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
                                              const std::vector<IODescriptor>& outputDescriptors) {
    ov::ParameterVector parameters;
    ov::ResultVector results;

    for (const IODescriptor& inputDescriptor : inputDescriptors) {
        if (inputDescriptor.isStateInput || inputDescriptor.isStateOutput || inputDescriptor.isShapeTensor ||
            inputDescriptor.isInitInputWeights || inputDescriptor.isMainInputWeights) {
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
        if (outputDescriptor.isStateInput || outputDescriptor.isStateOutput || outputDescriptor.isShapeTensor ||
            outputDescriptor.isInitOutputWeights) {
            continue;
        }

        std::shared_ptr<ov::Node> constantDummy =
            std::make_shared<ov::op::v0::Constant>(outputDescriptor.precision, CONSTANT_NODE_DUMMY_SHAPE);

        const std::shared_ptr<ov::descriptor::Tensor>& tensorDummy = std::make_shared<ov::descriptor::Tensor>(
            outputDescriptor.precision,
            outputDescriptor.shapeFromIRModel.has_value() ? *outputDescriptor.shapeFromIRModel
                                                          : outputDescriptor.shapeFromCompiler,
            outputDescriptor.outputTensorNames);

        auto& result = results.emplace_back(std::make_shared<ov::op::v0::Result>(constantDummy));
        result->output(0).set_tensor_ptr(tensorDummy);
        result->set_friendly_name(outputDescriptor.nodeFriendlyName);
    }

    return std::make_shared<ov::Model>(results, parameters);
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

static ov::intel_npu::CompilerType resolveCompilerType(const FilteredConfig& base_conf, const ov::AnyMap& local_conf) {
    // first look if provided config changes compiler type
    auto it = local_conf.find(std::string(COMPILER_TYPE::key()));
    if (it != local_conf.end()) {
        // if compiler_type is provided by local config = use that
        return COMPILER_TYPE::parse(it->second.as<std::string>());
    }
    // if there is no compiler_type provided = use base_config value
    return base_conf.get<COMPILER_TYPE>();
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
    _logger.setLevel(_globalConfig.get<LOG_LEVEL>());

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
        _globalConfig.enable(std::move(o_name), false);       \
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
    REGISTER_OPTION(MAX_TILES);
    REGISTER_OPTION(DISABLE_VERSION_CHECK);
    REGISTER_OPTION(MODEL_PTR);
    REGISTER_OPTION(BATCH_COMPILER_MODE_SETTINGS);
    REGISTER_OPTION(TURBO);
    REGISTER_OPTION(WEIGHTLESS_BLOB);
    REGISTER_OPTION(SEPARATE_WEIGHTS_VERSION);
    REGISTER_OPTION(WS_COMPILE_CALL_NUMBER);

    if (_backend) {
        if (_backend->isCommandQueueExtSupported()) {
            REGISTER_OPTION(WORKLOAD_TYPE);
        }
        // register backend options
        _backend->registerOptions(*_options);
    }

    // parse again env_variables to update registered configs which have env vars set
    _globalConfig.parseEnvVars();

    // filter out unsupported options
    filter_config_by_compiler_support(_globalConfig);

    // NPUW properties are requested by OV Core during caching and have no effect on the NPU plugin. But we still need
    // to enable those for OV Core to query. Note: do this last to not filter them out. register npuw caching properties
    REGISTER_OPTION(NPU_USE_NPUW);
    REGISTER_OPTION(NPUW_DEVICES);
    REGISTER_OPTION(NPUW_SUBMODEL_DEVICE);
    REGISTER_OPTION(NPUW_WEIGHTS_BANK);
    REGISTER_OPTION(NPUW_WEIGHTS_BANK_ALLOC);
    REGISTER_OPTION(NPUW_ONLINE_PIPELINE);
    REGISTER_OPTION(NPUW_ONLINE_AVOID);
    REGISTER_OPTION(NPUW_ONLINE_ISOLATE);
    REGISTER_OPTION(NPUW_ONLINE_NO_FOLD);
    REGISTER_OPTION(NPUW_ONLINE_MIN_SIZE);
    REGISTER_OPTION(NPUW_ONLINE_KEEP_BLOCKS);
    REGISTER_OPTION(NPUW_ONLINE_KEEP_BLOCK_SIZE);
    REGISTER_OPTION(NPUW_FOLD);
    REGISTER_OPTION(NPUW_CWAI);
    REGISTER_OPTION(NPUW_DQ);
    REGISTER_OPTION(NPUW_DQ_FULL);
    REGISTER_OPTION(NPUW_PMM);
    REGISTER_OPTION(NPUW_SLICE_OUT);
    REGISTER_OPTION(NPUW_SPATIAL);
    REGISTER_OPTION(NPUW_SPATIAL_NWAY);
    REGISTER_OPTION(NPUW_SPATIAL_DYN);
    REGISTER_OPTION(NPUW_F16IC);
    REGISTER_OPTION(NPUW_HOST_GATHER);
    REGISTER_OPTION(NPUW_DCOFF_TYPE);
    REGISTER_OPTION(NPUW_DCOFF_SCALE);
    REGISTER_OPTION(NPUW_FUNCALL_FOR_ALL);
    REGISTER_OPTION(NPUW_FUNCALL_ASYNC);
    REGISTER_OPTION(NPUW_UNFOLD_IREQS);
    REGISTER_OPTION(NPUW_LLM);
    REGISTER_OPTION(NPUW_LLM_BATCH_DIM);
    REGISTER_OPTION(NPUW_LLM_SEQ_LEN_DIM);
    REGISTER_OPTION(NPUW_LLM_MAX_PROMPT_LEN);
    REGISTER_OPTION(NPUW_LLM_MIN_RESPONSE_LEN);
    REGISTER_OPTION(NPUW_LLM_OPTIMIZE_V_TENSORS);
    REGISTER_OPTION(NPUW_LLM_CACHE_ROPE);
    REGISTER_OPTION(NPUW_LLM_PREFILL_HINT);
    REGISTER_OPTION(NPUW_LLM_PREFILL_CONFIG);
    REGISTER_OPTION(NPUW_LLM_GENERATE_HINT);
    REGISTER_OPTION(NPUW_LLM_GENERATE_CONFIG);
}

void Plugin::filter_config_by_compiler_support(FilteredConfig& cfg) const {
    bool legacy = false;
    bool nocompiler = false;
    std::unique_ptr<ICompilerAdapter> compiler = nullptr;
    std::vector<std::string> compiler_support_list{};
    uint32_t compiler_version = 0;
    // create a dummy compiler to fetch version and supported options

    try {
        CompilerAdapterFactory compilerAdapterFactory;
        compiler = compilerAdapterFactory.getCompiler(_backend, cfg.get<COMPILER_TYPE>());
    } catch (...) {
        // assuming getCompiler failed, meaning we are offline
        _logger.warning("No available compiler. Enabling only runtime options ");
        nocompiler = true;
    }

    if (!nocompiler || (compiler != nullptr)) {
        compiler_version = compiler->get_version();
        compiler_support_list = compiler->get_supported_options();
    }
    if (compiler_support_list.size() == 0) {
        _logger.info("No compiler support options list received! Fallback to version-based option registration");
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
        // Runtime (plugin-only) options are always enabled
        if (opt.mode() == OptionMode::RunTime) {
            isEnabled = true;
        } else {  // Compiler and common options
            if (nocompiler && (opt.mode() == OptionMode::CompileTime)) {
                // we do not register compileTime options if there is no compiler
                isEnabled = false;
            } else if (legacy) {
                // Compiler or common option in Legacy mode? Checking its supported version
                if (compiler_version >= opt.compilerSupportVersion()) {
                    isEnabled = true;
                }
            } else {
                // We have compiler, we are not in legacy mode = we have a valid list of supported options
                // Searching in the list
                auto it = std::find(compiler_support_list.begin(), compiler_support_list.end(), key);
                if (it != compiler_support_list.end()) {
                    isEnabled = true;
                } else {
                    // Not found in the supported options list.
                    if (compiler != nullptr) {
                        // Checking if it is a private option?
                        isEnabled = compiler->is_option_supported(key);
                    } else {
                        // Not in the list and not a private option = disabling
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

    // Special case for NPU_TURBO which might not be supported by compiler, but driver will still use it
    // if it exists in config = driver supports it
    // if compiler->is_option_suported is false = compiler doesn't support it and gets marked disabled by default logic
    // however, if driver supports it, we still need it (and will skip giving it to compiler) = force-enable
    if (_backend && _backend->isCommandQueueExtSupported()) {
        cfg.enable(ov::intel_npu::turbo.name(), true);
    }
}

FilteredConfig Plugin::fork_local_config(const std::map<std::string, std::string>& rawConfig,
                                         const std::unique_ptr<ICompilerAdapter>& compiler,
                                         OptionMode mode) const {
    update_log_level(rawConfig);
    // create a copy of the global config
    FilteredConfig localConfig = _globalConfig;
    bool compiler_changed = false;

    // Check if compiler was changed
    // 1. Check for compiler change
    auto it = rawConfig.find(std::string(COMPILER_TYPE::key()));
    if (it != rawConfig.end()) {
        if (localConfig.getString<COMPILER_TYPE>() != it->second) {
            // Compiler type has changed!
            // Set new compiler type
            localConfig.update({{std::string(COMPILER_TYPE::key()), it->second}});
            // enable/disable config keys based on what the new compiler supports
            filter_config_by_compiler_support(localConfig);
            compiler_changed = true;
        }
    }
    // 2. Revalidate unknown internal configs
    // look for unsupported internals
    // first in what we inherited from globalconfig by forking it - ONLY if compiler has changed
    if (compiler_changed) {
        localConfig.walkInternals([&](const std::string& key) {
            if (!compiler->is_option_supported(key)) {
                OPENVINO_THROW("[ NOT_FOUND ] Option '", key, "' is not supported for current configuration");
            }
        });
    }
    // secondly, in the new config provided by user
    std::map<std::string, std::string> cfgs_to_set;
    for (const auto& [key, value] : rawConfig) {
        if (!localConfig.hasOpt(key)) {
            // not a known config key
            if (!compiler->is_option_supported(key)) {
                OPENVINO_THROW("[ NOT_FOUND ] Option '", key, "' is not supported for current configuration");
            } else {
                localConfig.addOrUpdateInternal(key, value);
            }
        } else {
            cfgs_to_set.emplace(key, value);
        }
    }

    // 3. If all good so far, update values
    localConfig.update(cfgs_to_set, mode);
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
            filter_config_by_compiler_support(_globalConfig);
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
        _backend->updateInfo(_globalConfig);
    }
}

ov::Any Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    return _properties->get_property(name, arguments);
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "Plugin::compile_model");

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
    update_log_level(localPropertiesMap);

    // create compiler
    CompilerAdapterFactory compilerAdapterFactory;
    auto compiler = compilerAdapterFactory.getCompiler(_backend, resolveCompilerType(_globalConfig, properties));

    OV_ITT_TASK_CHAIN(PLUGIN_COMPILE_MODEL, itt::domains::NPUPlugin, "Plugin::compile_model", "fork_local_config");
    auto localConfig = fork_local_config(localPropertiesMap, compiler);

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

    if (localConfig.isAvailable(ov::intel_npu::batch_mode.name()) &&
        !localConfig.has(ov::intel_npu::batch_mode.name())) {
        std::stringstream strStream;
        strStream << ov::intel_npu::BatchMode::AUTO;
        localConfig.update({{ov::intel_npu::batch_mode.name(), strStream.str()}});
    }

    if (localConfig.isAvailable(ov::intel_npu::batch_mode.name()) && !model->get_variables().empty()) {
        if (localConfig.get<BATCH_MODE>() == ov::intel_npu::BatchMode::PLUGIN) {
            OPENVINO_THROW("This model contains states, thus it is not supported when handling batching on the plugin");
        }

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

    if (localConfig.isAvailable(ov::intel_npu::weightless_blob.name()) && !localConfig.get<CACHE_DIR>().empty()) {
        // If OV caching is enabled, then weights separation is performed only if the user opted for optimizing the
        // size of the binary object
        const bool cacheModeOptimizeSize = (localConfig.get<CACHE_MODE>() == ov::CacheMode::OPTIMIZE_SIZE);
        if (localConfig.get<WEIGHTLESS_BLOB>() && !cacheModeOptimizeSize) {
            _logger.warning("The cache mode was not set to \"optimize size\" but the \"WEIGHTLESS_BLOB\" configuration "
                            "option was set to true. Weights separation WILL NOT be performed in this case.");
        } else if (!localConfig.get<WEIGHTLESS_BLOB>() && cacheModeOptimizeSize) {
            _logger.warning("The cache mode was set to \"optimize size\" but the \"WEIGHTLESS_BLOB\" configuration "
                            "option was set to false. Weights separation WILL be performed in this case.");
        }

        localConfig.update({{ov::intel_npu::weightless_blob.name(), cacheModeOptimizeSize ? "YES" : "NO"}});
    }

    std::shared_ptr<intel_npu::IGraph> graph;

    try {
        _logger.debug("performing compile");

        if (!localConfig.get<WEIGHTLESS_BLOB>()) {
            graph = compiler->compile(model->clone(), localConfig);
        } else {
            check_weightless_cache_attribute_occurrence(model);
            graph = compiler->compileWS(model->clone(), localConfig);
        }
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
    return std::make_shared<RemoteContextImpl>(_backend, remoteProperties);
}

ov::SoPtr<ov::IRemoteContext> Plugin::get_default_context(const ov::AnyMap&) const {
    return std::make_shared<RemoteContextImpl>(_backend);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& stream, const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "Plugin::import_model");

    if (properties.find(ov::hint::compiled_blob.name()) != properties.end()) {
        _logger.warning("ov::hint::compiled_blob is no longer supported for import_model(stream) API! Please use new "
                        "import_model(tensor) API instead.");
    }

    auto npu_plugin_properties = properties;
    // NPUW properties from npu_plugin_properties will be erased if import_model_npuw returns nullptr
    auto compiledModel = import_model_npuw(stream, npu_plugin_properties, shared_from_this());
    if (compiledModel) {
        return compiledModel;
    }

    try {
        const bool skipCompatibility =
            npu_plugin_properties.find(DISABLE_VERSION_CHECK::key().data()) != npu_plugin_properties.end() &&
            npu_plugin_properties[DISABLE_VERSION_CHECK::key().data()].as<bool>() == true;
        std::unique_ptr<MetadataBase> metadata = nullptr;
        size_t blobSize = MetadataBase::getFileSize(stream);
        if (!skipCompatibility) {
            // Read only metadata from the stream and check if blob is compatible. Load blob into memory only in case it
            // passes compatibility checks.
            metadata = read_metadata_from(stream);
            if (!metadata->is_compatible()) {
                OPENVINO_THROW("Incompatible blob version!");
            }
            blobSize = metadata->get_blob_size();
        }
        ov::Allocator customAllocator{utils::AlignedAllocator{utils::STANDARD_PAGE_SIZE}};
        ov::Tensor tensor(ov::element::u8, ov::Shape{blobSize}, customAllocator);
        if (blobSize > static_cast<decltype(blobSize)>(std::numeric_limits<std::streamsize>::max())) {
            OPENVINO_THROW("Blob size is too large to be represented on a std::streamsize!");
        }
        stream.read(tensor.data<char>(), static_cast<std::streamsize>(blobSize));
        auto compiledModel = parse(tensor, std::move(metadata), npu_plugin_properties);
        _logger.debug(
            "Parsed net with name: %s, having inputs:",
            std::dynamic_pointer_cast<CompiledModel>(compiledModel)->get_graph()->get_metadata().name.c_str());
        for (const auto& input :
             std::dynamic_pointer_cast<CompiledModel>(compiledModel)->get_graph()->get_metadata().inputs) {
            std::stringstream ss;
            ss << input.shapeFromCompiler.get_shape();
            _logger.debug("\t%s: %s = %d",
                          input.nameFromCompiler.c_str(),
                          ss.str().c_str(),
                          ov::shape_size(input.shapeFromCompiler.get_shape()) * input.precision.size());
        }
        _logger.debug("and outputs:");
        for (const auto& output :
             std::dynamic_pointer_cast<CompiledModel>(compiledModel)->get_graph()->get_metadata().outputs) {
            std::stringstream ss;
            ss << output.shapeFromCompiler.get_shape();
            _logger.debug("\t%s: %s = %d",
                          output.nameFromCompiler.c_str(),
                          ss.str().c_str(),
                          ov::shape_size(output.shapeFromCompiler.get_shape()) * output.precision.size());
        }
        return compiledModel;
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

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& compiled_blob,
                                                         const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "Plugin::import_model");

    // Need to create intermediate istream for NPUW
    ov::SharedStreamBuffer buffer{reinterpret_cast<char*>(compiled_blob.data()), compiled_blob.get_byte_size()};
    std::istream stream{&buffer};

    auto npu_plugin_properties = properties;
    // NPUW properties from npu_plugin_properties will be erased if import_model_npuw returns nullptr
    auto compiledModel = import_model_npuw(stream, npu_plugin_properties, shared_from_this());
    if (compiledModel) {
        return compiledModel;
    }

    try {
        const bool skipCompatibility =
            npu_plugin_properties.find(DISABLE_VERSION_CHECK::key().data()) != npu_plugin_properties.end() &&
            npu_plugin_properties[DISABLE_VERSION_CHECK::key().data()].as<bool>() == true;
        std::unique_ptr<MetadataBase> metadata = nullptr;
        size_t blobSize = compiled_blob.get_byte_size();
        if (!skipCompatibility) {
            metadata = read_metadata_from(compiled_blob);
            if (!metadata->is_compatible()) {
                OPENVINO_THROW("Incompatible blob version!");
            }
            blobSize = metadata->get_blob_size();
        }
        const ov::Tensor roiTensor(compiled_blob,
                                   ov::Coordinate{0},
                                   ov::Coordinate{blobSize});  // ROI tensor to skip NPU plugin metadata
        return parse(roiTensor, std::move(metadata), npu_plugin_properties);
    } catch (const std::exception& ex) {
        OPENVINO_THROW("Can't import network: ", ex.what());
    } catch (...) {
        OPENVINO_THROW("NPU import_model got unexpected exception from CompiledModel");
    }
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& compiled_blob,
                                                         const ov::SoPtr<ov::IRemoteContext>& context,
                                                         const ov::AnyMap& properties) const {
    auto casted = std::dynamic_pointer_cast<RemoteContextImpl>(context._ptr);
    if (casted == nullptr) {
        OPENVINO_THROW("Invalid remote context type. Can't cast to ov::intel_npu::RemoteContext type");
    }
    return import_model(compiled_blob, properties);
}

ov::SupportedOpsMap Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                        const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "Plugin::query_model");
    CompilerAdapterFactory compilerAdapterFactory;
    const std::map<std::string, std::string> propertiesMap = any_copy(properties);
    update_log_level(propertiesMap);
    auto compiler = compilerAdapterFactory.getCompiler(_backend, resolveCompilerType(_globalConfig, properties));
    auto localConfig = fork_local_config(propertiesMap, compiler, OptionMode::CompileTime);
    _logger.setLevel(localConfig.get<LOG_LEVEL>());
    const auto platform =
        utils::getCompilationPlatform(localConfig.get<PLATFORM>(),
                                      localConfig.get<DEVICE_ID>(),
                                      _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());
    localConfig.update({{ov::intel_npu::platform.name(), platform}});

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

std::shared_ptr<ov::ICompiledModel> Plugin::parse(const ov::Tensor& tensorBig,
                                                  std::unique_ptr<MetadataBase> metadata,
                                                  const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "Plugin::parse");
    CompilerAdapterFactory compilerAdapterFactory;
    const auto propertiesMap = any_copy(properties);
    update_log_level(propertiesMap);
    auto compiler = compilerAdapterFactory.getCompiler(_backend, resolveCompilerType(_globalConfig, properties));

    OV_ITT_TASK_CHAIN(PLUGIN_PARSE_MODEL, itt::domains::NPUPlugin, "Plugin::parse", "fork_local_config");
    auto localConfig = fork_local_config(propertiesMap, compiler, OptionMode::RunTime);
    _logger.setLevel(localConfig.get<LOG_LEVEL>());
    const auto platform =
        utils::getCompilationPlatform(localConfig.get<PLATFORM>(),
                                      localConfig.get<DEVICE_ID>(),
                                      _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());
    localConfig.update({{ov::intel_npu::platform.name(), platform}});
    auto device = _backend == nullptr ? nullptr : _backend->getDevice(localConfig.get<DEVICE_ID>());

    const auto loadedFromCache = localConfig.get<LOADED_FROM_CACHE>();
    if (!loadedFromCache) {
        _logger.warning(
            "The usage of a compiled model can lead to undefined behavior. Please use OpenVINO IR instead!");
    }

    uint64_t mainSize = tensorBig.get_byte_size();
    std::optional<std::vector<uint64_t>> initSizes;

    if (metadata) {
        size_t accumulator = 0;
        initSizes = metadata->get_init_sizes();
        mainSize = initSizes.has_value()
                       ? metadata->get_blob_size() - std::accumulate(initSizes->begin(), initSizes->end(), accumulator)
                       : metadata->get_blob_size();
    } else {
        _logger.info("Blob compatibility check skipped.");
    }

    ov::Tensor tensorMain(tensorBig,
                          ov::Coordinate{0},
                          ov::Coordinate{mainSize});  // ROI tensor to skip NPU plugin metadata

    std::shared_ptr<const ov::Model> originalModel;
    std::vector<ov::Tensor> tensorsInits;
    const bool weightsSeparationEnabled = initSizes.has_value();

    if (weightsSeparationEnabled) {
        // Read the init compiled models as well
        size_t cursorPosition = mainSize;
        for (uint64_t initSize : initSizes.value()) {
            ov::Tensor tensorInit(tensorBig, ov::Coordinate{cursorPosition}, ov::Coordinate{cursorPosition + initSize});
            tensorsInits.push_back(tensorInit);
            cursorPosition += initSize;
        }

        // Retrieve the ov::Model used for compilation. This is required for extracting and matching the weights
        if (properties.count(ov::hint::model.name())) {
            try {
                originalModel = properties.at(ov::hint::model.name()).as<std::shared_ptr<const ov::Model>>();
            } catch (const ov::AssertFailure&) {
                try {
                    originalModel = std::const_pointer_cast<const ov::Model>(
                        properties.at(ov::hint::model.name()).as<std::shared_ptr<ov::Model>>());
                } catch (const ov::Exception&) {
                    OPENVINO_THROW("The value of the \"ov::hint::model\" configuration option (\"MODEL_PTR\") has the "
                                   "wrong data type. Expected: std::shared_ptr<const ov::Model>.");
                }
            }
        } else if (!localConfig.get<WEIGHTS_PATH>().empty()) {
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

            originalModel = get_core()->read_model(xmlPath, weightsPath, properties);
        } else {
            OPENVINO_THROW("Attempted to load a weightless compiled model, but no weights have been provided");
        }

        check_weightless_cache_attribute_occurrence(originalModel);
    }

    auto graph = compiler->parse(std::move(tensorMain),
                                 localConfig,
                                 weightsSeparationEnabled ? std::make_optional(std::move(tensorsInits)) : std::nullopt,
                                 weightsSeparationEnabled ? std::make_optional(originalModel) : std::nullopt);
    graph->update_network_name("net" + std::to_string(_compiledModelLoadCounter++));
    const std::shared_ptr<ov::Model> modelDummy =
        create_dummy_model(graph->get_metadata().inputs, graph->get_metadata().outputs);

    OV_ITT_TASK_NEXT(PLUGIN_PARSE_MODEL, "parse");

    return std::make_shared<CompiledModel>(modelDummy, shared_from_this(), device, graph, localConfig);
}

std::atomic<int> Plugin::_compiledModelLoadCounter{1};

static const ov::Version version = {CI_BUILD_NUMBER, NPU_PLUGIN_LIB_NAME};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(Plugin, version)

}  // namespace intel_npu
