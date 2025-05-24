// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <fstream>
#include <transformations/common_optimizations/add_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/batch_to_space_fusion.hpp>
#include <transformations/common_optimizations/conv_mul_fusion.hpp>
#include <transformations/common_optimizations/convert_quantize_dequantize.hpp>
#include <transformations/common_optimizations/depth_to_space_fusion.hpp>
#include <transformations/common_optimizations/dropout_with_random_uniform_replacer.hpp>
#include <transformations/common_optimizations/fq_mul_fusion.hpp>
#include <transformations/common_optimizations/lin_op_sequence_fusion.hpp>
#include <transformations/common_optimizations/moc_transformations.hpp>
#include <transformations/common_optimizations/mul_conv_fusion.hpp>
#include <transformations/common_optimizations/mul_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/mvn_fusion.hpp>
#include <transformations/common_optimizations/pad_fusion.hpp>
#include <transformations/common_optimizations/pull_through_reduce.hpp>
#include <transformations/common_optimizations/reduce_reshape_fusion.hpp>
#include <transformations/common_optimizations/relu_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/rms_fusion.hpp>
#include <transformations/common_optimizations/shared_ops_optimization.hpp>
#include <transformations/common_optimizations/shuffle_channels_fusion.hpp>
#include <transformations/common_optimizations/space_to_batch_fusion.hpp>
#include <transformations/common_optimizations/strides_optimization.hpp>
#include <transformations/common_optimizations/transpose_to_reshape.hpp>
#include <transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp>
#include <transformations/control_flow/unroll_if.hpp>
#include <transformations/control_flow/unroll_tensor_iterator.hpp>
#include <transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/mark_dequantization_subgraph.hpp>
#include <transformations/op_conversions/batch_norm_decomposition.hpp>
#include <transformations/op_conversions/bidirectional_sequences_decomposition.hpp>
#include <transformations/op_conversions/convert_avgpool_downgrade.hpp>
#include <transformations/op_conversions/convert_broadcast_to_tiles.hpp>
#include <transformations/op_conversions/convert_convertlike.hpp>
#include <transformations/op_conversions/convert_deformable_conv_v8_to_v1.hpp>
#include <transformations/op_conversions/convert_gather_upgrade.hpp>
#include <transformations/op_conversions/convert_interpolate11_downgrade.hpp>
#include <transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp>
#include <transformations/op_conversions/convert_maxpool_downgrade.hpp>
#include <transformations/op_conversions/convert_nms9_to_nms_ie_internal.hpp>
#include <transformations/op_conversions/convert_pad12_downgrade.hpp>
#include <transformations/op_conversions/convert_pad_to_group_conv.hpp>
#include <transformations/op_conversions/convert_previous_nms_to_nms_9.hpp>
#include <transformations/op_conversions/convert_reduce_to_pooling.hpp>
#include <transformations/op_conversions/convert_scatter_elements_update12_downgrade.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>
#include <transformations/op_conversions/convert_shapeof3.hpp>
#include <transformations/op_conversions/convert_slice_to_strided_slice.hpp>
#include <transformations/op_conversions/convert_softmax_upgrade.hpp>
#include <transformations/op_conversions/convert_topk11_downgrade.hpp>
#include <transformations/op_conversions/detection_output_downgrade.hpp>
#include <transformations/op_conversions/einsum_decomposition.hpp>
#include <transformations/op_conversions/gelu7_downgrade.hpp>
#include <transformations/op_conversions/group_normalization_decomposition.hpp>
#include <transformations/op_conversions/log_softmax_decomposition.hpp>
#include <transformations/op_conversions/normalize_l2_decomposition.hpp>
#include <transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp>
#include <transformations/op_conversions/softmax_decomposition.hpp>
#include <transformations/rt_info/fused_names_attribute.hpp>
#include <transformations/utils/utils.hpp>

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
constexpr std::string_view WEIGHTS_EXTENSION = ".bin";
constexpr std::string_view XML_EXTENSION = ".xml";
constexpr std::string_view ONNX_EXTENSION = ".onnx";

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
                                              const std::vector<IODescriptor>& outputDescriptors,
                                              const bool benchmarkInit = false) {
    ov::ParameterVector parameters;
    ov::ResultVector results;

    for (const IODescriptor& inputDescriptor : inputDescriptors) {
        if (!benchmarkInit) {
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
            parameters.push_back(parameter);
        } else {
            if (inputDescriptor.isStateInput || inputDescriptor.isStateOutput || inputDescriptor.isShapeTensor ||
                inputDescriptor.isMainInputWeights) {
                continue;
            }

            std::shared_ptr<ov::op::v0::Parameter> parameter = std::make_shared<ov::op::v0::Parameter>(
                inputDescriptor.precision,
                inputDescriptor.shapeFromIRModel.has_value() ? *inputDescriptor.shapeFromIRModel
                                                             : inputDescriptor.shapeFromCompiler);
            parameter->set_friendly_name(inputDescriptor.nameFromCompiler);
            parameter->output(0).get_tensor().set_names(
                std::unordered_set<std::string>{inputDescriptor.nameFromCompiler});
            parameters.push_back(std::move(parameter));
        }
    }

    // The "result" nodes require a parent node in order to satisfy the API conventions. Additionally, a dummy shape for
    // the "Constant" node was required since the specific constructor does not accept "ov::PartialShape" values (a
    // constant can't have dynamic shape). The dummy tensor was also brought in order to register the correct,
    // potentially dynamic, output shape.
    for (const IODescriptor& outputDescriptor : outputDescriptors) {
        if (!benchmarkInit) {
            if (outputDescriptor.isStateInput || outputDescriptor.isStateOutput || outputDescriptor.isShapeTensor ||
                outputDescriptor.isInitOutputWeights) {
                continue;
            }

            std::shared_ptr<ov::Node> constantDummy =
                std::make_shared<ov::op::v0::Constant>(outputDescriptor.precision, CONSTANT_NODE_DUMMY_SHAPE);

            const std::shared_ptr<ov::descriptor::Tensor>& tensorDummy =
                std::make_shared<ov::descriptor::Tensor>(outputDescriptor.precision,
                                                         outputDescriptor.shapeFromCompiler,
                                                         outputDescriptor.outputTensorNames);

            auto& result = results.emplace_back(std::make_shared<ov::op::v0::Result>(constantDummy));
            result->output(0).set_tensor_ptr(tensorDummy);

            result->set_friendly_name(outputDescriptor.nodeFriendlyName);
        } else {
            if (outputDescriptor.isStateInput || outputDescriptor.isStateOutput || outputDescriptor.isShapeTensor) {
                continue;
            }

            std::shared_ptr<ov::Node> constantDummy =
                std::make_shared<ov::op::v0::Constant>(outputDescriptor.precision, CONSTANT_NODE_DUMMY_SHAPE);

            const std::shared_ptr<ov::descriptor::Tensor>& tensorDummy = std::make_shared<ov::descriptor::Tensor>(
                outputDescriptor.precision,
                outputDescriptor.shapeFromCompiler,
                std::unordered_set<std::string>{outputDescriptor.nameFromCompiler});

            auto& result = results.emplace_back(std::make_shared<ov::op::v0::Result>(constantDummy));
            result->output(0).set_tensor_ptr(tensorDummy);

            result->set_friendly_name(outputDescriptor.nameFromCompiler);
        }
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

struct ImportDataWs {
    std::vector<uint8_t> mainBlob;
    std::vector<std::vector<uint8_t>> initBlobs;
};

void runOVPasses(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();

    ov::element::TypeVector decompression_precisions{ov::element::u4,
                                                     ov::element::i4,
                                                     ov::element::nf4,
                                                     ov::element::u8,
                                                     ov::element::i8,
                                                     ov::element::f8e4m3,
                                                     ov::element::f8e5m2,
                                                     ov::element::f8e8m0};
    manager.register_pass<ov::pass::MarkDequantization>(decompression_precisions, /*fold_subtract_const=*/true);
    manager.register_pass<ov::pass::KeepConstPrecision>(decompression_precisions, /*fold_subtract_const=*/true);
    manager.register_pass<ov::pass::SharedOpOptimization>();
    manager.register_pass<ov::pass::ConvertQuantizeDequantize>();
    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<ov::pass::ConvertScatterElementsUpdate12ToScatterElementsUpdate3>();
    manager.register_pass<ov::pass::ConvertInterpolate1ToInterpolate4>();
    manager.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
    manager.register_pass<ov::pass::ConvertTopK11ToTopK3>();
    manager.register_pass<ov::pass::ConvertPad12ToPad1>();
    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<ov::pass::SliceToStridedSlice>(true);
    manager.register_pass<ov::pass::MOCTransformations>(true, false);

    auto pass_config = manager.get_pass_config();
    pass_config->disable<ov::pass::PadFusionConvolution>();
    pass_config->disable<ov::pass::PadFusionGroupConvolution>();
    pass_config->disable<ov::pass::MVNFusionWithConstantsInside>();
    pass_config->disable<ov::pass::PullThroughReduce>();
    pass_config->disable<ov::pass::AddFakeQuantizeFusion>();
    pass_config->disable<ov::pass::FakeQuantizeMulFusion>();
    pass_config->disable<ov::pass::MulFakeQuantizeFusion>();

    manager.register_pass<ov::pass::ConvertNMS1ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS3ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS4ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS5ToNMS9>();

    auto static_shape = manager.register_pass<ov::pass::GraphRewrite>();
    static_shape->add_matcher<ov::pass::ConvertNMS9ToNMSIEInternal>();
    static_shape->set_name("ov::pass::CommonStaticShape");

    auto common_fusions = manager.register_pass<ov::pass::GraphRewrite>();
    common_fusions->add_matcher<ov::pass::DepthToSpaceFusion>();
    common_fusions->add_matcher<ov::pass::ShuffleChannelsFusion>(false);
    common_fusions->add_matcher<ov::pass::SpaceToBatchFusion>();
    common_fusions->add_matcher<ov::pass::BatchToSpaceFusion>();
    common_fusions->add_matcher<ov::pass::TransposeToReshape>();
    common_fusions->add_matcher<ov::pass::RMSFusion>();
    common_fusions->set_name("ov::pass::CommonFusions");

    auto decomp = manager.register_pass<ov::pass::GraphRewrite>();
    decomp->add_matcher<ov::pass::Gelu7Downgrade>();
    decomp->add_matcher<ov::pass::BidirectionalGRUSequenceDecomposition>();
    decomp->add_matcher<ov::pass::BidirectionalRNNSequenceDecomposition>();
    decomp->add_matcher<ov::pass::ConvertBroadcastToTiles>();
    decomp->add_matcher<ov::pass::ConvertConvertLike>();
    decomp->add_matcher<ov::pass::BatchNormDecomposition>();
    decomp->add_matcher<ov::pass::EinsumDecomposition>();
    decomp->add_matcher<ov::pass::DropoutWithRandomUniformReplacer>();
    decomp->add_matcher<ov::pass::ScaledDotProductAttentionDecomposition>();
    decomp->add_matcher<ov::pass::GroupNormalizationDecomposition>();
    decomp->set_name("ov::pass::CommonDecompositions");

    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<ov::pass::LinOpSequenceFusion>();
    manager.register_pass<ov::pass::UnrollIf>();

    auto conv_fusions = manager.register_pass<ov::pass::GraphRewrite>();
    conv_fusions->add_matcher<ov::pass::ConvolutionMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::GroupConvolutionMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::ConvolutionBackpropDataMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::GroupConvolutionBackpropDataMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::MultiplyConvolutionFusion>();
    conv_fusions->add_matcher<ov::pass::MultiplyGroupConvolutionFusion>();
    conv_fusions->add_matcher<ov::pass::MultiplyConvolutionBackpropDataFusion>();
    conv_fusions->add_matcher<ov::pass::MultiplyGroupConvolutionBackpropDataFusion>();
    conv_fusions->set_name("ov::pass::ConvFusions");

    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<ov::pass::ConvertGather1ToGather7>();
    manager.register_pass<ov::pass::ConvertGather7ToGather8>();
    manager.register_pass<ov::pass::ConvertDeformableConv8To1>();
    manager.register_pass<ov::pass::ConvertMaxPool14ToMaxPool8>();
    manager.register_pass<ov::pass::ConvertMaxPool8ToMaxPool1>();
    manager.register_pass<ov::pass::ConvertAvgPool14ToAvgPool1>();
    manager.register_pass<ov::pass::ConvertSoftMax1ToSoftMax8>();
    manager.register_pass<ov::pass::ConvertDetectionOutput8ToDetectionOutput1>();
    manager.register_pass<ov::pass::ConvertShapeOf3>();

    manager.register_pass<ov::pass::StridesOptimization>();
    manager.register_pass<ov::pass::ConvertSoftMax1ToSoftMax8>();

    manager.run_passes(model);
}

static ov::intel_npu::CompilerType resolveCompilerType(const FilteredConfig base_conf, const ov::AnyMap& local_conf) {
    // first look if provided config changes compiler type
    auto it = local_conf.find(std::string(COMPILER_TYPE::key()));
    if (it != local_conf.end()) {
        // if compiler_type is provided by local config = use that
        return COMPILER_TYPE::parse(it->second.as<std::string>());
    }
    // if there is no compiler_type provided = use base_config value
    return base_conf.get<COMPILER_TYPE>();
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
    REGISTER_OPTION(MODEL_PTR);
    REGISTER_OPTION(BATCH_COMPILER_MODE_SETTINGS);
    REGISTER_OPTION(WEIGHTLESS_BLOB);
    REGISTER_OPTION(SEPARATE_WEIGHTS_VERSION);
    REGISTER_OPTION(WS_COMPILE_CALL_NUMBER);
    REGISTER_OPTION(BENCHMARK_INIT);
    if (_backend) {
        if (_backend->isCommandQueueExtSupported()) {
            REGISTER_OPTION(TURBO);
            REGISTER_OPTION(WORKLOAD_TYPE);
        }
        // register backend options
        _backend->registerOptions(*_options);
    }

    // parse again env_variables to update registered configs which have env vars set
    _globalConfig.parseEnvVars();

    // filter out unsupported options
    filter_config_by_compiler_support(_globalConfig);
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
    if (cfg.hasOpt(ov::intel_npu::turbo.name())) {
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
    for (const auto& [key, value] : rawConfig) {
        if (!localConfig.hasOpt(key)) {
            // not a known config key
            if (!compiler->is_option_supported(key)) {
                OPENVINO_THROW("[ NOT_FOUND ] Option '", key, "' is not supported for current configuration");
            }
        }
    }

    // 3. If all good so far, update values
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

    // create compiler
    CompilerAdapterFactory compilerAdapterFactory;
    auto compiler = compilerAdapterFactory.getCompiler(_backend, resolveCompilerType(_globalConfig, properties));

    const std::map<std::string, std::string> localPropertiesMap = any_copy(localProperties);
    OV_ITT_TASK_CHAIN(PLUGIN_COMPILE_MODEL, itt::domains::NPUPlugin, "Plugin::compile_model", "fork_local_config");
    auto localConfig = fork_local_config(localPropertiesMap, compiler);
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

    auto originalModel = model->clone();

    OV_ITT_TASK_NEXT(PLUGIN_COMPILE_MODEL, "compile");

    std::shared_ptr<intel_npu::IGraph> graph;
    std::vector<std::shared_ptr<intel_npu::IGraph>> initGraphs;
    std::shared_ptr<ov::Model> initModel;

    try {
        _logger.debug("performing compile");

        if (!localConfig.get<WEIGHTLESS_BLOB>()) {
            graph = compiler->compile(model->clone(), localConfig);
        } else {
            initModel = model->clone();

            auto begin = std::chrono::steady_clock::now();
            std::vector<std::shared_ptr<intel_npu::IGraph>> initMainGraphs =
                compiler->compileWS(initModel, localConfig);
            auto end = std::chrono::steady_clock::now();
            std::cout << "compiler->compileWS() call "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]"
                      << std::endl;

            graph = initMainGraphs.back();
            initMainGraphs.pop_back();
            initGraphs = std::move(initMainGraphs);
        }
    } catch (const std::exception& ex) {
        OPENVINO_THROW(ex.what());
    } catch (...) {
        _logger.error("Unexpected exception");
        OPENVINO_THROW("NPU plugin: got an unexpected exception from compiler");
    }

    std::shared_ptr<ov::ICompiledModel> compiledModel;
    try {
        compiledModel = std::make_shared<CompiledModel>(model,
                                                        shared_from_this(),
                                                        device,
                                                        graph,
                                                        localConfig,
                                                        initGraphs,
                                                        initModel);
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

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& origStream, const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "Plugin::import_model");

    ov::AnyMap npu_plugin_properties = properties;
    ov::Tensor tensorBig, tensorSmall;
    bool tensorFromProperty = false;

    std::istream stream{origStream.rdbuf()};
    ov::SharedStreamBuffer buffer{nullptr, 0};  // used only if blob is given by tensor, but it is not OV cached blob

    // ov::hint::compiled_blob has no corresponding "Config" implementation thus we need to remove it from the
    // list of properties
    if (auto blob_it = npu_plugin_properties.find(ov::hint::compiled_blob.name());
        blob_it != npu_plugin_properties.end()) {
        tensorBig = blob_it->second.as<ov::Tensor>();
        tensorFromProperty = true;
        if (auto loadedFromCache = npu_plugin_properties.find(ov::loaded_from_cache.name());
            loadedFromCache != npu_plugin_properties.end() && loadedFromCache->second.as<bool>() != false) {
            tensorBig = ov::Tensor(
                tensorBig,
                ov::Coordinate{static_cast<size_t>(origStream.tellg())},
                ov::Coordinate{tensorBig.get_byte_size()});  // ROI tensor to skip OV header in case of cached blob
        } else {
            buffer = ov::SharedStreamBuffer(reinterpret_cast<char*>(tensorBig.data()), tensorBig.get_byte_size());
            stream.rdbuf(&buffer);
        }
        npu_plugin_properties.erase(blob_it);
    }

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
            return ov::npuw::LLMCompiledModel::import_model(stream, shared_from_this(), properties);
        } else if (compiled_model_indicator == NPUW_COMPILED_MODEL_INDICATOR) {
            // Properties are required for ov::weights_path
            return ov::npuw::CompiledModel::import_model(stream, shared_from_this(), properties);
        } else {
            OPENVINO_THROW("Couldn't deserialize NPUW blob - fatal error!");
        }
    }
    stream.seekg(-stream.tellg() + stream_start_pos, std::ios::cur);

    // Drop NPUW properties if there are any
    for (auto it = properties.begin(); it != properties.end(); ++it) {
        if (it->first.find("NPUW") != it->first.npos) {
            npu_plugin_properties.erase(it->first);
        }
    }

    CompilerAdapterFactory compilerAdapterFactory;
    auto compiler = compilerAdapterFactory.getCompiler(_backend, resolveCompilerType(_globalConfig, properties));

    const auto propertiesMap = any_copy(npu_plugin_properties);
    OV_ITT_TASK_CHAIN(PLUGIN_IMPORT_MODEL, itt::domains::NPUPlugin, "Plugin::import_model", "fork_local_config");
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

    OV_ITT_TASK_NEXT(PLUGIN_IMPORT_MODEL, "parse");

    std::shared_ptr<ov::ICompiledModel> compiledModel;

    try {
        uint64_t mainSize;
        std::vector<uint64_t> initSizes;
        const bool skipCompatibility = localConfig.get<DISABLE_VERSION_CHECK>();
        if (!skipCompatibility) {
            auto storedMeta = read_metadata_from(stream);
            if (!storedMeta->is_compatible()) {
                OPENVINO_THROW("Incompatible blob version!");
            }

            size_t accumulator = 0;
            initSizes = storedMeta->get_init_sizes();
            mainSize = storedMeta->get_blob_size() - std::accumulate(initSizes.begin(), initSizes.end(), accumulator);
        } else {
            _logger.info("Blob compatibility check skipped.");
            mainSize = MetadataBase::getFileSize(stream);
        }

        if (tensorFromProperty == false) {  // tensor was not received from ov::compiled_blob property, copy from stream
            tensorSmall = ov::Tensor(ov::element::u8, ov::Shape{mainSize});
            stream.read(tensorSmall.data<char>(), mainSize);
        } else {
            tensorSmall = ov::Tensor(tensorBig,
                                     ov::Coordinate{0},
                                     ov::Coordinate{mainSize});  // ROI tensor to skip NPU plugin metadata
        }
        auto graph = compiler->parse(std::move(tensorSmall), !tensorFromProperty, localConfig);
        graph->update_network_name("net" + std::to_string(_compiledModelLoadCounter++));

        if (initSizes.empty()) {
            const std::shared_ptr<ov::Model> modelDummy =
                create_dummy_model(graph->get_metadata().inputs, graph->get_metadata().outputs);
            compiledModel = std::make_shared<CompiledModel>(modelDummy, shared_from_this(), device, graph, localConfig);
        } else {
            // Read the init compiled models as well
            // TODO adjust for multiple init parts
            size_t cursorPosition = mainSize;
            std::vector<std::shared_ptr<IGraph>> initGraphs;
            for (uint64_t initSize : initSizes) {
                if (tensorFromProperty == false) {
                    tensorSmall = ov::Tensor(ov::element::u8, ov::Shape{initSize});
                    stream.read(tensorSmall.data<char>(), initSize);
                } else {
                    tensorSmall = ov::Tensor(tensorBig,
                                             ov::Coordinate{cursorPosition},
                                             ov::Coordinate{cursorPosition + initSize});
                }

                std::shared_ptr<IGraph> initGraph =
                    compiler->parse(std::move(tensorSmall), !tensorFromProperty, localConfig);
                initGraph->update_network_name("net" + std::to_string(_compiledModelLoadCounter++));
                initGraphs.push_back(initGraph);
                cursorPosition += initSize;
            }

            // Retrieve the ov::Model used for compilation. This is required for extracting and matching the weights
            std::shared_ptr<ov::Model> originalModel;
            if (localConfig.get<MODEL_PTR>()) {
                originalModel = properties.at(ov::hint::model.name()).as<std::shared_ptr<ov::Model>>();
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

            runOVPasses(originalModel);

            if (!localConfig.get<BENCHMARK_INIT>()) {
                const std::shared_ptr<ov::Model> modelDummy =
                    create_dummy_model(graph->get_metadata().inputs, graph->get_metadata().outputs);
                compiledModel = std::make_shared<CompiledModel>(modelDummy,
                                                                shared_from_this(),
                                                                device,
                                                                graph,
                                                                localConfig,
                                                                initGraphs,
                                                                originalModel);
            } else {
                // TODO: BENCHMARK_INIT must become an integer?
                if (initGraphs.empty()) {
                    OPENVINO_THROW("Can't BENCHMARK_INIT: single init function not found");
                }

                const std::shared_ptr<ov::Model> modelDummy =
                    create_dummy_model(initGraphs.at(0)->get_metadata().inputs,
                                       initGraphs.at(0)->get_metadata().outputs,
                                       true);
                compiledModel = std::make_shared<CompiledModel>(modelDummy,
                                                                shared_from_this(),
                                                                device,
                                                                initGraphs.at(0),
                                                                localConfig);
            }
        }
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
    CompilerAdapterFactory compilerAdapterFactory;
    auto compiler = compilerAdapterFactory.getCompiler(_backend, resolveCompilerType(_globalConfig, properties));
    const std::map<std::string, std::string> propertiesMap = any_copy(properties);
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

std::atomic<int> Plugin::_compiledModelLoadCounter{1};

static const ov::Version version = {CI_BUILD_NUMBER, NPU_PLUGIN_LIB_NAME};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(Plugin, version)

}  // namespace intel_npu
