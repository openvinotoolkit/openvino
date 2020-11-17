// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_metric_helpers.hpp"
#include "mkldnn_plugin.h"
#include "mkldnn_extension_mngr.h"
#include "mkldnn_weights_cache.hpp"
#include "mkldnn_itt.h"

#include <legacy/net_pass.h>
#include <threading/ie_executor_manager.hpp>
#include <memory>
#include <ie_plugin_config.hpp>
#include <vector>
#include <tuple>
#include <ie_system_conf.h>
#include <generic_ie.hpp>
#include <nodes/list.hpp>
#include <legacy/ie_util_internal.hpp>
#include <legacy/graph_transformer.h>
#include <ie_ngraph_utils.hpp>

#include <legacy/convert_function_to_cnn_network.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_prior_to_ie_prior.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/reshape_fully_connected.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_nms_5_to_legacy.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_interpolate_to_interp_or_resample.hpp>
#include <legacy/ngraph_ops/fully_connected.hpp>

#include <transformations/opset_conversions/convert_opset3_to_opset2.hpp>
#include <transformations/opset_conversions/convert_opset2_to_opset1.hpp>

#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/common_optimizations/depth_to_space_fusion.hpp>
#include <transformations/op_conversions/convert_depth_to_space.hpp>
#include <transformations/op_conversions/convert_space_to_depth.hpp>
#include <transformations/op_conversions/convert_gelu.hpp>
#include <transformations/op_conversions/hswish_decomposition.hpp>
#include <transformations/op_conversions/hsigmoid_decomposition.hpp>
#include <transformations/op_conversions/reduce_l1_decomposition.hpp>
#include <transformations/op_conversions/reduce_l2_decomposition.hpp>
#include <transformations/op_conversions/convert_pad_to_group_conv.hpp>
#include <transformations/op_conversions/softplus_decomposition.hpp>
#include <transformations/op_conversions/convert_space_to_batch.hpp>
#include <transformations/op_conversions/convert_batch_to_space.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>
#include <transformations/control_flow/unroll_tensor_iterator.hpp>
#include <transformations/op_conversions/convert_mod.hpp>
#include <transformations/op_conversions/convert_ti_to_sequences.hpp>
#include <transformations/op_conversions/lstm_cell_decomposition.hpp>
#include <transformations/op_conversions/rnn_cell_decomposition.hpp>
#include <transformations/op_conversions/gru_cell_decomposition.hpp>
#include <transformations/op_conversions/log_softmax_decomposition.hpp>
#include <transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp>
#include <transformations/convert_precision.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/rt_info/fused_names_attribute.hpp>

#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/op/util/op_types.hpp>
#include <ngraph/pass/manager.hpp>

#include <transformations/common_optimizations/lin_op_sequence_fusion.hpp>

# include <low_precision/transformer.hpp>
# include <low_precision/convolution.hpp>
# include <low_precision/group_convolution.hpp>
# include <low_precision/multiply_to_group_convolution.hpp>

#if !defined(__arm__) && !defined(_M_ARM) && !defined(__aarch64__) && !defined(_M_ARM64)
#if defined(_WIN32) || defined(WIN32)
#include <intrin.h>
#include <windows.h>
#else
#include <cpuid.h>

#endif
#endif

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

Engine::Engine() {
    _pluginName = "CPU";
    extensionManager->AddExtension(std::make_shared<Extensions::Cpu::MKLDNNExtensions>());
}

Engine::~Engine() {
    ExecutorManager::getInstance()->clear("CPUStreamsExecutor");
    ExecutorManager::getInstance()->clear("CPUCallbackExecutor");
}

static void Transformation(ICNNNetwork::Ptr& clonedNetwork, const Config& conf) {
    auto nGraphFunc = clonedNetwork->getFunction();
    // Disable shape inference (WA for generic operations)
    ngraph::op::GenericIE::DisableReshape noReshape(nGraphFunc);

    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    // WA: ConvertPriorBox must be executed before the 1st ConstantFolding pass
    manager.register_pass<ngraph::pass::ConvertPriorBox>();
    manager.register_pass<ngraph::pass::ConvertNMS5ToLegacyMatcher>();
    manager.register_pass<ngraph::pass::CommonOptimizations>();
    manager.register_pass<ngraph::pass::ConvertRNNSequenceToTensorIterator>();
    manager.register_pass<ngraph::pass::ConvertGRUSequenceToTensorIterator>();
    manager.register_pass<ngraph::pass::ConvertLSTMSequenceToTensorIterator>();
    manager.register_pass<ngraph::pass::ConvertOpSet3ToOpSet2>();
    manager.register_pass<ngraph::pass::ConvertOpSet2ToOpSet1>();
    manager.register_pass<ngraph::pass::ConvertTensorIteratorToGRUSequence>();
    manager.register_pass<ngraph::pass::ConvertTensorIteratorToLSTMSequence>();
    manager.register_pass<ngraph::pass::ConvertTensorIteratorToRNNSequence>();
    manager.register_pass<ngraph::pass::LSTMCellDecomposition>();
    manager.register_pass<ngraph::pass::GRUCellDecomposition>();
    manager.register_pass<ngraph::pass::RNNCellDecomposition>();

    std::vector<std::pair<ngraph::element::Type, ngraph::element::Type>> convert_precision_list{
            {ngraph::element::i64,     ngraph::element::i32},
            {ngraph::element::u64,     ngraph::element::i32},
            {ngraph::element::u16,     ngraph::element::i32},
            {ngraph::element::u32,     ngraph::element::i32},
            {ngraph::element::f16,     ngraph::element::f32},
            {ngraph::element::boolean, ngraph::element::u8},
    };

    for (auto &precision : convert_precision_list) {
        manager.register_pass<ngraph::pass::ConvertPrecision>(precision.first, precision.second);
    }

    auto pass_config = manager.get_pass_config();

    using const_node_ptr = const std::shared_ptr<const ngraph::Node>;

    // SpaceToDepth/ DepthToSpace node implementation supports only equal input/output tensors with rank <= 5
    pass_config->set_callback<ngraph::pass::ConvertSpaceToDepth,
            ngraph::pass::ConvertDepthToSpace>(
            [](const_node_ptr &node) -> bool {
                return node->input_value(0).get_shape().size() <= 5lu &&
                       node->input_value(0).get_shape().size() == node->get_output_shape(0).size();
            });

    // Disable FC reshaping for 3D case
    pass_config->set_callback<ngraph::pass::ReshapeFullyConnected>(
            [](const_node_ptr &node) -> bool {
                return node->input_value(0).get_shape().size() == 3ul;
            });

    pass_config->set_callback<ngraph::pass::ConvertBatchToSpace,
                              ngraph::pass::ConvertSpaceToBatch>(
            [](const_node_ptr &node) -> bool {
                const auto & rank = node->input(0).get_partial_shape().rank().get_length();
                return rank == 4lu || rank == 5lu;
            });

    auto isCellPrimitiveSupported = [](const_node_ptr &node) -> bool {
        if (const auto &rnn_cell = std::dynamic_pointer_cast<const ngraph::opset4::RNNCell>(node)) {
            return rnn_cell->get_clip() == 0.0f;
        } else if (const auto &gru_cell = std::dynamic_pointer_cast<const ngraph::opset4::GRUCell>(
                node)) {
            return gru_cell->get_clip() == 0.0f
                   && gru_cell->get_activations() == std::vector<std::string>{"sigmoid", "tanh"};
        } else if (const auto &lstm_cell = std::dynamic_pointer_cast<const ngraph::opset4::LSTMCell>(
                node)) {
            return lstm_cell->get_clip() == 0.0f &&
                   lstm_cell->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"};
        } else if (const auto &lstm_cell_v1 = std::dynamic_pointer_cast<const ngraph::opset1::LSTMCell>(
                node)) {
            return lstm_cell_v1->get_clip() == 0.0f &&
                   lstm_cell_v1->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"};
        }
        return false;
    };

    pass_config->set_callback<ngraph::pass::RNNCellDecomposition, ngraph::pass::GRUCellDecomposition,
            ngraph::pass::LSTMCellDecomposition>(
            [isCellPrimitiveSupported](const_node_ptr &node) -> bool {
                return isCellPrimitiveSupported(node);
            });

    pass_config->set_callback<ngraph::pass::ConvertTensorIteratorToRNNSequence,
                              ngraph::pass::ConvertTensorIteratorToLSTMSequence,
                              ngraph::pass::ConvertTensorIteratorToGRUSequence>(
            [isCellPrimitiveSupported](const_node_ptr &node) -> bool {
                if (const auto& ti_op = std::dynamic_pointer_cast<const ngraph::op::TensorIterator>(node)) {
                    size_t count_rnn = 0;
                    for (const auto &op : ti_op->get_body()->get_ops())
                        count_rnn += isCellPrimitiveSupported(op);
                    return count_rnn != 1;
                }
                return true;
            });

    // List of enabled/disabled transformations
    pass_config->disable<ngraph::pass::ConvertGELU>();
    pass_config->disable<ngraph::pass::HSwishDecomposition>();
    pass_config->disable<ngraph::pass::ReduceL1Decomposition>();
    pass_config->disable<ngraph::pass::ReduceL2Decomposition>();
    pass_config->disable<ngraph::pass::SoftPlusDecomposition>();
    pass_config->disable<ngraph::pass::HSigmoidDecomposition>();
    pass_config->disable<ngraph::pass::ConvertMod>();
    pass_config->disable<ngraph::pass::LogSoftmaxDecomposition>();
    pass_config->disable<ngraph::pass::ConvertInterpolateToInterpOrResampleMatcher>();

    pass_config->enable<ngraph::pass::ConvertPadToGroupConvolution>();
    pass_config->enable<ngraph::pass::ConvertInterpolate1ToInterpolate4>();

    manager.run_passes(nGraphFunc);

    using namespace ngraph::pass::low_precision;
    if (conf.lpTransformsMode == Config::LPTransformsMode::On) {
        auto params = LayerTransformation::Params(
            true,  // updatePrecisions
            LayerTransformation::QuantizedTensorAlignment::UpdateLevel,  // quantizedTensorAlignmentOnActivations
            LayerTransformation::QuantizedTensorAlignment::None,  // quantizedTensorAlignmentOnWeights
            true);  // supportAsymmetricQuantization
        LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(params)
            .add<ConvolutionTransformation, ngraph::opset1::Convolution>(
                LayerTransformation::Params(params).setPrecisionsOnActivations({ngraph::element::u8}).setSupportAsymmetricQuantization(true))
            .add<GroupConvolutionTransformation, ngraph::opset1::GroupConvolution>(
                LayerTransformation::Params(params).setPrecisionsOnActivations({ ngraph::element::u8 }).setSupportAsymmetricQuantization(true))
            .addStandaloneCleanup<MultiplyToGroupConvolutionTransformation, ngraph::opset1::Multiply>(
                LayerTransformation::Params(params).setPrecisionsOnActivations({ ngraph::element::u8 })));

        transformer.transform(nGraphFunc);
    }

    ngraph::pass::Manager legacyManager;
    legacyManager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();
    legacyManager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::i64, ngraph::element::i32);
    // not legacy actually, but it should be the last transformation in the transformation pipeline
    legacyManager.register_pass<ngraph::pass::UnrollTensorIterator>();

    auto legacyPassConfig = legacyManager.get_pass_config();
    legacyPassConfig->set_callback<ngraph::pass::AddMultiplyFusion>([](const_node_ptr &node) -> bool {
        if (auto mul_op = std::dynamic_pointer_cast<const ngraph::opset1::Multiply>(node)) {
            auto add_op = std::dynamic_pointer_cast<const ngraph::opset1::Add>(mul_op->get_input_node_shared_ptr(0));
            auto constant = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(mul_op->get_input_node_shared_ptr(1));
            bool is_dequantization = mul_op->get_rt_info().count("DEQUANTIZATION") != 0;
            if (add_op && constant && is_dequantization) {
                return ngraph::is_type<ngraph::opset1::Convolution>(add_op->get_input_node_shared_ptr(0)) ||
                    ngraph::is_type<ngraph::opset1::GroupConvolution>(add_op->get_input_node_shared_ptr(0)) ||
                    ngraph::is_type<ngraph::opset1::MatMul>(add_op->get_input_node_shared_ptr(0));
            }
        }
        return false;
    });

    legacyManager.get_pass_config()->set_callback<ngraph::pass::UnrollTensorIterator>([](const_node_ptr &node) -> bool {
        // UnrollTI transformation is disabled by default, is turned on by LowLatency transformation
        return node->get_rt_info().count("UNROLL_TI") == 0;
    });
    legacyManager.run_passes(nGraphFunc);

    OV_ITT_TASK_CHAIN(taskChain, MKLDNNPlugin::itt::domains::MKLDNN_LT, "Transformation", "convertFunctionToICNNNetwork");

    clonedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(nGraphFunc, *clonedNetwork);

    OV_ITT_TASK_NEXT(taskChain, "ConvertIOPrecision");

    // WA: after conversion to CNNNetwork user precision can redefine input/output precisions
    // so we need to apply additional precision conversion but only for inputs and outputs
    for (auto & precision : convert_precision_list) {
        NetPass::ConvertIOPrecision(*clonedNetwork, convertPrecision(precision.first), convertPrecision(precision.second));
    }
}

InferenceEngine::ExecutableNetworkInternal::Ptr
Engine::LoadExeNetworkImpl(const InferenceEngine::CNNNetwork &network, const std::map<std::string, std::string> &config) {
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNNPlugin, "Engine::LoadExeNetworkImpl");

    // verification of supported input
    InferenceEngine::InputsDataMap _networkInputs = network.getInputsInfo();
    for (const auto &ii : _networkInputs) {
        auto input_precision = ii.second->getPrecision();
        if (input_precision != InferenceEngine::Precision::FP32 &&
            input_precision != InferenceEngine::Precision::I32 &&
            input_precision != InferenceEngine::Precision::U16 &&
            input_precision != InferenceEngine::Precision::I16 &&
            input_precision != InferenceEngine::Precision::I8 &&
            input_precision != InferenceEngine::Precision::U8 &&
            input_precision != InferenceEngine::Precision::BF16 &&
            input_precision != InferenceEngine::Precision::BOOL &&
            input_precision != InferenceEngine::Precision::I64 &&
            input_precision != InferenceEngine::Precision::U64) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                               << "Input image format " << input_precision << " is not supported yet...";
        }
    }

    // TODO: handle input precision differently - per input and not one per network...

    // TODO: Clarify the behavior of SetConfig method. Skip eng_config or not?
    Config conf = engConfig;
    conf.readProperties(config);

    if (conf.enableDynamicBatch) {
        conf.batchLimit = static_cast<int>(network.getBatchSize());
    }

    std::shared_ptr<ICNNNetwork> clonedNetwork = cloneNetwork(network);

    bool is_transformed = false;
    if (clonedNetwork->getFunction()) {
        Transformation(clonedNetwork, conf);
        is_transformed = true;
    }
    auto implNetwork = std::dynamic_pointer_cast<details::CNNNetworkImpl>(clonedNetwork);
    if (implNetwork) {
        OV_ITT_SCOPED_TASK(itt::domains::MKLDNN_LT, "CNNNet_based_ConstFolding");
        // valid for CNNNetworkImpl only, while there's no API in ICNNNetwork to change network
        ConstTransformer transformator(implNetwork.get());
        transformator.fullTrim();
        if (!is_transformed) {
            NetPass::ConvertPrecision(*implNetwork, Precision::I64, Precision::I32);
            NetPass::ConvertPrecision(*implNetwork, Precision::U64, Precision::I32);
            NetPass::ConvertPrecision(*implNetwork, Precision::U32, Precision::I32);
            NetPass::ConvertPrecision(*implNetwork, Precision::FP16, Precision::FP32);
            NetPass::ConvertPrecision(*implNetwork, Precision::BOOL, Precision::U8);
            NetPass::ConvertPrecision(*implNetwork, Precision::U16, Precision::I32);
        }
    }

    return std::make_shared<MKLDNNExecNetwork>(*clonedNetwork, conf, extensionManager, weightsSharing);
}

void Engine::SetConfig(const std::map<std::string, std::string> &config) {
    // accumulate config parameters on engine level
    engConfig.readProperties(config);
}

Parameter Engine::GetConfig(const std::string& name, const std::map<std::string, Parameter>& /*options*/) const {
    Parameter result;
    auto option = engConfig._config.find(name);
    if (option != engConfig._config.end()) {
        result = option->second;
    } else {
        THROW_IE_EXCEPTION << "Unsupported config key " << name;
    }
    return result;
}

static bool hasAVX512() {
#if !defined(__arm__) && !defined(_M_ARM) && !defined(__aarch64__) && !defined(_M_ARM64)
    unsigned int regs[4] = {7, 0, 0, 0};
#if defined(_WIN32) || defined(WIN32)
    __cpuid(reinterpret_cast<int*>(regs), regs[0]);
#else
    __cpuid_count(regs[0], regs[1], regs[0], regs[1], regs[2], regs[3]);
#endif
    if (regs[1] & (1U << 16))
        return true;
#endif
    return false;
}

Parameter Engine::GetMetric(const std::string& name, const std::map<std::string, Parameter>& /*options*/) const {
    if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(AVAILABLE_DEVICES));
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(FULL_DEVICE_NAME));
        metrics.push_back(METRIC_KEY(OPTIMIZATION_CAPABILITIES));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        metrics.push_back(METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS));
        metrics.push_back(METRIC_KEY(RANGE_FOR_STREAMS));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        std::string brand_string;
#if !defined(__arm__) && !defined(_M_ARM) && !defined(__aarch64__) && !defined(_M_ARM64)
        unsigned int addr_list[3] = { 0x80000002, 0x80000003, 0x80000004 };
        unsigned int regs[4];
        for (auto addr : addr_list) {
            regs[0] = addr;
#if defined(_WIN32) || defined(WIN32)
            __cpuid(reinterpret_cast<int*>(regs), regs[0]);
#else
            __get_cpuid(regs[0], &regs[0], &regs[1], &regs[2], &regs[3]);
#endif
            char *ch = reinterpret_cast<char*>(&regs[0]);
            for (size_t j = 0; j < sizeof(regs); j++)
                brand_string += ch[j];
        }
#else
        brand_string = "Non Intel Architecture";
#endif
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, brand_string);
    } else if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
        std::vector<std::string> availableDevices = { "" };
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, availableDevices);
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        std::vector<std::string> capabilities;
        if (with_cpu_x86_bfloat16())
            capabilities.push_back(METRIC_VALUE(BF16));
        if (hasAVX512())
            capabilities.push_back(METRIC_VALUE(WINOGRAD));
        capabilities.push_back(METRIC_VALUE(FP32));
        capabilities.push_back(METRIC_VALUE(FP16));
        capabilities.push_back(METRIC_VALUE(INT8));
        capabilities.push_back(METRIC_VALUE(BIN));
        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, capabilities);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys;
        for (auto && opt : engConfig._config)
            configKeys.push_back(opt.first);
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)) {
        std::tuple<unsigned int, unsigned int, unsigned int> range = std::make_tuple(1, 1, 1);
        IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, range);
    } else if (name == METRIC_KEY(RANGE_FOR_STREAMS)) {
        std::tuple<unsigned int, unsigned int> range = std::make_tuple(1, parallel_get_max_threads());
        IE_SET_METRIC_RETURN(RANGE_FOR_STREAMS, range);
    } else {
        THROW_IE_EXCEPTION << "Unsupported metric key " << name;
    }
}

void Engine::AddExtension(InferenceEngine::IExtensionPtr extension) {
    extensionManager->AddExtension(extension);
}

QueryNetworkResult Engine::QueryNetwork(const CNNNetwork& network, const std::map<std::string, std::string>& config) const {
    QueryNetworkResult res;
    MKLDNNWeightsSharing::Ptr fake_w_cache;
    auto function = network.getFunction();
    if (function != nullptr) {
        std::unordered_set<std::string> originalOps;
        for (auto&& node : function->get_ops()) {
            originalOps.emplace(node->get_friendly_name());
        }

        // TODO: Clarify the behavior of SetConfig method. Skip eng_config or not?
        Config conf = engConfig;
        conf.readProperties(config);

        if (conf.enableDynamicBatch) {
            conf.batchLimit = static_cast<int>(network.getBatchSize());
        }

        auto clonedNetwork = cloneNetwork(network);
        Transformation(clonedNetwork, conf);
        std::unordered_set<std::string> supported;
        std::unordered_set<std::string> unsupported;
        for (details::CNNNetworkIterator itLayer{clonedNetwork.get()}; itLayer != details::CNNNetworkIterator(); itLayer++) {
            auto layerIsSupported = [&] {
                std::unique_ptr<MKLDNNNode> ptr;
                try {
                    ptr.reset(MKLDNNNode::factory().create(*itLayer, {mkldnn::engine::kind::cpu, 0}, extensionManager, fake_w_cache));
                } catch (InferenceEngine::details::InferenceEngineException&) {
                     return false;
                }
                return true;
            } ();
            for (auto&& fusedLayerName : ngraph::getFusedNamesVector((*itLayer)->getNode())) {
                if (contains(originalOps, fusedLayerName)) {
                    if (layerIsSupported) {
                        supported.emplace(fusedLayerName);
                    } else {
                        unsupported.emplace(fusedLayerName);
                    }
                }
            }
        }

        for (auto&& node : function->get_ops()) {
            if (!contains(unsupported, node->get_friendly_name())) {
                for (auto&& inputNodeOutput : node->input_values()) {
                    if (ngraph::op::is_constant(inputNodeOutput.get_node())) {
                        supported.emplace(inputNodeOutput.get_node()->get_friendly_name());
                    }
                }
                for (auto&& outputs : node->outputs()) {
                    for (auto&& outputNodeInput : outputs.get_target_inputs()) {
                        if (ngraph::op::is_output(outputNodeInput.get_node())) {
                            supported.emplace(outputNodeInput.get_node()->get_friendly_name());
                        }
                    }
                }
            }
        }

        for (auto&& layerName : supported) {
            if (!contains(unsupported, layerName)) {
                res.supportedLayersMap.emplace(layerName, GetName());
            }
        }
    } else {
        details::CNNNetworkIterator i(network);
        while (i != details::CNNNetworkIterator()) {
            try {
                mkldnn::engine eng(mkldnn::engine(mkldnn::engine::kind::cpu, 0));
                // if we can create and have not thrown exception, then layer is supported
                std::unique_ptr <MKLDNNNode>(MKLDNNNode::factory().create(*i, eng, extensionManager, fake_w_cache));
                res.supportedLayersMap.insert({ (*i)->name, GetName() });
            } catch (InferenceEngine::details::InferenceEngineException&) {
            }
            i++;
        }
    }

    return res;
}

static const Version version = {{2, 1}, CI_BUILD_NUMBER, "MKLDNNPlugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(Engine, version)
