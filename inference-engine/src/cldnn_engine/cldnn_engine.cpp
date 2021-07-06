// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <algorithm>
#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <tuple>
#include <cctype>
#include <memory>

#include "ie_metric_helpers.hpp"
#include "ie_plugin_config.hpp"
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ie_ngraph_utils.hpp>
#include <ie_algorithm.hpp>

#include <transformations/opset_conversions/convert_opset3_to_opset2.hpp>
#include <transformations/opset_conversions/convert_opset2_to_opset1.hpp>

#include <transformations/control_flow/unroll_tensor_iterator.hpp>

#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/common_optimizations/lin_op_sequence_fusion.hpp>
#include <transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp>
#include "transformations/common_optimizations/convert_quantize_dequantize.hpp"
#include "transformations/common_optimizations/softmax_fusion.hpp"
#include <transformations/op_conversions/convert_depth_to_space.hpp>
#include <transformations/op_conversions/convert_space_to_depth.hpp>
#include <transformations/op_conversions/convert_gelu.hpp>
#include <transformations/op_conversions/convert_mod.hpp>
#include <transformations/op_conversions/convert_broadcast3.hpp>
#include <transformations/op_conversions/reduce_l1_decomposition.hpp>
#include <transformations/op_conversions/reduce_l2_decomposition.hpp>
#include <transformations/op_conversions/convert_pad_to_group_conv.hpp>
#include <transformations/op_conversions/softplus_decomposition.hpp>
#include <transformations/op_conversions/convert_space_to_batch.hpp>
#include <transformations/op_conversions/convert_batch_to_space.hpp>
#include <transformations/op_conversions/convert_reduce_to_pooling.hpp>
#include <transformations/op_conversions/convert_shuffle_channels3.hpp>
#include <transformations/op_conversions/hswish_decomposition.hpp>
#include <transformations/op_conversions/hsigmoid_decomposition.hpp>
#include <transformations/op_conversions/log_softmax_decomposition.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>
#include <transformations/op_conversions/convert_subtract.hpp>
#include <transformations/op_conversions/convert_ti_to_sequences.hpp>
#include <transformations/op_conversions/gru_cell_decomposition.hpp>
#include <transformations/op_conversions/lstm_cell_decomposition.hpp>
#include <transformations/op_conversions/rnn_cell_decomposition.hpp>
#include <transformations/op_conversions/mvn6_decomposition.hpp>
#include <transformations/op_conversions/bidirectional_sequences_decomposition.hpp>
#include <transformations/op_conversions/convert_previous_nms_to_nms_5.hpp>
#include <transformations/op_conversions/convert_nms_to_nms_ie_internal.hpp>
#include <transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp>
#include <transformations/op_conversions/convert_gather_0d.hpp>
#include <transformations/op_conversions/simplify_ctc_greedy_decoder_seq_len.hpp>
#include <transformations/convert_precision.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/rt_info/fused_names_attribute.hpp>

#include <transformations/low_precision/disable_convert_constant_folding_on_const_path.hpp>
#include <low_precision/pull_reshape_through_dequantization.hpp>
#include <low_precision/pull_transpose_through_dequantization.hpp>
#include <low_precision/transformer.hpp>
#include <low_precision/convolution_backprop_data.hpp>
#include <low_precision/mat_mul.hpp>
#include <low_precision/strided_slice.hpp>
#include <low_precision/network_helper.hpp>

#include "cldnn_engine.h"
#include "cldnn_executable_network.h"
#include "cldnn_custom_layer.h"
#include "cldnn_itt.h"
#include "gpu/gpu_config.hpp"

#include "cldnn/runtime/device_query.hpp"

#ifdef __linux__
# include <dlfcn.h>
#endif

// Undef DEVICE_TYPE macro which can be defined somewhere in windows headers as DWORD and conflict with our metric
#ifdef DEVICE_TYPE
#undef DEVICE_TYPE
#endif

using namespace InferenceEngine;
using namespace InferenceEngine::gpu;
using namespace InferenceEngine::details;

namespace CLDNNPlugin {

#define FACTORY_DECLARATION(op_version, op_name) \
    void __register ## _ ## op_name ## _ ## op_version();

#define FACTORY_CALL(op_version, op_name) \
    __register ## _ ## op_name ## _ ## op_version();

#define REGISTER_FACTORY(op_version, op_name) FACTORY_DECLARATION(op_version, op_name)
#include "cldnn_primitives_list.hpp"
#undef REGISTER_FACTORY

void clDNNEngine::RegisterPrimitives() {
    #define REGISTER_FACTORY(op_version, op_name) FACTORY_CALL(op_version, op_name)
    #include "cldnn_primitives_list.hpp"
    #undef REGISTER_FACTORY
}

struct clDNNEngine::impl {
    CLDNNPlugin::Config m_config;
};

cldnn::device_info clDNNEngine::GetDeviceInfo(const std::map<std::string, std::string> &config) const {
    auto device_info = device_map.begin()->second->get_info();
    if (config.find(PluginConfigParams::KEY_DEVICE_ID) != config.end()) {
        auto val = config.at(PluginConfigParams::KEY_DEVICE_ID);
        if (device_map.find(val) == device_map.end()) {
            IE_THROW() << "Invalid device ID: " << val;
        }
        device_info = device_map.at(val)->get_info();
    }

    return device_info;
}

template<typename T>
static bool disableReduceDecomposition(const std::shared_ptr<const ngraph::Node> node) {
    if (auto op = std::dynamic_pointer_cast<const T>(node)) {
        bool fp16_batch_not_1 = op->get_element_type() == ngraph::element::f16 && op->input(0).get_shape()[0] != 1;
        return !fp16_batch_not_1;
    }
    return false;
}

InferenceEngine::CNNNetwork clDNNEngine::CloneAndTransformNetwork(const InferenceEngine::CNNNetwork& network,
                                                                  const CLDNNPlugin::Config& config) const {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "clDNNEngine::CloneAndTransformNetwork");
    CNNNetwork clonedNetwork = InferenceEngine::details::cloneNetwork(network);

    if (clonedNetwork.getFunction()) {
        OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "clDNNEngine::TransformNetwork");
        auto nGraphFunc = clonedNetwork.getFunction();

        bool enableInt8;
        {
            ngraph::pass::Manager manager;
            enableInt8 = config.enableInt8 && ngraph::pass::low_precision::LowPrecisionTransformer::isFunctionQuantized(nGraphFunc);
            if (enableInt8) {
                manager.register_pass<ngraph::pass::DisableConvertConstantFoldingOnConstPath>(
                    std::vector<ngraph::element::Type>{ ngraph::element::i8, ngraph::element::u8, ngraph::element::i4, ngraph::element::u4 });
            }

            manager.register_pass<ngraph::pass::InitNodeInfo>();
            manager.register_pass<ngraph::pass::CommonOptimizations>();

            if (!config.enable_loop_unrolling) {
                manager.register_pass<ngraph::pass::BidirectionalLSTMSequenceDecomposition>();
                manager.register_pass<ngraph::pass::BidirectionalGRUSequenceDecomposition>();
                manager.register_pass<ngraph::pass::BidirectionalRNNSequenceDecomposition>();
            }

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

            if (config.enable_loop_unrolling) {
                manager.register_pass<ngraph::pass::BidirectionalLSTMSequenceDecomposition>();
                manager.register_pass<ngraph::pass::BidirectionalGRUSequenceDecomposition>();
                manager.register_pass<ngraph::pass::BidirectionalRNNSequenceDecomposition>();
            }

            manager.register_pass<ngraph::pass::ConvertNMS1ToNMS5>();
            manager.register_pass<ngraph::pass::ConvertNMS3ToNMS5>();
            manager.register_pass<ngraph::pass::ConvertNMS4ToNMS5>();
            manager.register_pass<ngraph::pass::ConvertNMSToNMSIEInternal>();
            manager.register_pass<ngraph::pass::ConvertGather0D>();

            static const precisions_array convert_precision_list {
                    {ngraph::element::i64, ngraph::element::i32},
                    {ngraph::element::u64, ngraph::element::i32},
                    {ngraph::element::u16, ngraph::element::i32},
                    {ngraph::element::u32, ngraph::element::i32},
                    {ngraph::element::boolean, ngraph::element::u8},
                    {ngraph::element::i4, ngraph::element::i8},
                    {ngraph::element::u4, ngraph::element::u8},
            };

            manager.register_pass<ngraph::pass::ConvertPrecision>(convert_precision_list);

            auto pass_config = manager.get_pass_config();

            using const_node_ptr = const std::shared_ptr<const ngraph::Node>;

            // SpaceToDepth/DepthToSpace node implementation supports only equal input/output tensors with rank <= 5
            pass_config->set_callback<ngraph::pass::ConvertSpaceToDepth,
                                      ngraph::pass::ConvertDepthToSpace>(
                    [](const_node_ptr &node) -> bool {
                        return node->input_value(0).get_shape().size() <= 5lu &&
                            node->input_value(0).get_shape().size() == node->get_output_shape(0).size();
                    });

            pass_config->set_callback<ngraph::pass::ConvertBatchToSpace,
                                      ngraph::pass::ConvertSpaceToBatch>(
                    [](const_node_ptr &node) -> bool {
                        const auto & rank = node->input(0).get_partial_shape().rank().get_length();
                        return rank <= 5lu;
                    });

            pass_config->set_callback<ngraph::pass::ConvertReduceSumToPooling>(
                [](const_node_ptr &node) -> bool {
                    return disableReduceDecomposition<ngraph::opset1::ReduceSum>(node);
                });

            pass_config->set_callback<ngraph::pass::ConvertReduceMeanToPooling>(
                [](const_node_ptr &node) -> bool {
                    return disableReduceDecomposition<ngraph::opset1::ReduceMean>(node);
                });

            pass_config->set_callback<ngraph::pass::ConvertReduceMaxToPooling>(
                [](const_node_ptr &node) -> bool {
                    return disableReduceDecomposition<ngraph::opset1::ReduceMax>(node);
                });

            auto isCellPrimitiveSupported = [](const_node_ptr &node) -> bool {
                if (std::dynamic_pointer_cast<const ngraph::opset6::RNNCell>(node)) {
                    return false;
                } else if (std::dynamic_pointer_cast<const ngraph::opset6::GRUCell>(node)) {
                    return false;
                } else if (const auto &lstm_cell = std::dynamic_pointer_cast<const ngraph::opset6::LSTMCell>(node)) {
                    return lstm_cell->get_clip() == 0.0f && lstm_cell->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"};
                } else if (const auto &lstm_cell_v1 = std::dynamic_pointer_cast<const ngraph::opset1::LSTMCell>(node)) {
                    return lstm_cell_v1->get_clip() == 0.0f && lstm_cell_v1->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"};
                }
                return false;
            };

            // Sequences supported by the plugin shouldn't be converted to TensorIterator.
            // sequence_length input is not supported in all Sequences, so if is_seq_len_provided() == true, we
            // should always convert to TensorIterator.
            // RNN/GRU Sequences are not supported in GPU plugin
            // LSTM Sequence supported with clip == 0, and activations have default values (sigmoid, tanh, tanh)
            auto isSequencePrimitiveSupported = [](const_node_ptr &node) -> bool {
                const auto& data = node->input(0);
                const auto& data_pshape = data.get_partial_shape();
                if (data_pshape.rank().is_static() && data_pshape.rank().get_length() > 1 && !data_pshape[1].is_static())
                    return false;
                auto max_seq_len = data.get_shape().at(1);
                if (std::dynamic_pointer_cast<const ngraph::opset6::RNNSequence>(node)) {
                    return false;
                } else if (std::dynamic_pointer_cast<const ngraph::opset6::GRUSequence>(node)) {
                    return false;
                } else if (const auto &lstm_seq = std::dynamic_pointer_cast<const ngraph::opset6::LSTMSequence>(node)) {
                    return lstm_seq->get_clip() == 0.0f &&
                           lstm_seq->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"} &&
                           !ngraph::op::util::is_seq_len_provided(lstm_seq->get_input_node_shared_ptr(3),
                                                                  max_seq_len);
                }
                return false;
            };

            pass_config->set_callback<ngraph::pass::RNNCellDecomposition,
                                      ngraph::pass::GRUCellDecomposition,
                                      ngraph::pass::LSTMCellDecomposition>(
                [isCellPrimitiveSupported](const_node_ptr &node) -> bool {
                    return isCellPrimitiveSupported(node);
                });

            pass_config->set_callback<ngraph::pass::ConvertRNNSequenceToTensorIterator,
                                      ngraph::pass::ConvertGRUSequenceToTensorIterator,
                                      ngraph::pass::ConvertLSTMSequenceToTensorIterator>(
                    [isSequencePrimitiveSupported](const_node_ptr &node) -> bool {
                        return isSequencePrimitiveSupported(node);
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

            pass_config->set_callback<ngraph::pass::ConvertNMS1ToNMS5,
                                      ngraph::pass::ConvertNMS3ToNMS5,
                                      ngraph::pass::ConvertNMS4ToNMS5,
                                      ngraph::pass::ConvertNMSToNMSIEInternal>(
                    [](const_node_ptr &node) -> bool {
                        return node->input_value(0).get_shape().back() == 4lu &&
                               node->input_value(0).get_shape().front() == node->input_value(1).get_shape().front() &&
                               node->input_value(0).get_shape()[1] == node->input_value(1).get_shape().back() &&
                               node->input_value(0).get_shape().size() == 3lu &&
                               node->input_value(1).get_shape().size() == 3lu;
                    });

            pass_config->set_callback<ngraph::pass::MVN6Decomposition>(
                [](const_node_ptr &node) -> bool {
                    const auto mvn = std::dynamic_pointer_cast<const ngraph::op::v6::MVN>(node);
                    if (mvn != nullptr && node->get_input_size() == 2) {
                        if (auto axesNode = dynamic_cast<ngraph::op::v0::Constant*>(mvn->get_input_node_ptr(1))) {
                            auto axesVal = axesNode->cast_vector<int>();
                            auto& mvnShape = mvn->get_output_shape(0);
                            for (int32_t& axis : axesVal)
                                axis = axis < 0 ? axis + mvnShape.size() : axis;
                            std::sort(axesVal.begin(), axesVal.end());
                            if (mvnShape.size() == 1)
                                return false;
                            if (mvnShape.size() > 5 || (mvnShape.size() != axesVal.size() + 1 && mvnShape.size() != axesVal.size() + 2))
                                return false;
                            int value = mvnShape.size() - 1;
                            for (int i = axesVal.size() - 1; i >= 0; i--, value--) {
                                if (axesVal[i] != value)
                                    return false;
                            }
                            return true;
                        }
                    }
                    return false;
                });

            pass_config->set_callback<ngraph::pass::SoftmaxFusion>(
                [](const_node_ptr &node) -> bool {
                    return node->input_value(0).get_partial_shape().rank().get_length() > 5;
                });

            // List of enabled/disabled transformations
            pass_config->disable<ngraph::pass::ConvertGELU>();
            pass_config->disable<ngraph::pass::ConvertMod>();
            pass_config->disable<ngraph::pass::ConvertShuffleChannels3>();
            pass_config->disable<ngraph::pass::HSwishDecomposition>();
            pass_config->disable<ngraph::pass::HSigmoidDecomposition>();
            pass_config->disable<ngraph::pass::ReduceL1Decomposition>();
            pass_config->disable<ngraph::pass::ReduceL2Decomposition>();
            pass_config->disable<ngraph::pass::SoftPlusDecomposition>();
            pass_config->disable<ngraph::pass::LogSoftmaxDecomposition>();
            pass_config->disable<ngraph::pass::ConvertBroadcast3>();
            pass_config->disable<ngraph::pass::WeightsDequantizeToFakeQuantize>();
            pass_config->disable<ngraph::pass::SimplifyCTCGreedyDecoderSeqLen>();

            if (!config.enable_loop_unrolling) {
                pass_config->disable<ngraph::pass::ConvertTensorIteratorToRNNSequence>();
                pass_config->disable<ngraph::pass::ConvertTensorIteratorToLSTMSequence>();
                pass_config->disable<ngraph::pass::ConvertTensorIteratorToGRUSequence>();
            }

            pass_config->enable<ngraph::pass::ConvertInterpolate1ToInterpolate4>();

            if (enableInt8) {
                pass_config->set_callback<ngraph::pass::ConvertQuantizeDequantize>([](const_node_ptr &node) -> bool {
                    return ngraph::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForMultiply(node);
                });

                pass_config->set_callback<ngraph::pass::ConvertSubtract>([](const_node_ptr &node) -> bool {
                    return ngraph::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForSubtract(node);
                });
            }

            manager.run_passes(nGraphFunc);
        }

        if (enableInt8) {
            OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "clDNNEngine::TransformNetwork::LPT");
            using namespace ngraph::pass::low_precision;

            ngraph::pass::Manager manager;
            // Conversion to FP32 might be needed for quantized models that face any fp16 related issues (e.g. overflow) for non-quantized layers
            // With this key users can work-around such issues
            if (!config.enable_fp16_for_quantized_models) {
                manager.register_pass<ngraph::pass::ConvertPrecision>(precisions_array {{ ngraph::element::f16, ngraph::element::f32 }});
            }
            auto lptPrerequisites = manager.register_pass<ngraph::pass::GraphRewrite>();
            const std::vector<ngraph::element::Type> supportedTypes = { ngraph::element::i8, ngraph::element::u8 };
            lptPrerequisites->add_matcher<PullReshapeThroughDequantization>(supportedTypes);
            lptPrerequisites->add_matcher<PullTransposeThroughDequantization>(supportedTypes);
            lptPrerequisites->add_matcher<ngraph::pass::LinOpSequenceFusion>();
            manager.run_passes(nGraphFunc);

            auto params = LayerTransformation::Params(true,                                                        // updatePrecisions
                                                      LayerTransformation::QuantizedTensorAlignment::UpdateLevel,  // quantizedTensorAlignmentOnActivations
                                                      LayerTransformation::QuantizedTensorAlignment::None,         // quantizedTensorAlignmentOnWeights
                                                      true);                                                       // supportAsymmetricQuantization
            LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(params)
                .add<MatMulTransformation, ngraph::opset1::MatMul>(LayerTransformation::Params(params)
                    .setSupportAsymmetricQuantization(false)
                    .setSupport3DTensorOnActivations(false))
                .add<ConvolutionBackpropDataTransformation, ngraph::opset1::ConvolutionBackpropData>(LayerTransformation::Params(params)
                    .setSupportAsymmetricQuantization(false)
                    .setDeconvolutionSpecificChannelsRatio(true))
                // INT8 StridedSlice not supported
                .remove<StridedSliceTransformation, ngraph::opset1::StridedSlice>());

            transformer.transform(nGraphFunc);
        }

        {
            OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "clDNNEngine::TransformNetwork::RunPasses");
            ngraph::pass::Manager manager;
            // This ConstantFolding pass is added to fold reshapes added for constant inputs on NMS internal operation which prevents upper-bound calculation
            // TODO: check why we have these reshapes
            manager.register_pass<ngraph::pass::ConstantFolding>();

            manager.register_pass<ngraph::pass::UnrollTensorIterator>();
            auto pass_config = manager.get_pass_config();
            pass_config->set_callback<ngraph::pass::UnrollTensorIterator>(
                [config](const std::shared_ptr<const ngraph::Node> &node) -> bool {
                    auto sub_graph_op = std::dynamic_pointer_cast<const ngraph::op::util::SubGraphOp>(node);
                    int64_t num_iter = sub_graph_op->get_num_iterations();
                    if (num_iter == 1) {
                        return false;
                    }
                    return !config.enable_loop_unrolling;
                });

            manager.run_passes(nGraphFunc);
        }
    }
    return clonedNetwork;
}

clDNNEngine::clDNNEngine() : m_defaultContext(nullptr) {
    _pluginName = "GPU";
    _impl = std::make_shared<impl>();
    RegisterPrimitives();
    // try loading clDNN engine and get info from it
    {
        // Set OCL runtime which should be always available
        cldnn::device_query device_query(cldnn::engine_types::ocl, cldnn::runtime_types::ocl);
        device_map = device_query.get_available_devices();
    }
    // locate global custom kernel config
    // and auto-load kernels from it
#ifdef _WIN32
    CHAR mpath[MAX_PATH + 1];
    HMODULE nModule;
    GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCSTR)CLDNNCustomLayer::LoadFromFile,
        &nModule);
    GetModuleFileName(nModule, mpath, sizeof(mpath));
#elif __linux__
    Dl_info dl_info;
    dladdr(reinterpret_cast<void *>(CLDNNCustomLayer::LoadFromFile), &dl_info);
    const char* mpath = dl_info.dli_fname;
#endif
    std::string configFile(mpath);
    std::size_t dir_split_pos = configFile.find_last_of("/\\");
    std::string config_path;

    if (dir_split_pos != std::string::npos) {
        // path contains directory
        config_path = configFile.substr(0, dir_split_pos);
    }
    config_path += "/cldnn_global_custom_kernels/cldnn_global_custom_kernels.xml";
    CLDNNCustomLayer::LoadFromFile(config_path, _impl->m_config.customLayers, true);
}

auto check_inputs = [](InferenceEngine::InputsDataMap _networkInputs) {
    for (auto ii : _networkInputs) {
        auto input_precision = ii.second->getTensorDesc().getPrecision();
        if (input_precision != InferenceEngine::Precision::FP16 &&
            input_precision != InferenceEngine::Precision::FP32 &&
            input_precision != InferenceEngine::Precision::U8 &&
            input_precision != InferenceEngine::Precision::I8 &&
            input_precision != InferenceEngine::Precision::I16 &&
            input_precision != InferenceEngine::Precision::U16 &&
            input_precision != InferenceEngine::Precision::I32 &&
            input_precision != InferenceEngine::Precision::I64 &&
            input_precision != InferenceEngine::Precision::BOOL) {
            IE_THROW(NotImplemented)
                << "Input image format " << input_precision << " is not supported yet...";
        }
    }
};

void clDNNEngine::UpdateConfig(CLDNNPlugin::Config& conf, const InferenceEngine::CNNNetwork &network, const std::map<std::string, std::string> &params) const {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "clDNNEngine::UpdateConfig");
    auto device_info = GetDeviceInfo(params);
    conf.enableInt8 = device_info.supports_imad || device_info.supports_immad;
    conf.UpdateFromMap(params);
    if (conf.enableDynamicBatch) {
        conf.max_dynamic_batch = static_cast<int>(network.getBatchSize());
    }
}

IExecutableNetworkInternal::Ptr clDNNEngine::LoadExeNetworkImpl(const InferenceEngine::CNNNetwork &network,
                                                                const std::map<std::string, std::string> &config) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "clDNNEngine::LoadExeNetworkImpl");
    // verification of supported input
    InferenceEngine::InputsDataMap _networkInputs = network.getInputsInfo();
    check_inputs(_networkInputs);

    CLDNNPlugin::Config conf = _impl->m_config;
    UpdateConfig(conf, network, config);

    CLDNNRemoteCLContext::Ptr context;

    auto canReuseDefaultContext = [&]() -> bool {
        if (m_defaultContext == nullptr)
            return false;

        const Config& context_config = m_defaultContext->GetConfig();
        const Config& current_config = conf;

        return context_config.throughput_streams == current_config.throughput_streams &&
               context_config.useProfiling == current_config.useProfiling &&
               context_config.dumpCustomKernels == current_config.dumpCustomKernels &&
               context_config.memory_pool_on == current_config.memory_pool_on &&
               context_config.queueThrottle == current_config.queueThrottle &&
               context_config.queuePriority == current_config.queuePriority &&
               context_config.sources_dumps_dir == current_config.sources_dumps_dir &&
               context_config.tuningConfig.mode == current_config.tuningConfig.mode &&
               context_config.tuningConfig.cache_file_path == current_config.tuningConfig.cache_file_path &&
               context_config.kernels_cache_dir == current_config.kernels_cache_dir &&
               context_config.device_id == current_config.device_id &&
               context_config.n_threads == current_config.n_threads &&
               context_config.enable_loop_unrolling == current_config.enable_loop_unrolling;
    };

    {
        OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "clDNNEngine::LoadExeNetworkImpl::CreateContext");
        std::lock_guard<std::mutex> lock(engine_mutex);
        if (!canReuseDefaultContext()) {
            m_defaultContext.reset(new CLDNNRemoteCLContext(shared_from_this(), ParamMap(), conf));
        }
    }

    context = m_defaultContext;

    auto transformedNetwork = CloneAndTransformNetwork(network, conf);
    {
        OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "clDNNEngine::LoadExeNetworkImpl::CreateExeNetwork");
        return std::make_shared<CLDNNExecNetwork>(transformedNetwork, context, conf);
    }
}

IExecutableNetworkInternal::Ptr clDNNEngine::LoadExeNetworkImpl(const InferenceEngine::CNNNetwork &network,
                                                                const RemoteContext::Ptr &context,
                                                                const std::map<std::string, std::string> &config) {
    InferenceEngine::InputsDataMap _networkInputs = network.getInputsInfo();
    check_inputs(_networkInputs);

    auto casted = std::dynamic_pointer_cast<ClContext>(context);
    if (nullptr == casted) {
        IE_THROW() << "Invalid context";
    }

    CLDNNPlugin::Config conf = getContextImpl(casted)->GetConfig();
    UpdateConfig(conf, network, config);

    auto transformedNetwork = CloneAndTransformNetwork(network, conf);
    return std::make_shared<CLDNNExecNetwork>(transformedNetwork, casted, conf);
}

RemoteContext::Ptr clDNNEngine::CreateContext(const ParamMap& params) {
    // parameter map is non-empty
    std::string contextTypeStr = _StrFromParams(params, GPU_PARAM_KEY(CONTEXT_TYPE));

    if (GPU_PARAM_VALUE(OCL) == contextTypeStr) {
        auto context = std::make_shared<CLDNNRemoteCLContext>(shared_from_this(), params, _impl->m_config);
        return std::dynamic_pointer_cast<RemoteContext>(context);
    } else if (GPU_PARAM_VALUE(VA_SHARED) == contextTypeStr) {
#ifdef _WIN32
        auto context = std::make_shared<CLDNNRemoteD3DContext>(shared_from_this(), params, _impl->m_config);
#else
        auto context = std::make_shared<CLDNNRemoteVAContext>(shared_from_this(), params, _impl->m_config);
#endif
        return std::dynamic_pointer_cast<RemoteContext>(context);
    } else {
        IE_THROW() << "Invalid remote context type" << contextTypeStr;
    }
}

RemoteContext::Ptr clDNNEngine::GetDefaultContext(const ParamMap& params) {
    if (nullptr == m_defaultContext) {
        m_defaultContext.reset(new CLDNNRemoteCLContext(shared_from_this(), params, _impl->m_config));
    }
    return std::dynamic_pointer_cast<RemoteContext>(m_defaultContext);
}

void clDNNEngine::SetConfig(const std::map<std::string, std::string> &config) {
    _impl->m_config.UpdateFromMap(config);
}

QueryNetworkResult clDNNEngine::QueryNetwork(const CNNNetwork& network,
                                             const std::map<std::string, std::string>& config) const {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "clDNNEngine::QueryNetwork");
    QueryNetworkResult res;
    CLDNNPlugin::Config conf = _impl->m_config;
    UpdateConfig(conf, network, config);

    if (m_defaultContext == nullptr) {
        m_defaultContext.reset(new CLDNNRemoteCLContext(
            std::const_pointer_cast<InferenceEngine::IInferencePlugin>(shared_from_this()),
            ParamMap(), conf));
    }
    Program prog(m_defaultContext->getImpl()->GetEngine(), conf);
    auto function = network.getFunction();
    if (function == nullptr) {
        IE_THROW() << "CNNetworkImpl representation is not supported anymore";
    }

    std::unordered_set<std::string> originalOpNames;
    auto originalOps = function->get_ops();
    for (auto&& node : originalOps) {
        originalOpNames.emplace(node->get_friendly_name());
    }

    auto clonedNetwork = CloneAndTransformNetwork(network, conf);
    auto ops = clonedNetwork.getFunction()->get_ordered_ops();
    std::unordered_set<std::string> supported;
    std::unordered_set<std::string> unsupported;

    std::unordered_set<std::string> splitNames;
    std::unordered_set<std::string> concatNames;
    std::unordered_set<std::string> constantsNames;
    std::unordered_set<std::string> depLayerNames;

    std::vector<std::shared_ptr<ngraph::Node>> splits;
    std::vector<std::shared_ptr<ngraph::Node>> concats;
    std::vector<std::shared_ptr<ngraph::Node>> constants;
    std::vector<std::shared_ptr<ngraph::Node>> nextLayerDependent;

    auto layerIsSupported = [&](std::shared_ptr<ngraph::Node> node) {
        if (ngraph::is_type<const ngraph::op::v0::DetectionOutput>(node) ||
            ngraph::is_type<const ngraph::op::v0::PriorBox>(node) ||
            ngraph::is_type<const ngraph::op::v0::PriorBoxClustered>(node) ||
            ngraph::is_type<const ngraph::op::v0::Proposal>(node)) {
            return false;
        } else if (ngraph::is_type<const ngraph::op::v1::Split>(node)) {
            splitNames.emplace(node->get_friendly_name());
            splits.push_back(node);
            return false;
        } else if (ngraph::is_type<const ngraph::op::v0::Concat>(node)) {
            concatNames.emplace(node->get_friendly_name());
            concats.push_back(node);
            return false;
        } else if (ngraph::is_type<const ngraph::op::v1::Reshape>(node) ||
                   ngraph::is_type<const ngraph::op::v0::Squeeze>(node) ||
                   ngraph::is_type<const ngraph::op::v0::Unsqueeze>(node) ||
                   ngraph::is_type<const ngraph::op::v1::Transpose>(node)) {
            depLayerNames.emplace(node->get_friendly_name());
            nextLayerDependent.push_back(node);
            return false;
        } else if (ngraph::is_type<const ngraph::op::v0::Constant>(node)) {
            constantsNames.emplace(node->get_friendly_name());
            constants.push_back(node);
            return false;
        } else if (prog.IsOpSupported(network, node) &&
                   !ngraph::op::is_parameter(node) &&
                   !ngraph::op::is_output(node)) {
            return true;
        } else {
            return false;
        }
    };

    // Get ops after transformations and check if it's supported
    // Transformations might lead to the situation when single node is merged to multiple operations,
    // so we mark original op as supported only if all nodes that it was merged into are supported
    for (auto&& op : ops) {
        for (auto&& fusedLayerName : ngraph::getFusedNamesVector(op)) {
            if (InferenceEngine::details::contains(originalOpNames, fusedLayerName)) {
                if (layerIsSupported(op)) {
                    supported.emplace(fusedLayerName);
                } else {
                    unsupported.emplace(fusedLayerName);
                }
            }
        }
    }

    for (auto&& layerName : supported) {
        if (InferenceEngine::details::contains(unsupported, layerName)) {
            supported.erase(layerName);
        }
    }
    unsupported.clear();

    // Check set of heuristics to produce more efficient hetero sub-graph. Note: checks order is important.
    // 1. Split is marked as supported when all output ops can be offloaded to GPU
    for (const auto & op : splits) {
        bool is_supported = true;
        for (size_t i = 0; i < op->get_output_size(); i++) {
            auto outTensors = op->get_output_target_inputs(i);
            for (auto& t : outTensors) {
                auto output = t.get_node();
                const auto& name = output->get_friendly_name();
                if (!InferenceEngine::details::contains(supported, name) &&
                    !InferenceEngine::details::contains(depLayerNames, name) &&
                    !InferenceEngine::details::contains(concatNames, name) &&
                    !InferenceEngine::details::contains(splitNames, name)) {
                    is_supported = false;
                    break;
                }
            }
        }
        if (is_supported) {
            supported.emplace(op->get_friendly_name());
        }
    }

    // 2. Concat is marked as supported when all inputs can be offloaded to GPU
    for (const auto& op : concats) {
        bool is_supported = true;
        for (size_t i = 0; i < op->get_input_size(); i++) {
            auto input = op->get_input_node_shared_ptr(i);
            const auto& name = input->get_friendly_name();
            if (!InferenceEngine::details::contains(supported, name) &&
                !InferenceEngine::details::contains(depLayerNames, name) &&
                !InferenceEngine::details::contains(concatNames, name)) {
                is_supported = false;
                break;
            }
        }
        if (is_supported) {
            supported.emplace(op->get_friendly_name());
        }
    }

    // 3. Some layers are marked as supported when all inputs and outputs can be offloaded to GPU
    for (const auto& op : nextLayerDependent) {
        bool is_supported = true;
        // both inputs and output should be GPU to remain on GPU
        for (size_t i = 0; i < op->get_input_size(); i++) {
            auto input = op->get_input_node_shared_ptr(i);
            const auto& name = input->get_friendly_name();
            // All inputs must be supported or be a constant
            if (!InferenceEngine::details::contains(supported, name) && !InferenceEngine::details::contains(constantsNames, name)) {
                is_supported = false;
                break;
            }
        }
        for (size_t i = 0; i < op->get_output_size(); i++) {
            auto outTensors = op->get_output_target_inputs(i);
            for (auto& t : outTensors) {
                auto output = t.get_node();
                const auto& name = output->get_friendly_name();
                if (!InferenceEngine::details::contains(supported, name)) {
                    is_supported = false;
                    break;
                }
            }
        }
        if (is_supported) {
            supported.emplace(op->get_friendly_name());
        }
    }

    // 4. Constants are marked as supported when all outputs can be offloaded to GPU
    for (const auto& op : constants) {
        bool is_supported = true;
        for (size_t i = 0; i < op->get_output_size(); i++) {
            auto outTensors = op->get_output_target_inputs(i);
            for (auto& t : outTensors) {
                auto output = t.get_node();
                const auto& name = output->get_friendly_name();
                if (!InferenceEngine::details::contains(supported, name)) {
                    is_supported = false;
                    break;
                }
            }
        }
        if (is_supported) {
            supported.emplace(op->get_friendly_name());
        }
    }

    // Mark original constants/parameters/results ops as supported for each supported operation
    // since rt_info doesn't contain names of constant that are removed during constant folding
    for (auto&& node : originalOps) {
        if (InferenceEngine::details::contains(supported, node->get_friendly_name())) {
            for (auto&& inputNodeOutput : node->input_values()) {
                if (ngraph::op::is_constant(inputNodeOutput.get_node()) || ngraph::op::is_parameter(inputNodeOutput.get_node())) {
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

        if (ngraph::op::is_constant(node) || ngraph::op::is_parameter(node)) {
                if (!InferenceEngine::details::contains(supported, node->output(0).get_target_inputs().begin()->get_node()->get_friendly_name())) {
                    supported.erase(node->get_friendly_name());
                }
            } else if (ngraph::op::is_output(node)) {
                if (!InferenceEngine::details::contains(supported, node->input_values().begin()->get_node()->get_friendly_name())) {
                    supported.erase(node->get_friendly_name());
                }
            }
    }

    for (auto&& layerName : supported) {
        res.supportedLayersMap.emplace(layerName, GetName());
    }

    return res;
}

Parameter clDNNEngine::GetConfig(const std::string& name, const std::map<std::string, Parameter>& /*options*/) const {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "clDNNEngine::GetConfig");
    Parameter result;
    auto option = _impl->m_config.key_config_map.find(name);
    if (option != _impl->m_config.key_config_map.end()) {
        result = option->second;
    } else {
        IE_THROW() << "Unsupported config key : " << name;
    }
    return result;
}

auto StringRightTrim = [](std::string string, std::string substring, bool case_sensitive = true) {
    auto ret_str = string;
    if (!case_sensitive) {
        std::transform(string.begin(), string.end(), string.begin(), ::tolower);
        std::transform(substring.begin(), substring.end(), substring.begin(), ::tolower);
    }
    auto erase_position = string.rfind(substring);
    if (erase_position != std::string::npos) {
        // if space exists before substring remove it also
        if (std::isspace(string.at(erase_position - 1))) {
            erase_position--;
        }
        return ret_str.substr(0, erase_position);
    }
    return ret_str;
};

static float GetGOPS(cldnn::device_info info, cldnn::data_types dt) {
    auto freqGHz = info.gpu_frequency / 1000.f;
    auto numEUs = info.execution_units_count;
    auto opsPerComputeBlock = 0;
    auto computeBlockIPC = 1.0f;
    switch (dt) {
    case cldnn::data_types::u8:
    case cldnn::data_types::i8: {
        if (info.supports_imad) {
            // fma * simd size
            opsPerComputeBlock = 2 * 32;
        } else {
            // separate mul + add instructions for int8 data type
            opsPerComputeBlock = 2 * 16;
            // mul/add instructions can't be executed in parallel, so we need 2 clocks to execute compute block
            computeBlockIPC = 0.5f;
        }
        break;
    }
    case cldnn::data_types::f16: {
        // fma * simd size
        opsPerComputeBlock = 2 * 16;
        break;
    }
    case cldnn::data_types::f32: {
        // fma * simd size
        opsPerComputeBlock = 2 * 8;
        break;
    }

    default: throw std::runtime_error("GetGOPS: Unsupported precision");
    }

    return freqGHz * opsPerComputeBlock * computeBlockIPC * numEUs;
}

Parameter clDNNEngine::GetMetric(const std::string& name, const std::map<std::string, Parameter>& options) const {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "clDNNEngine::GetMetric");
    auto device_id = GetConfig(CONFIG_KEY(DEVICE_ID), {});
    if (options.find(CONFIG_KEY(DEVICE_ID)) != options.end())
        device_id = options.at(CONFIG_KEY(DEVICE_ID)).as<std::string>();

    auto iter = device_map.find(device_id);
    auto device_info = iter != device_map.end() ?
        iter->second->get_info() :
        device_map.begin()->second->get_info();

    if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(AVAILABLE_DEVICES));
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(FULL_DEVICE_NAME));
        metrics.push_back(METRIC_KEY(OPTIMIZATION_CAPABILITIES));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        metrics.push_back(METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS));
        metrics.push_back(METRIC_KEY(RANGE_FOR_STREAMS));
        metrics.push_back(METRIC_KEY(DEVICE_TYPE));
        metrics.push_back(METRIC_KEY(DEVICE_GOPS));
        metrics.push_back(GPU_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE));
        metrics.push_back(GPU_METRIC_KEY(UARCH_VERSION));
        metrics.push_back(GPU_METRIC_KEY(EXECUTION_UNITS_COUNT));

        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
        std::vector<std::string> availableDevices = { };
        for (auto const& dev : device_map)
            availableDevices.push_back(dev.first);
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, availableDevices);
    } else if (name == GPU_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE)) {
        IE_SET_METRIC_RETURN(GPU_DEVICE_TOTAL_MEM_SIZE, device_info.max_global_mem_size);
    } else if (name == METRIC_KEY(DEVICE_TYPE)) {
        auto dev_type = device_info.dev_type == cldnn::device_type::discrete_gpu ? Metrics::DeviceType::discrete : Metrics::DeviceType::integrated;
        IE_SET_METRIC_RETURN(DEVICE_TYPE, dev_type);
    } else if (name == METRIC_KEY(DEVICE_GOPS)) {
        std::map<InferenceEngine::Precision, float> gops;
        gops[InferenceEngine::Precision::I8] = GetGOPS(device_info, cldnn::data_types::i8);
        gops[InferenceEngine::Precision::U8] = GetGOPS(device_info, cldnn::data_types::u8);
        gops[InferenceEngine::Precision::FP16] = GetGOPS(device_info, cldnn::data_types::f16);
        gops[InferenceEngine::Precision::FP32] = GetGOPS(device_info, cldnn::data_types::f32);
        IE_SET_METRIC_RETURN(DEVICE_GOPS, gops);
    } else if (name == GPU_METRIC_KEY(EXECUTION_UNITS_COUNT)) {
        IE_SET_METRIC_RETURN(GPU_EXECUTION_UNITS_COUNT, device_info.execution_units_count);
    } else if (name == GPU_METRIC_KEY(UARCH_VERSION)) {
        std::stringstream s;
        if (device_info.gfx_ver.major == 0 && device_info.gfx_ver.minor == 0 && device_info.gfx_ver.revision == 0) {
            s << "unknown";
        } else {
            s << static_cast<int>(device_info.gfx_ver.major) << "."
              << static_cast<int>(device_info.gfx_ver.minor) << "."
              << static_cast<int>(device_info.gfx_ver.revision);
        }
        IE_SET_METRIC_RETURN(GPU_UARCH_VERSION, s.str());
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        auto deviceName = StringRightTrim(device_info.dev_name, "NEO", false);
        deviceName += std::string(" (") + (device_info.dev_type == cldnn::device_type::discrete_gpu ? "dGPU" : "iGPU") + ")";
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, deviceName);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys;
        for (auto opt : _impl->m_config.key_config_map)
            configKeys.push_back(opt.first);
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        std::vector<std::string> capabilities;

        capabilities.push_back(METRIC_VALUE(FP32));
        capabilities.push_back(METRIC_VALUE(BIN));
        capabilities.push_back(METRIC_VALUE(BATCHED_BLOB));
        if (device_info.supports_fp16)
            capabilities.push_back(METRIC_VALUE(FP16));
        if (device_info.supports_imad || device_info.supports_immad)
            capabilities.push_back(METRIC_VALUE(INT8));
        if (device_info.supports_immad)
            capabilities.push_back(METRIC_VALUE(GPU_HW_MATMUL));

        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, capabilities);
    } else if (name == METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)) {
        std::tuple<unsigned int, unsigned int, unsigned int> range = std::make_tuple(1, 2, 1);
        IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, range);
    } else if (name == METRIC_KEY(RANGE_FOR_STREAMS)) {
        std::tuple<unsigned int, unsigned int> range = std::make_tuple(1, 2);
        IE_SET_METRIC_RETURN(RANGE_FOR_STREAMS, range);
    } else {
        IE_THROW() << "Unsupported metric key " << name;
    }
}

};  // namespace CLDNNPlugin

static const Version version = { {2, 1}, CI_BUILD_NUMBER, "clDNNPlugin" };
IE_DEFINE_PLUGIN_CREATE_FUNCTION(CLDNNPlugin::clDNNEngine, version)
