// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_metric_helpers.hpp"
#include "plugin.h"
#include "extension_mngr.h"
#include "weights_cache.hpp"
#include "extension.h"
#include "itt.h"
#include "serialize.h"

#include <threading/ie_executor_manager.hpp>
#include <memory>
#include <ie_plugin_config.hpp>
#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>
#include <ie_icore.hpp>
#include <fstream>
#include <vector>
#include <tuple>
#include <unordered_set>
#include <ie_system_conf.h>
#include <ie_ngraph_utils.hpp>

#include <transformations/opset_conversions/convert_opset3_to_opset2.hpp>
#include <transformations/opset_conversions/convert_opset2_to_opset1.hpp>

#include <transformations/common_optimizations/add_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/common_optimizations/fq_mul_fusion.hpp>
#include <transformations/common_optimizations/mul_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp>
#include <transformations/common_optimizations/convert_quantize_dequantize.hpp>
#include <transformations/common_optimizations/nop_elimination.hpp>
#include <transformations/common_optimizations/wrap_interpolate_into_transposes.hpp>
#include <transformations/common_optimizations/transpose_sinking.hpp>
#include "transformations/common_optimizations/convert_compression_only_to_legacy.hpp"
#include <transformations/op_conversions/convert_broadcast_to_tiles.hpp>
#include <transformations/op_conversions/convert_depth_to_space.hpp>
#include <transformations/op_conversions/convert_shuffle_channels3.hpp>
#include <transformations/op_conversions/convert_slice_to_strided_slice.hpp>
#include <transformations/op_conversions/convert_space_to_depth.hpp>
#include <transformations/op_conversions/convert_gelu.hpp>
#include <transformations/op_conversions/convert_gather_downgrade.hpp>
#include <transformations/op_conversions/convert_gather_upgrade.hpp>
#include <transformations/op_conversions/detection_output_downgrade.hpp>
#include <transformations/op_conversions/detection_output_upgrade.hpp>
#include <transformations/op_conversions/gelu7_downgrade.hpp>
#include <transformations/op_conversions/hswish_decomposition.hpp>
#include <transformations/op_conversions/hsigmoid_decomposition.hpp>
#include <transformations/op_conversions/mvn6_decomposition.hpp>
#include <transformations/op_conversions/normalize_l2_decomposition.hpp>
#include <transformations/op_conversions/reduce_l1_decomposition.hpp>
#include <transformations/op_conversions/reduce_l2_decomposition.hpp>
#include <transformations/op_conversions/softplus_decomposition.hpp>
#include <transformations/op_conversions/convert_space_to_batch.hpp>
#include <transformations/op_conversions/convert_batch_to_space.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>
#include <transformations/op_conversions/convert_subtract.hpp>
#include <transformations/op_conversions/softmax_decomposition.hpp>
#include <transformations/control_flow/unroll_tensor_iterator.hpp>
#include <transformations/op_conversions/convert_mod.hpp>
#include <transformations/op_conversions/convert_ti_to_sequences.hpp>
#include <transformations/op_conversions/lstm_cell_decomposition.hpp>
#include <transformations/op_conversions/rnn_cell_decomposition.hpp>
#include <transformations/op_conversions/gru_cell_decomposition.hpp>
#include <transformations/op_conversions/log_softmax_decomposition.hpp>
#include <transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp>
#include <transformations/op_conversions/simplify_ctc_greedy_decoder_seq_len.hpp>
#include <transformations/op_conversions/convert_previous_nms_to_nms_9.hpp>
#include <transformations/op_conversions/convert_nms9_to_nms_ie_internal.hpp>
#include <transformations/op_conversions/convert_multiclass_nms_to_multiclass_nms_ie.hpp>
#include <transformations/op_conversions/convert_matrix_nms_to_matrix_nms_ie.hpp>
#include <transformations/op_conversions/convert_deformable_conv_v8_to_v1.hpp>
#include <transformations/smart_reshape/matmul_sr.hpp>
#include <transformations/op_conversions/convert_minimum_to_power_and_max.hpp>
#include <transformations/op_conversions/convert_reduce_to_pooling.hpp>
#include <transformations/convert_precision.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/disable_decompression_convert_constant_folding.hpp>
#include <transformations/rt_info/fused_names_attribute.hpp>
#include <transformations/op_conversions/fq_decomposition.hpp>
#include <transformations/utils/utils.hpp>
#include <snippets/pass/collapse_subgraph.hpp>
#include "ngraph_transformations/snippets_mark_skipped.hpp"
#include <transformations/op_conversions/convert_roi_align_v9_to_v3.hpp>
#include <transformations/op_conversions/convert_roi_align_v3_to_v9.hpp>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/op/util/op_types.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/graph_util.hpp>

#include <transformations/common_optimizations/lin_op_sequence_fusion.hpp>

#include <transformations/low_precision/disable_convert_constant_folding_on_const_path.hpp>
#include <low_precision/common/quantization_granularity_restriction.hpp>
#include <low_precision/common/precisions_restriction.hpp>
#include <low_precision/convert_subtract_constant.hpp>
#include <low_precision/convolution.hpp>
#include <low_precision/convolution_backprop_data.hpp>
#include <low_precision/layer_transformation.hpp>
#include <low_precision/low_precision.hpp>
#include <low_precision/multiply_to_group_convolution.hpp>
#include <low_precision/network_helper.hpp>
#include "openvino/runtime/core.hpp"
#include "openvino/util/common_util.hpp"

#include <ie_algorithm.hpp>
#include "performance_heuristics.hpp"

#include "nodes/mvn.h"
#include "nodes/fake_quantize.h"
#include "nodes/normalize.h"
#include "ngraph_transformations/convert_to_cpu_specific_opset.hpp"
#include "ngraph_transformations/move_eltwise_up_data_movement.hpp"
#include "transformations/smart_reshape/smart_reshape.hpp"
#include "ngraph_transformations/swap_convert_transpose.hpp"

#if !defined(__arm__) && !defined(_M_ARM) && !defined(__aarch64__) && !defined(_M_ARM64)
#ifndef __GNUC_PREREQ
#define __GNUC_PREREQ(major, minor) ((((__GNUC__) << 16) + (__GNUC_MINOR__)) >= (((major) << 16) + (minor)))
#endif
# ifdef _WIN32
#  include <intrin.h>
#  include <windows.h>
# elif !(__GNUC_PREREQ(4, 3) && !defined(__APPLE__))
#  include <cpuid.h>
# endif
#endif

#include <cpu/x64/cpu_isa_traits.hpp>
#include <itt.h>

using namespace InferenceEngine;

#define IE_CPU_PLUGIN_THROW(...) IE_THROW(__VA_ARGS__) << "CPU plugin: "

namespace ov {
namespace intel_cpu {

static std::string getDeviceFullName() {
    std::string brand_string;
#if !defined(__arm__) && !defined(_M_ARM) && !defined(__aarch64__) && !defined(_M_ARM64)
    const unsigned int addr_list[3] = { 0x80000002, 0x80000003, 0x80000004 };
    unsigned int regs[4];
    for (auto addr : addr_list) {
        regs[0] = addr;
#ifdef _WIN32
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
    return brand_string;
}

Engine::Engine() :
    deviceFullName(getDeviceFullName()) {
    _pluginName = "CPU";
    extensionManager->AddExtension(std::make_shared<Extension>());
}

Engine::~Engine() {
    executorManager()->clear("CPU");
    executorManager()->clear("CPUStreamsExecutor");
    executorManager()->clear("CPUCallbackExecutor");
}

static void TransformationUpToCPUSpecificOpSet(std::shared_ptr<ngraph::Function> nGraphFunc, const bool _enableLPT,
                                               const bool _enableSnippets, const bool isLegacyApi) {
    ngraph::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ngraph::pass::InitNodeInfo>();

    const bool useLpt =
            _enableLPT &&
        ngraph::pass::low_precision::LowPrecision::isFunctionQuantized(nGraphFunc);
    auto defaultPrecisions = useLpt ? ngraph::pass::low_precision::precision_set::int8_support : std::vector<ov::element::Type>{};
    bool hasINT16orINT32Levels = false;
    if (useLpt) {
        CPU_LPT_SCOPE(LowPrecisionTransformations_Part1);
        hasINT16orINT32Levels = ngraph::pass::low_precision::LowPrecision::isFQLevelsPresent(
                nGraphFunc,
                {ngraph::pass::low_precision::levels::int16, ngraph::pass::low_precision::levels::int16_narrow_range,
                 ngraph::pass::low_precision::levels::int32, ngraph::pass::low_precision::levels::int32_narrow_range});
        if (hasINT16orINT32Levels) {
            defaultPrecisions = ngraph::pass::low_precision::precision_set::int8_int16_int32_support;
        }
        manager.register_pass<ngraph::pass::DisableConvertConstantFoldingOnConstPath>(defaultPrecisions);
    }
    auto get_convert_precisions = []() {
        precisions_array array = {
            {ngraph::element::i64,     ngraph::element::i32},
            {ngraph::element::u64,     ngraph::element::i32},
            {ngraph::element::i16,     ngraph::element::i32},
            {ngraph::element::u16,     ngraph::element::i32},
            {ngraph::element::u32,     ngraph::element::i32},
            {ngraph::element::f64,     ngraph::element::f32},
            {ngraph::element::f16,     ngraph::element::f32},
            {ngraph::element::boolean, ngraph::element::u8},
            {ngraph::element::i4,      ngraph::element::i8},
            {ngraph::element::u4,      ngraph::element::u8}
        };

        if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core))
            array.push_back({ngraph::element::bf16, ngraph::element::f32});

        return array;
    };

    static const auto precisions = get_convert_precisions();

    manager.register_pass<ngraph::pass::CommonOptimizations>();
    manager.register_pass<ngraph::pass::WrapInterpolateIntoTransposes>();
    manager.register_pass<ngraph::pass::TransposeSinking>();
    manager.register_pass<ngraph::pass::ConvertSequenceToTensorIterator>();
    manager.register_pass<ngraph::pass::ConvertOpSet3ToOpSet2>();
    manager.register_pass<ngraph::pass::ConvertOpSet2ToOpSet1>();
    manager.register_pass<ngraph::pass::ConvertTensorIteratorToSequence>();
    manager.register_pass<ngraph::pass::LSTMCellDecomposition>();
    manager.register_pass<ngraph::pass::GRUCellDecomposition>();
    manager.register_pass<ngraph::pass::RNNCellDecomposition>();
    manager.register_pass<ngraph::pass::ConvertNMS1ToNMS9>();
    manager.register_pass<ngraph::pass::ConvertNMS3ToNMS9>();
    manager.register_pass<ngraph::pass::ConvertNMS4ToNMS9>();
    manager.register_pass<ngraph::pass::ConvertNMS5ToNMS9>();
    manager.register_pass<ngraph::pass::ConvertNMS9ToNMSIEInternal>();
    manager.register_pass<ngraph::pass::ConvertMulticlassNmsToMulticlassNmsIE>();
    manager.register_pass<ngraph::pass::ConvertMatrixNmsToMatrixNmsIE>();
    manager.register_pass<ngraph::pass::TransposeMatMul>();
    manager.register_pass<ngraph::pass::ConstantFolding>();

    if (useLpt) {
        CPU_LPT_SCOPE(LowPrecisionTransformations_Part2);
        manager.register_pass<ngraph::pass::low_precision::ConvertSubtractConstant>(defaultPrecisions);
    }
    manager.register_pass<ngraph::pass::Validate>();
    manager.register_pass<ngraph::pass::ConvertPrecision>(precisions);
    manager.register_pass<ngraph::pass::EliminateConvert>();
    manager.register_pass<SwapConvertTranspose>();

    auto pass_config = manager.get_pass_config();

    using const_node_ptr = const std::shared_ptr<const ngraph::Node>;

    // SpaceToDepth/ DepthToSpace node implementation supports only equal input/output tensors with rank <= 5
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

    // Sequences supported by the plugin shouldn't be converted to TensorIterator.
    // sequence_length input is not supported in all Sequences, so if is_seq_len_provided() == true, we
    // should always convert to TensorIterator.
    // RNN/GRU/LSTM Sequences are supported with clip == 0, and with default activations.
    auto isSequencePrimitiveSupported = [](const_node_ptr &node) -> bool {
        const auto& data = node->input(0);
        const auto& data_pshape = data.get_partial_shape();
        // WA: dynamic shapes make impossible to check seq_len due to shapeOf subgraphs
        // but the sequence is still supported in CPU and doesn't need to be decomposed
        if (data_pshape.is_dynamic())
            return true;
        if (data_pshape.rank().is_static() && data_pshape.rank().get_length() > 1 && !data_pshape[1].is_static())
            return false;
        auto max_seq_len = data.get_shape().at(1);
        if (const auto &rnn_seq = std::dynamic_pointer_cast<const ngraph::opset6::RNNSequence>(node)) {
            return rnn_seq->get_clip() == 0.0f &&
                   !ngraph::op::util::is_seq_len_provided(rnn_seq->get_input_node_shared_ptr(2),
                                                          max_seq_len);
        } else if (const auto &gru_seq = std::dynamic_pointer_cast<const ngraph::opset6::GRUSequence>(
                node)) {
            return gru_seq->get_clip() == 0.0f &&
                   gru_seq->get_activations() == std::vector<std::string>{"sigmoid", "tanh"} &&
                   !ngraph::op::util::is_seq_len_provided(gru_seq->get_input_node_shared_ptr(2),
                                                          max_seq_len);
        } else if (const auto &lstm_seq = std::dynamic_pointer_cast<const ngraph::opset6::LSTMSequence>(
                node)) {
            return lstm_seq->get_clip() == 0.0f &&
                   lstm_seq->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"} &&
                   !ngraph::op::util::is_seq_len_provided(lstm_seq->get_input_node_shared_ptr(3),
                                                          max_seq_len);
        }
        return false;
    };

    pass_config->set_callback<ngraph::pass::ConvertRNNSequenceToTensorIterator,
                              ngraph::pass::ConvertGRUSequenceToTensorIterator,
                              ngraph::pass::ConvertLSTMSequenceToTensorIterator>(
            [isSequencePrimitiveSupported](const_node_ptr &node) -> bool {
                return isSequencePrimitiveSupported(node);
            });

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

    pass_config->set_callback<ngraph::pass::MVN6Decomposition>(
            [](const_node_ptr &node) -> bool {
                std::string errorMessage;
                return node::MVN::isSupportedOperation(node, errorMessage);
            });

    pass_config->set_callback<ngraph::pass::NormalizeL2Decomposition>(
            [](const_node_ptr &node) -> bool {
                std::string errorMsg;
                return node::NormalizeL2::isSupportedOperation(node, errorMsg);
            });

    pass_config->enable<ngraph::pass::SoftmaxDecomposition>();
    pass_config->set_callback<ngraph::pass::SoftmaxDecomposition>(
            [](const_node_ptr &node) -> bool {
                return node->input_value(0).get_partial_shape().rank().get_length() <= 5;
            });

    if (!isLegacyApi) {
        auto nmsCallback = [](const_node_ptr &node) -> bool {
                               for (size_t i = 0; i < node->get_output_size(); i++) {
                                   const auto outputs = node->get_output_target_inputs(i);
                                   for (const auto &out : outputs) {
                                       if (!ngraph::op::is_output(out.get_node())) {
                                           return false;
                                       }
                                   }
                               }
                               return true;
                           };

        pass_config->set_callback<ngraph::pass::ConvertNMS9ToNMSIEInternal>(nmsCallback);
        pass_config->set_callback<ngraph::pass::ConvertMulticlassNmsToMulticlassNmsIE>(nmsCallback);
        pass_config->set_callback<ngraph::pass::ConvertMatrixNmsToMatrixNmsIE>(nmsCallback);
    }

    // List of enabled/disabled transformations

    // Allow FP16 Converts to be folded and FP16 constants to be upgraded to FP32 data type
    pass_config->disable<ov::pass::DisableDecompressionConvertConstantFolding>();
    pass_config->disable<ov::pass::ConvertCompressedOnlyToLegacy>();

    pass_config->disable<ngraph::pass::ConvertGELU>();
    pass_config->disable<ngraph::pass::ConvertShuffleChannels3>();
    pass_config->disable<ngraph::pass::Gelu7Downgrade>();
    pass_config->disable<ngraph::pass::HSwishDecomposition>();
    pass_config->disable<ngraph::pass::ReduceL1Decomposition>();
    pass_config->disable<ngraph::pass::ReduceL2Decomposition>();
    pass_config->disable<ngraph::pass::SoftPlusDecomposition>();
    pass_config->disable<ngraph::pass::HSigmoidDecomposition>();
    pass_config->disable<ngraph::pass::ConvertMod>();
    pass_config->disable<ngraph::pass::LogSoftmaxDecomposition>();
    pass_config->disable<ngraph::pass::ConvertShuffleChannels3>();
    pass_config->disable<ngraph::pass::WeightsDequantizeToFakeQuantize>();
    pass_config->disable<ngraph::pass::SimplifyCTCGreedyDecoderSeqLen>();
    pass_config->disable<ngraph::pass::ConvertGather7ToGather1>();
    pass_config->disable<ngraph::pass::ConvertGather8ToGather7>();
    pass_config->disable<ngraph::pass::ConvertMinimum>();
    pass_config->disable<ngraph::pass::ConvertBroadcastToTiles>();
    pass_config->disable<ngraph::pass::ConvertReduceMeanToPooling>();
    pass_config->disable<ngraph::pass::ConvertReduceMaxToPooling>();
    pass_config->disable<ngraph::pass::ConvertReduceSumToPooling>();
    pass_config->disable<ngraph::pass::SliceToStridedSlice>();
    pass_config->disable<ngraph::pass::ConvertDetectionOutput8ToDetectionOutput1>();
    pass_config->disable<ngraph::pass::ConvertROIAlign9To3>();

    pass_config->enable<ngraph::pass::NormalizeL2Decomposition>();
    pass_config->enable<ngraph::pass::ConvertInterpolate1ToInterpolate4>();
    pass_config->enable<ngraph::pass::ConvertGather1ToGather7>();
    pass_config->enable<ngraph::pass::ConvertDetectionOutput1ToDetectionOutput8>();
    pass_config->enable<ngraph::pass::ConvertROIAlign3To9>();

    if (useLpt) {
        CPU_LPT_SCOPE(LowPrecisionTransformations_Part3);
        pass_config->set_callback<ngraph::pass::AddFakeQuantizeFusion,
                                  ngraph::pass::MulFakeQuantizeFusion,
                                  ngraph::pass::FakeQuantizeMulFusion>([](const_node_ptr &node) -> bool {
            std::string errMsg;
            return !node::FakeQuantize::isSupportedOperation(node, errMsg);
        });

        pass_config->set_callback<ngraph::pass::ConvertQuantizeDequantize>([&defaultPrecisions](const_node_ptr &node) -> bool {
            return ngraph::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForMultiply(node, defaultPrecisions);
        });

        pass_config->set_callback<ngraph::pass::ConvertSubtract>([&defaultPrecisions](const_node_ptr &node) -> bool {
            return ngraph::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForSubtract(node, defaultPrecisions);
        });
    }

    manager.run_passes(nGraphFunc);

    using namespace ngraph::pass::low_precision;
    if (useLpt) {
        CPU_LPT_SCOPE(LowPrecisionTransformations_Part4);
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "LowPrecisionTransformations");

        auto supportedPrecisions = std::vector<PrecisionsRestriction>({
            PrecisionsRestriction::create<ngraph::opset1::Convolution>({
                {0, {ngraph::element::u8, ngraph::element::i8}},
                {1, {ngraph::element::i8}},
            }),
            PrecisionsRestriction::create<ngraph::opset1::ConvolutionBackpropData>({
                {0, {ngraph::element::u8, ngraph::element::i8}},
                {1, {ngraph::element::i8}}
            }),
            PrecisionsRestriction::create<ngraph::opset1::GroupConvolution>({
                {0, {ngraph::element::u8}},
                {1, {ngraph::element::i8}}
            }),
            PrecisionsRestriction::create<ngraph::opset1::Multiply>({
                {0, {ngraph::element::u8}},
                {1, {ngraph::element::i8}},
            }),
            PrecisionsRestriction::create<ngraph::opset1::MatMul>({
                {0, {ngraph::element::u8, ngraph::element::i8}},
                {1, {ngraph::element::i8}}
            }),
        });

        auto quantizationRestrictions = std::vector<QuantizationGranularityRestriction>({
            QuantizationGranularityRestriction::create<ngraph::opset1::Convolution>({0}),
            QuantizationGranularityRestriction::create<ngraph::opset1::ConvolutionBackpropData>({0})
        });

        // for GNA networks reference execution
        bool updatePrecision = true;
        if (hasINT16orINT32Levels) {
            updatePrecision = false;
            supportedPrecisions = std::vector<PrecisionsRestriction>({});
        }

        ngraph::pass::Manager lptManager;
        lptManager.register_pass<ngraph::pass::low_precision::LowPrecision>(
            supportedPrecisions,
            quantizationRestrictions,
            LayerTransformation::Params(updatePrecision, ngraph::element::f32, defaultPrecisions));
        lptManager.get_pass_config()->set_callback<ngraph::pass::low_precision::MarkupPrecisions>([](const_node_ptr& node) -> bool {
            if (const auto mulitply = std::dynamic_pointer_cast<const ngraph::opset1::Multiply>(node)) {
                return !MultiplyToGroupConvolutionTransformation::canBeTransformedToGroupConvolution(mulitply);
            }
            return false;
        });
        lptManager.get_pass_config()->set_callback<ngraph::pass::low_precision::ConvolutionBackpropDataTransformation>(
            [&defaultPrecisions](const_node_ptr& node) -> bool {
            return LayerTransformation::isAsymmetricQuantization(node, defaultPrecisions) ||
                WeightableLayerTransformation::isAsymmetricOnWeights(node, defaultPrecisions);
        });
        lptManager.get_pass_config()->set_callback<ngraph::pass::low_precision::MultiplyToGroupConvolutionTransformation>([](const_node_ptr& node) -> bool {
            return true;//MultiplyToGroupConvolutionTransformation::isDynamicOrScalar(node);
        });
        lptManager.run_passes(nGraphFunc);
    }

    ngraph::pass::Manager postLPTPassManager;
    postLPTPassManager.register_pass<ngraph::pass::FakeQuantizeDecomposition>();
    postLPTPassManager.register_pass<ngraph::pass::UnrollTensorIterator>();
    postLPTPassManager.register_pass<ReshapePRelu>();

    postLPTPassManager.get_pass_config()->set_callback<ngraph::pass::FakeQuantizeDecomposition>([](const_node_ptr &node) -> bool {
        std::string errMsg;
        return node::FakeQuantize::isSupportedOperation(node, errMsg);
    });
    postLPTPassManager.get_pass_config()->set_callback<ngraph::pass::UnrollTensorIterator>([](const_node_ptr &node) -> bool {
        // UnrollTI transformation is disabled by default, is turned on by LowLatency transformation
        return node->get_rt_info().count("UNROLL_TI") == 0;
    });


    postLPTPassManager.register_pass<MoveEltwiseUpThroughDataMov>();
    postLPTPassManager.get_pass_config()->set_callback<MoveEltwiseUpThroughDataMov>([](const std::shared_ptr<const ngraph::Node>& node) -> bool {
        if (node->get_input_size() >= 2) {
            return node->get_input_element_type(1) == ngraph::element::i8 || node->get_input_element_type(1) == ngraph::element::u8;
        }
        return false;
    });

    postLPTPassManager.register_pass<ngraph::pass::ConstantFolding>();
    postLPTPassManager.run_passes(nGraphFunc);

    if (!useLpt && _enableSnippets && dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
        ngraph::pass::Manager tokenization_manager;
        tokenization_manager.register_pass<SnippetsMarkSkipped>();
        tokenization_manager.register_pass<ngraph::snippets::pass::EnumerateNodes>();
        tokenization_manager.register_pass<ngraph::snippets::pass::TokenizeSnippets>();
        tokenization_manager.get_pass_config()->set_callback<ngraph::snippets::pass::TokenizeSnippets>(
                [](const std::shared_ptr<const ov::Node>& n) -> bool {
                    const auto& inputs = n->inputs();
                    // todo: clarify whether we can evaluate snippets on const paths
                    const bool has_only_const_inputs = std::all_of(inputs.begin(), inputs.end(),
                                [](const ov::Input<const ov::Node> &in) {
                                        return ov::is_type<ov::op::v0::Constant>(in.get_source_output().get_node_shared_ptr());
                                      });
                    // todo: clarify whether we can evaluate snippets on inputs with larger ranks
                    auto rank_is_too_large = [](const ov::descriptor::Tensor& t ) {
                        // callback is called has_supported_in_out(), so it's safe to assume that the shapes are static
                        return t.get_partial_shape().rank().get_length() > 6;
                    };
                    const bool bad_input_rank = std::any_of(inputs.begin(), inputs.end(),
                                                            [&](const ov::Input<const ov::Node>& in) {return  rank_is_too_large(in.get_tensor());});
                    const auto& outputs = n->outputs();
                    const bool bad_output_rank = std::any_of(outputs.begin(), outputs.end(),
                                                             [&](const ov::Output<const ov::Node>& out) {return  rank_is_too_large(out.get_tensor());});
                    return has_only_const_inputs || bad_input_rank || bad_output_rank;
                });
        tokenization_manager.run_passes(nGraphFunc);
    }
}

static void Transformation(CNNNetwork& clonedNetwork, const bool _enableLPT, const bool _enableSnippets, const bool isLegacyApi) {
    auto nGraphFunc = clonedNetwork.getFunction();
    TransformationUpToCPUSpecificOpSet(nGraphFunc, _enableLPT, _enableSnippets, isLegacyApi);
    ConvertToCPUSpecificOpset(nGraphFunc);
}

static bool streamsSet(const std::map<std::string, std::string>& config) {
    return config.count(PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS) ||
           config.count(ov::num_streams.name());
}

void Engine::ApplyPerformanceHints(std::map<std::string, std::string> &config, const std::shared_ptr<ngraph::Function>& ngraphFunc) const {
    const bool streamsExplicitlySetForModel = streamsSet(config);
    // checking streams (to avoid overriding what user might explicitly set in the incoming config or previously via SetConfig)
    if (streamsExplicitlySetForModel ||
        streamsExplicitlySetForEngine)
        return;

    const auto& mode = config.find(CONFIG_KEY(PERFORMANCE_HINT));
    // the mode may have just arrived to the LoadNetwork, or was set with the plugin's SetConfig
    if (mode == config.end() && engConfig.perfHintsConfig.ovPerfHint.empty())
        return;
    /* performance hints set for network has higher pririty than engine ones.
     * This applies for all the configuration parameters */
    const auto mode_name = (mode != config.end()) ?
        PerfHintsConfig::CheckPerformanceHintValue(mode->second) :
        engConfig.perfHintsConfig.ovPerfHint;

    if (mode_name == CONFIG_VALUE(LATENCY)) {
        config[CONFIG_KEY(CPU_THROUGHPUT_STREAMS)] = CONFIG_VALUE(CPU_THROUGHPUT_NUMA);
        config[ov::num_streams.name()] = ov::util::to_string(ov::streams::NUMA);
    } else if (mode_name == CONFIG_VALUE(THROUGHPUT)) {
        const auto isa = dnnl::get_effective_cpu_isa();
        float isaSpecificThreshold = 1.0f;
        switch (isa) {
        case dnnl::cpu_isa::sse41 :
            isaSpecificThreshold = 0.5f;
            break;
        case dnnl::cpu_isa::avx2:
        case dnnl::cpu_isa::avx512_core:
            isaSpecificThreshold = 1.0f;
            break;
        case dnnl::cpu_isa::avx512_core_vnni:
        case dnnl::cpu_isa::avx2_vnni:
            isaSpecificThreshold = 2.0f;
            break;
        case dnnl::cpu_isa::avx512_core_amx:
            isaSpecificThreshold = 4.0f;
            break;
        default:
            isaSpecificThreshold = 1.0f;
        }
        // the more "capable" the CPU in general, the more streams we may want to keep to keep it utilized
        const float memThresholdAssumeLimitedForISA = ov::MemBandwidthPressure::LIMITED/isaSpecificThreshold;
        const float L2_cache_size = dnnl::utils::get_cache_size(2 /*level*/, true /*per core */);
        ov::MemBandwidthPressure networkToleranceForLowCache = ov::MemBandwidthPressureTolerance(
            ngraphFunc,
            L2_cache_size, memThresholdAssumeLimitedForISA);
        // num of phys CPU cores (most aggressive value for #streams)
        const auto num_cores = getNumberOfCPUCores();
        // less aggressive
        const auto num_streams_less_aggressive = num_cores / 2;
        // default #streams value (most conservative)
        const auto default_num_streams = IStreamsExecutor::Config::GetDefaultNumStreams();
        int num_streams = default_num_streams;
        if (networkToleranceForLowCache.max_mem_tolerance == ov::MemBandwidthPressure::UNKNOWN) {
            if ((networkToleranceForLowCache.ratio_compute_convs == ov::MemBandwidthPressure::ALL)
                || (networkToleranceForLowCache.ratio_compute_deconvs == ov::MemBandwidthPressure::ALL)) {
                // all relevant layers (convs, etc) are compute-limited, the most aggressive val for #streams
                num_streams = num_cores;
            }   // otherwise (no recognized layers) falling back to the default value
        } else if (networkToleranceForLowCache.max_mem_tolerance > memThresholdAssumeLimitedForISA) {
            // network is below the ISA-specific threshold
            num_streams = num_cores;
        } else if (networkToleranceForLowCache.max_mem_tolerance > ov::MemBandwidthPressure::LIMITED) {
            // network is below general threshold
            num_streams = std::max(default_num_streams, num_streams_less_aggressive);
        }
        auto num_requests = config.find(CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS));
        if (num_requests != config.end()) {  // arrived with config to the LoadNetwork (and thus higher pri)
            auto val = PerfHintsConfig::CheckPerformanceHintRequestValue(num_requests->second);
            if (val > 0)
                num_streams = std::min(num_streams, val);
        } else if (engConfig.perfHintsConfig.ovPerfHintNumRequests) {  //set thru SetConfig to the plugin, 2nd priority
            num_streams = std::min(num_streams,
                                   engConfig.perfHintsConfig.ovPerfHintNumRequests);
        }
        config[CONFIG_KEY(CPU_THROUGHPUT_STREAMS)] = std::to_string(num_streams);
        config[ov::num_streams.name()] = ov::util::to_string(num_streams);
    }
}

InferenceEngine::IExecutableNetworkInternal::Ptr
Engine::LoadExeNetworkImpl(const InferenceEngine::CNNNetwork &network, const std::map<std::string, std::string> &orig_config) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, "Engine::LoadExeNetworkImpl");

    // verification of supported input
    for (const auto &ii : network.getInputsInfo()) {
        auto input_precision = ii.second->getPrecision();

        using hash_t = std::hash<typename std::underlying_type<Precision::ePrecision>::type>;

        static const std::unordered_set<Precision::ePrecision, hash_t> supported_precisions = {
            Precision::U8,   Precision::I8,
            Precision::U16,  Precision::I16,
            Precision::U32,  Precision::I32,
            Precision::U64,  Precision::I64,
            Precision::BF16, Precision::FP16,
            Precision::FP32, Precision::FP64,
            Precision::BOOL
        };

        if (!supported_precisions.count(input_precision)) {
            IE_CPU_PLUGIN_THROW(NotImplemented)
                        << "Input image format " << input_precision << " is not supported yet...";
        }
    }

    auto config = orig_config;

    CNNNetwork clonedNetwork = InferenceEngine::details::cloneNetwork(network);
    const auto& lptProp = config.find(InferenceEngine::PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE);
    const bool enableLPT = (lptProp != config.end() && lptProp->second == PluginConfigParams::YES) /* enabled in the orig_config*/
            || Config::LPTransformsMode::On == engConfig.lpTransformsMode /* or already enabled for the plugin */;
    const auto& BF16Prop = config.find(InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16);
    bool enableBF16;
    if (BF16Prop != config.end()) {
        if (BF16Prop->second == PluginConfigParams::YES) {
            enableBF16 = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core);
        } else {
            enableBF16 = false;
        }
    } else {
        enableBF16 = engConfig.enforceBF16 && dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core);
    }
    const auto& modelCacheProp = config.find(InferenceEngine::PluginConfigParams::KEY_CACHE_DIR);
    const bool enableModelCache = (modelCacheProp != config.end() && !modelCacheProp->second.empty())
            || !engConfig.cache_dir.empty();
    const auto& dynamicBatchProp = config.find(InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED);
    const bool enableDynamicBatch = (dynamicBatchProp != config.end() && dynamicBatchProp->second == PluginConfigParams::YES)
            || engConfig.enableDynamicBatch;
    const bool enableSnippets = !(enableModelCache || enableDynamicBatch || enableBF16);
    auto nGraphFunc = clonedNetwork.getFunction();
    TransformationUpToCPUSpecificOpSet(nGraphFunc, enableLPT, enableSnippets, isLegacyAPI());

    // need to check that all outputs have static shapes
    // checking that all inputs have static shapes is performed in the common part
    if (isLegacyAPI()) {
        for (const auto& res : nGraphFunc->get_results()) {
            if (res->get_input_partial_shape(0).is_dynamic()) {
                IE_THROW() << "CPU plug-in can't load a model with dynamic output shapes via legacy API.";
            }
        }
    }

    ApplyPerformanceHints(config, nGraphFunc);

    ConvertToCPUSpecificOpset(nGraphFunc);

    // update the props after the perf mode translated to configs
    // TODO: Clarify the behavior of SetConfig method. Skip eng_config or not?
    Config conf = engConfig;

    conf.readProperties(config);
    if (conf.enableDynamicBatch) {
        conf.batchLimit = static_cast<int>(network.getBatchSize());
    }

    return std::make_shared<ExecNetwork>(clonedNetwork, conf, extensionManager, shared_from_this());
}

void Engine::SetConfig(const std::map<std::string, std::string> &config) {
    streamsExplicitlySetForEngine = streamsSet(config);

    engConfig.readProperties(config);
}

bool Engine::isLegacyAPI() const {
    const auto& core = GetCore();
    if (!core)
        IE_CPU_PLUGIN_THROW() << "Unable to get API version. Core is unavailable";

    return !core->isNewAPI();
}

Parameter Engine::GetConfigLegacy(const std::string& name, const std::map<std::string, Parameter>& options) const {
    Parameter result;
    auto option = engConfig._config.find(name);
    if (option != engConfig._config.end()) {
        result = option->second;
    } else {
        IE_CPU_PLUGIN_THROW() << ". Unsupported config parameter: " << name;
    }
    return result;
}

Parameter Engine::GetConfig(const std::string& name, const std::map<std::string, Parameter>& options) const {
    if (isLegacyAPI())
        return GetConfigLegacy(name, options);

    if (name == ov::optimal_number_of_infer_requests) {
        const auto streams = engConfig.streamExecutorConfig._streams;
        return decltype(ov::optimal_number_of_infer_requests)::value_type(streams); // ov::optimal_number_of_infer_requests has no negative values
    } else if (name == ov::num_streams) {
        const auto streams = engConfig.streamExecutorConfig._streams;
        return decltype(ov::num_streams)::value_type(streams); // ov::num_streams has special negative values (AUTO = -1, NUMA = -2)
    } else if (name == ov::affinity) {
        const auto affinity = engConfig.streamExecutorConfig._threadBindingType;
        switch (affinity) {
        case InferenceEngine::IStreamsExecutor::ThreadBindingType::NONE:
            return ov::Affinity::NONE;
        case InferenceEngine::IStreamsExecutor::ThreadBindingType::CORES:
            return ov::Affinity::CORE;
        case InferenceEngine::IStreamsExecutor::ThreadBindingType::NUMA:
            return ov::Affinity::NUMA;
        case InferenceEngine::IStreamsExecutor::ThreadBindingType::HYBRID_AWARE:
            return ov::Affinity::HYBRID_AWARE;
        }
        return ov::Affinity::NONE;
    } else if (name == ov::inference_num_threads) {
        const auto num_threads = engConfig.streamExecutorConfig._threads;
        return decltype(ov::inference_num_threads)::value_type(num_threads);
    } else if (name == ov::enable_profiling.name()) {
        const bool perfCount = engConfig.collectPerfCounters;
        return decltype(ov::enable_profiling)::value_type(perfCount);
    } else if (name == ov::hint::inference_precision) {
        const auto enforceBF16 = engConfig.enforceBF16;
        const auto inference_precision = enforceBF16 ? ov::element::bf16 : ov::element::f32;
        return decltype(ov::hint::inference_precision)::value_type(inference_precision);
    } else if (name == ov::hint::performance_mode) {
        const auto perfHint = ov::util::from_string(engConfig.perfHintsConfig.ovPerfHint, ov::hint::performance_mode);
        return perfHint;
    } else if (name == ov::hint::num_requests) {
        const auto perfHintNumRequests = engConfig.perfHintsConfig.ovPerfHintNumRequests;
        return decltype(ov::hint::num_requests)::value_type(perfHintNumRequests);
    }
    /* Internally legacy parameters are used with new API as part of migration procedure.
     * This fallback can be removed as soon as migration completed */
    return GetConfigLegacy(name, options);
}

Parameter Engine::GetMetricLegacy(const std::string& name, const std::map<std::string, Parameter>& options) const {
    if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics = {
            METRIC_KEY(AVAILABLE_DEVICES),
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(FULL_DEVICE_NAME),
            METRIC_KEY(OPTIMIZATION_CAPABILITIES),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS),
            METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS),
            METRIC_KEY(RANGE_FOR_STREAMS),
            METRIC_KEY(IMPORT_EXPORT_SUPPORT),
        };
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, deviceFullName);
    } else if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
        std::vector<std::string> availableDevices = { "" };
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, availableDevices);
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        std::vector<std::string> capabilities;
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16))
            capabilities.push_back(METRIC_VALUE(BF16));
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core))
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
    } else if (name == METRIC_KEY(IMPORT_EXPORT_SUPPORT)) {
        IE_SET_METRIC_RETURN(IMPORT_EXPORT_SUPPORT, true);
    }

    IE_CPU_PLUGIN_THROW() << "Unsupported metric key: " << name;
}

Parameter Engine::GetMetric(const std::string& name, const std::map<std::string, Parameter>& options) const {
    if (isLegacyAPI())
        return GetMetricLegacy(name, options);

    auto RO_property = [](const std::string& propertyName) {
        return ov::PropertyName(propertyName, ov::PropertyMutability::RO);
    };
    auto RW_property = [](const std::string& propertyName) {
        return ov::PropertyName(propertyName, ov::PropertyMutability::RW);
    };

    if (name == ov::supported_properties) {
        std::vector<ov::PropertyName> roProperties {RO_property(ov::supported_properties.name()),
                                                    RO_property(ov::available_devices.name()),
                                                    RO_property(ov::range_for_async_infer_requests.name()),
                                                    RO_property(ov::range_for_streams.name()),
                                                    RO_property(ov::device::full_name.name()),
                                                    RO_property(ov::device::capabilities.name()),
                                                    RO_property(ov::cache_dir.name())   // WA Can be removed after implementing snippet serialization.
        };
        // the whole config is RW before network is loaded.
        std::vector<ov::PropertyName> rwProperties {RW_property(ov::num_streams.name()),
                                                    RW_property(ov::affinity.name()),
                                                    RW_property(ov::inference_num_threads.name()),
                                                    RW_property(ov::enable_profiling.name()),
                                                    RW_property(ov::hint::inference_precision.name()),
                                                    RW_property(ov::hint::performance_mode.name()),
                                                    RW_property(ov::hint::num_requests.name()),
        };

        std::vector<ov::PropertyName> supportedProperties;
        supportedProperties.reserve(roProperties.size() + rwProperties.size());
        supportedProperties.insert(supportedProperties.end(), roProperties.begin(), roProperties.end());
        supportedProperties.insert(supportedProperties.end(), rwProperties.begin(), rwProperties.end());

        return decltype(ov::supported_properties)::value_type(supportedProperties);
    } else if (name == ov::device::full_name) {
        return decltype(ov::device::full_name)::value_type(deviceFullName);
    } else if (name == ov::available_devices) {
        const std::vector<std::string> availableDevices = { "" };
        return decltype(ov::available_devices)::value_type(availableDevices);
    } else if (name == ov::device::capabilities) {
        std::vector<std::string> capabilities;
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16))
            capabilities.push_back(METRIC_VALUE(BF16));
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core))
            capabilities.push_back(METRIC_VALUE(WINOGRAD));
        capabilities.push_back(METRIC_VALUE(FP32));
        capabilities.push_back(METRIC_VALUE(FP16));
        capabilities.push_back(METRIC_VALUE(INT8));
        capabilities.push_back(METRIC_VALUE(BIN));
        capabilities.push_back(ov::device::capability::EXPORT_IMPORT);
        return decltype(ov::device::capabilities)::value_type(capabilities);
    } else if (name == ov::range_for_async_infer_requests) {
        const std::tuple<unsigned int, unsigned int, unsigned int> range = std::make_tuple(1, 1, 1);
        return decltype(ov::range_for_async_infer_requests)::value_type(range);
    } else if (name == ov::range_for_streams) {
        const std::tuple<unsigned int, unsigned int> range = std::make_tuple(1, parallel_get_max_threads());
        return decltype(ov::range_for_streams)::value_type(range);
    }
    /* Internally legacy parameters are used with new API as part of migration procedure.
     * This fallback can be removed as soon as migration completed */
    return GetMetricLegacy(name, options);
}

void Engine::AddExtension(const InferenceEngine::IExtensionPtr& extension) {
    extensionManager->AddExtension(extension);
}

QueryNetworkResult Engine::QueryNetwork(const CNNNetwork& network, const std::map<std::string, std::string>& config) const {
    QueryNetworkResult res;

    WeightsSharing::Ptr fake_w_cache;
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

        auto clonedNetwork = InferenceEngine::details::cloneNetwork(network);
        auto clonnedFunction = clonedNetwork.getFunction();
        const auto& lptProp = config.find(InferenceEngine::PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE);
        const bool enableLPT = (lptProp != config.end() && lptProp->second == PluginConfigParams::YES) /* enabled in the orig_config*/
                               || Config::LPTransformsMode::On == engConfig.lpTransformsMode /* or already enabled */;
        const bool enableSnippets = !(conf.cache_dir.empty() || conf.enableDynamicBatch || (conf.enforceBF16
                && dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)));
        Transformation(clonedNetwork, enableLPT, enableSnippets, isLegacyAPI());
        auto ops = clonnedFunction->get_ordered_ops();

        //Mark removed nodes as supported
        std::unordered_set<std::string> supported = GetRemovedNodes(function, clonnedFunction);;
        std::unordered_set<std::string> unsupported;

        auto layerIsSupported = [&](const std::shared_ptr<ngraph::Node>& op) {
            std::unique_ptr<Node> ptr;
            try {
                ptr.reset(Node::factory().create(op, {dnnl::engine::kind::cpu, 0}, extensionManager, fake_w_cache));
            } catch (const InferenceEngine::Exception&) {
                return false;
            }
            return true;
        };

        for (auto&& op : ops) {
            bool isSupported = false;
            bool wasNodeAlreadyChecked = false;
            if (InferenceEngine::details::contains(originalOps, op->get_friendly_name())) {
                isSupported = layerIsSupported(op);
                wasNodeAlreadyChecked = true;
                if (isSupported) {
                    supported.emplace(op->get_friendly_name());
                } else {
                    unsupported.emplace(op->get_friendly_name());
                }
            }

            for (auto&& fusedLayerName : ngraph::getFusedNamesVector(op)) {
                if (InferenceEngine::details::contains(originalOps, fusedLayerName)) {
                    if (!wasNodeAlreadyChecked) {
                        isSupported = layerIsSupported(op);
                        wasNodeAlreadyChecked = true;
                    }
                    if (isSupported) {
                        supported.emplace(fusedLayerName);
                    } else {
                        unsupported.emplace(fusedLayerName);
                    }
                }
            }
        }
        for (auto&& unsupportedNode : unsupported) {
            supported.erase(unsupportedNode);
        }
        for (auto&& node : function->get_ops()) {
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
    } else {
        IE_CPU_PLUGIN_THROW() << "Only ngraph-based models are supported!";
    }

    return res;
}

InferenceEngine::IExecutableNetworkInternal::Ptr Engine::ImportNetwork(std::istream& networkModel,
                                            const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "ImportNetwork");

    CNNNetworkDeserializer deserializer(networkModel,
        [this](const std::string& model, const Blob::CPtr& weights) {
            return GetCore()->ReadNetwork(model, weights, true);
        });

    CNNNetwork cnnnetwork;
    deserializer >> cnnnetwork;

    Config conf = engConfig;
    conf.readProperties(config);

    if (conf.enableDynamicBatch) {
        conf.batchLimit = static_cast<int>(cnnnetwork.getBatchSize());
    }

    auto execNetwork = std::make_shared<ExecNetwork>(cnnnetwork, conf, extensionManager, shared_from_this());

    execNetwork->setNetworkInputs(cnnnetwork.getInputsInfo());
    execNetwork->setNetworkOutputs(cnnnetwork.getOutputsInfo());
    SetExeNetworkInfo(execNetwork, cnnnetwork.getFunction());

    return execNetwork;
}

}   // namespace intel_cpu
}   // namespace ov

using namespace ov::intel_cpu;
static const Version version = {{2, 1}, CI_BUILD_NUMBER, "openvino_intel_cpu_plugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(Engine, version)
