// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformation_pipeline.h"

// Operations
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset2.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset10.hpp"
#include <ov_ops/augru_cell.hpp>
#include <ov_ops/augru_sequence.hpp>

// Common transformations
#include "transformations/common_optimizations/add_fake_quantize_fusion.hpp"
#include "transformations/common_optimizations/convert_compression_only_to_legacy.hpp"
#include "transformations/common_optimizations/convert_quantize_dequantize.hpp"
#include "transformations/common_optimizations/fq_mul_fusion.hpp"
#include "transformations/common_optimizations/mul_fake_quantize_fusion.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/transpose_sinking.hpp"
#include "transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp"
#include "transformations/common_optimizations/augru_cell_fusion.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/common_optimizations/wrap_interpolate_into_transposes.hpp"
#include "transformations/control_flow/unroll_tensor_iterator.hpp"
#include "transformations/disable_decompression_convert_constant_folding.hpp"
#include "transformations/op_conversions/convert_batch_to_space.hpp"
#include "transformations/op_conversions/convert_broadcast_to_tiles.hpp"
#include "transformations/op_conversions/convert_depth_to_space.hpp"
#include "transformations/op_conversions/convert_gather_downgrade.hpp"
#include "transformations/op_conversions/convert_gather_upgrade.hpp"
#include "transformations/op_conversions/convert_gelu.hpp"
#include "transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp"
#include "transformations/op_conversions/convert_matrix_nms_to_matrix_nms_ie.hpp"
#include "transformations/op_conversions/convert_minimum_to_power_and_max.hpp"
#include "transformations/op_conversions/convert_mod.hpp"
#include "transformations/op_conversions/convert_multiclass_nms_to_multiclass_nms_ie.hpp"
#include "transformations/op_conversions/convert_nms9_to_nms_ie_internal.hpp"
#include "transformations/op_conversions/convert_previous_nms_to_nms_9.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"
#include "transformations/op_conversions/convert_roi_align_v3_to_v9.hpp"
#include "transformations/op_conversions/convert_roi_align_v9_to_v3.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"
#include "transformations/op_conversions/convert_shuffle_channels3.hpp"
#include "transformations/op_conversions/convert_slice_to_strided_slice.hpp"
#include "transformations/op_conversions/convert_space_to_batch.hpp"
#include "transformations/op_conversions/convert_space_to_depth.hpp"
#include "transformations/op_conversions/convert_subtract.hpp"
#include "transformations/op_conversions/convert_ti_to_sequences.hpp"
#include "transformations/op_conversions/detection_output_downgrade.hpp"
#include "transformations/op_conversions/detection_output_upgrade.hpp"
#include "transformations/op_conversions/eye_decomposition.hpp"
#include "transformations/op_conversions/fq_decomposition.hpp"
#include "transformations/op_conversions/gelu7_downgrade.hpp"
#include "transformations/op_conversions/hsigmoid_decomposition.hpp"
#include "transformations/op_conversions/hswish_decomposition.hpp"
#include "transformations/op_conversions/gru_cell_decomposition.hpp"
#include "transformations/op_conversions/lstm_cell_decomposition.hpp"
#include "transformations/op_conversions/mvn6_decomposition.hpp"
#include "transformations/op_conversions/normalize_l2_decomposition.hpp"
#include "transformations/op_conversions/reduce_l1_decomposition.hpp"
#include "transformations/op_conversions/reduce_l2_decomposition.hpp"
#include "transformations/op_conversions/rnn_cell_decomposition.hpp"
#include "transformations/op_conversions/simplify_ctc_greedy_decoder_seq_len.hpp"
#include "transformations/op_conversions/softplus_decomposition.hpp"
#include "transformations/op_conversions/softsign_decomposition.hpp"
#include "transformations/op_conversions/softmax_decomposition.hpp"
#include "transformations/op_conversions/unique_decomposition.hpp"
#include "transformations/opset_conversions/convert_opset2_to_opset1.hpp"
#include "transformations/opset_conversions/convert_opset3_to_opset2.hpp"
#include "transformations/smart_reshape/matmul_sr.hpp"
#include "transformations/init_node_info.hpp"
#include "utils/ngraph_transformation.hpp"

// LPT transformations
#include "transformations/low_precision/mark_dequantization_subgraph.hpp"
#include "low_precision/convolution_backprop_data.hpp"
#include "low_precision/convert_subtract_constant.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/multiply_to_group_convolution.hpp"
#include "low_precision/group_convolution.hpp"

// CPU specific transformations
#include "ngraph_transformations/convert_to_cpu_specific_opset.hpp"
#include "ngraph_transformations/snippets_mark_skipped.hpp"
#include "ngraph_transformations/mha_fusion.hpp"
#include "ngraph_transformations/convert_to_interaction.hpp"
#include "ngraph_transformations/convert_fq_rnn_to_quantized_rnn.hpp"
#include "ngraph_transformations/move_eltwise_up_data_movement.hpp"
#include "ngraph_transformations/swap_convert_transpose.hpp"

// Snippets
#include "snippets/pass/tokenization.hpp"
#include "snippets/pass/common_optimizations.hpp"

// Misc
#include "nodes/mvn.h"
#include "nodes/normalize.h"
#include "nodes/fake_quantize.h"
#include "nodes/mha.h"

#include "dnnl.hpp"
#include <cpu/x64/cpu_isa_traits.hpp>

namespace ov {
namespace intel_cpu {

using const_node_ptr = const std::shared_ptr<const ov::Node>;

bool Transformations::fuse_type_to_convert(const std::shared_ptr<ngraph::Node>& node, ov::element::Type to, size_t idx) {
    if (auto convert = ov::as_type_ptr<ov::opset10::Convert>(node)) {
        // For Convert node, converting precision from floating point to boolean will lead to mathematical
        // error, because here the output precision boolean is replaced by u8. E.g. floating point value 0.01
        // is converted to be 1 for boolean, but 0 for u8. Thus an Abs and Ceil node should be added before the
        // Convert node for this scenario.
        if (convert->input(0).get_element_type().is_real() &&
            convert->get_convert_element_type() == ngraph::element::boolean && to.is_integral_number()) {
            auto abs = std::make_shared<ov::opset10::Abs>(convert->input_value(0).get_node_shared_ptr());
            auto ceil = std::make_shared<ov::opset10::Ceiling>(abs);
            auto new_convert = std::make_shared<ov::opset10::Convert>(ceil, to);
            new_convert->set_friendly_name(convert->get_friendly_name());
            ov::copy_runtime_info(convert, {abs, ceil, new_convert});
            ov::replace_node(convert, new_convert);
            return true;
        } else {
            convert->set_convert_element_type(to);
            return true;
        }
    }
    return false;
}

void Transformations::UpToCpuSpecificOpSet() {
    const bool useLpt = enableLpt &&
        ngraph::pass::low_precision::LowPrecision::isFunctionQuantized(model) &&
        CPU_DEBUG_CAP_IS_TRANSFORMATION_ENABLED(config.debugCaps, Lpt);

    const bool useSnippets = snippetsMode != Config::SnippetsMode::Disable &&
        CPU_DEBUG_CAP_IS_TRANSFORMATION_ENABLED(config.debugCaps, Snippets);

    auto defaultPrecisions = useLpt ? ngraph::pass::low_precision::precision_set::int8_support : std::vector<ov::element::Type>{};
    bool hasINT16orINT32Levels = false;

    if (useLpt) {
        CPU_LPT_SCOPE(LowPrecisionTransformations_Part1);
        hasINT16orINT32Levels = ngraph::pass::low_precision::LowPrecision::isFQLevelsPresent(
            model,
            {ngraph::pass::low_precision::levels::int16, ngraph::pass::low_precision::levels::int16_narrow_range,
             ngraph::pass::low_precision::levels::int32, ngraph::pass::low_precision::levels::int32_narrow_range});
        if (hasINT16orINT32Levels) {
            defaultPrecisions = ngraph::pass::low_precision::precision_set::int8_int16_int32_support;
        }
    }

    PreLpt(defaultPrecisions, isLegacyApi);

    if (useLpt)
        Lpt(hasINT16orINT32Levels, defaultPrecisions);

    PostLpt();

    if (useSnippets)
        Snippets();
}

void Transformations::CpuSpecificOpSet(void) {
    CPU_DEBUG_CAP_TRANSFORMATION_SCOPE(this, Specific);

    ConvertToCPUSpecificOpset(model);
}

void Transformations::PreLpt(const std::vector<ov::element::Type>& defaultPrecisions, const bool isLegacyApi) {
    CPU_DEBUG_CAP_TRANSFORMATION_SCOPE(this, PreLpt);

    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::pass::InitNodeInfo>();

    const bool useLpt = !defaultPrecisions.empty();
    if (useLpt) {
        manager.register_pass<ov::pass::MarkDequantizationSubgraph>(defaultPrecisions);
    }

    auto get_convert_precisions = []() {
        precisions_array array = {
            {ov::element::i64,     ov::element::i32},
            {ov::element::u64,     ov::element::i32},
            {ov::element::i16,     ov::element::i32},
            {ov::element::u16,     ov::element::i32},
            {ov::element::u32,     ov::element::i32},
            {ov::element::f64,     ov::element::f32},
            {ov::element::f16,     ov::element::f32},
            {ov::element::boolean, ov::element::u8},
            {ov::element::i4,      ov::element::i8},
            {ov::element::u4,      ov::element::u8}
        };

        if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core))
            array.push_back({ov::element::bf16, ov::element::f32});

        return array;
    };
    static const auto precisions = get_convert_precisions();
    type_to_fuse_map type_to_fuse = {{ov::opset10::Convert::get_type_info_static(), fuse_type_to_convert}};

    manager.register_pass<ov::pass::AUGRUCellFusion>();
    manager.register_pass<ov::pass::CommonOptimizations>();
    manager.register_pass<ov::pass::WrapInterpolateIntoTransposes>();
    manager.register_pass<ov::pass::TransposeSinking>();
    manager.register_pass<ov::pass::ConvertSequenceToTensorIterator>();
    manager.register_pass<ov::pass::ConvertOpSet3ToOpSet2>();
    manager.register_pass<ov::pass::ConvertOpSet2ToOpSet1>();
    manager.register_pass<ov::pass::LSTMCellDecomposition>();
    manager.register_pass<ov::pass::GRUCellDecomposition>();
    manager.register_pass<ov::pass::RNNCellDecomposition>();
    manager.register_pass<ov::pass::ConvertNMS1ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS3ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS4ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS5ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS9ToNMSIEInternal>();
    manager.register_pass<ov::pass::Validate>();
    manager.register_pass<ov::pass::ConvertMulticlassNmsToMulticlassNmsIE>();
    manager.register_pass<ov::pass::Validate>();
    manager.register_pass<ov::pass::ConvertMatrixNmsToMatrixNmsIE>();
    manager.register_pass<ov::pass::Validate>();
    manager.register_pass<ov::pass::TransposeMatMul>();
    manager.register_pass<ov::pass::ConstantFolding>();

    if (useLpt) {
        CPU_LPT_SCOPE(LowPrecisionTransformations_Part2);
        manager.register_pass<ngraph::pass::low_precision::ConvertSubtractConstant>(defaultPrecisions);
    }
    manager.register_pass<ov::pass::Validate>();
    manager.register_pass<ov::pass::ConvertPrecision>(precisions, type_to_fuse);
    manager.register_pass<ov::pass::EliminateConvert>();
    manager.register_pass<SwapConvertTranspose>();
    manager.register_pass<ConvertToInteraction>();
    manager.register_pass<ConvertInteractionInt8>();

    auto pass_config = manager.get_pass_config();

    // SpaceToDepth/ DepthToSpace node implementation supports only equal input/output tensors with rank <= 5
    pass_config->set_callback<ov::pass::ConvertSpaceToDepth,
                              ov::pass::ConvertDepthToSpace>(
                                  [](const_node_ptr &node) -> bool {
                                      return node->input_value(0).get_shape().size() <= 5lu &&
                                          node->input_value(0).get_shape().size() == node->get_output_shape(0).size();
                                  });

    pass_config->set_callback<ov::pass::ConvertBatchToSpace,
                              ov::pass::ConvertSpaceToBatch>(
                                  [](const_node_ptr &node) -> bool {
                                      const auto & rank = node->input(0).get_partial_shape().rank().get_length();
                                      return rank == 4lu || rank == 5lu;
                                  });

    auto isCellPrimitiveSupported = [](const_node_ptr &node) -> bool {
        if (const auto &rnn_cell = std::dynamic_pointer_cast<const ov::opset4::RNNCell>(node)) {
            return rnn_cell->get_clip() == 0.0f;
        } else if (const auto &gru_cell = std::dynamic_pointer_cast<const ov::opset4::GRUCell>(
                       node)) {
            return gru_cell->get_clip() == 0.0f
                && gru_cell->get_activations() == std::vector<std::string>{"sigmoid", "tanh"};
        } else if (const auto &augru_cell = std::dynamic_pointer_cast<const ov::op::internal::AUGRUCell>(
                       node)) {
            return augru_cell->get_clip() == 0.0f
                && augru_cell->get_activations() == std::vector<std::string>{"sigmoid", "tanh"};
        } else if (const auto &lstm_cell = std::dynamic_pointer_cast<const ov::opset4::LSTMCell>(
                       node)) {
            return lstm_cell->get_clip() == 0.0f &&
                lstm_cell->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"};
        } else if (const auto &lstm_cell_v1 = std::dynamic_pointer_cast<const ov::opset1::LSTMCell>(
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
        if (const auto &rnn_seq = std::dynamic_pointer_cast<const ov::opset6::RNNSequence>(node)) {
            return rnn_seq->get_clip() == 0.0f &&
                !ov::op::util::is_seq_len_provided(rnn_seq->get_input_node_shared_ptr(2),
                                                   max_seq_len);
        } else if (const auto &gru_seq = std::dynamic_pointer_cast<const ov::opset6::GRUSequence>(
                       node)) {
            return gru_seq->get_clip() == 0.0f &&
                gru_seq->get_activations() == std::vector<std::string>{"sigmoid", "tanh"} &&
                !ov::op::util::is_seq_len_provided(gru_seq->get_input_node_shared_ptr(2),
                                                   max_seq_len);
        } else if (const auto &augru_seq = std::dynamic_pointer_cast<const ov::op::internal::AUGRUSequence>(
                       node)) {
            return augru_seq->get_clip() == 0.0f &&
                augru_seq->get_activations() == std::vector<std::string>{"sigmoid", "tanh"} &&
                !ov::op::util::is_seq_len_provided(augru_seq->get_input_node_shared_ptr(2),
                                                   max_seq_len);
        } else if (const auto &lstm_seq = std::dynamic_pointer_cast<const ov::opset6::LSTMSequence>(
                       node)) {
            return lstm_seq->get_clip() == 0.0f &&
                lstm_seq->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"} &&
                !ov::op::util::is_seq_len_provided(lstm_seq->get_input_node_shared_ptr(3),
                                                   max_seq_len);
        }
        return false;
    };

    pass_config->set_callback<ov::pass::ConvertRNNSequenceToTensorIterator,
                              ov::pass::ConvertGRUSequenceToTensorIterator,
                              ov::pass::ConvertLSTMSequenceToTensorIterator>(
                                  [isSequencePrimitiveSupported](const_node_ptr &node) -> bool {
                                      return isSequencePrimitiveSupported(node);
                                  });

    pass_config->set_callback<ov::pass::RNNCellDecomposition, ov::pass::GRUCellDecomposition,
                              ov::pass::LSTMCellDecomposition>(
                                  [isCellPrimitiveSupported](const_node_ptr &node) -> bool {
                                      return isCellPrimitiveSupported(node);
                                  });

    pass_config->set_callback<ov::pass::MVN6Decomposition>(
        [](const_node_ptr &node) -> bool {
            std::string errorMessage;
            return node::MVN::isSupportedOperation(node, errorMessage);
        });

    pass_config->set_callback<ov::pass::NormalizeL2Decomposition>(
        [](const_node_ptr &node) -> bool {
            std::string errorMsg;
            return node::NormalizeL2::isSupportedOperation(node, errorMsg);
        });

    pass_config->enable<ov::pass::SoftmaxDecomposition>();
    pass_config->set_callback<ov::pass::SoftmaxDecomposition>(
            [](const_node_ptr &node) -> bool {
                return node->input_value(0).get_partial_shape().rank().get_length() <= 5;
            });

    if (!isLegacyApi) {
        auto nmsCallback = [](const_node_ptr &node) -> bool {
            for (size_t i = 0; i < node->get_output_size(); i++) {
                const auto outputs = node->get_output_target_inputs(i);
                for (const auto &out : outputs) {
                    if (!ov::op::util::is_output(out.get_node())) {
                        return false;
                    }
                }
            }
            return true;
        };

        pass_config->set_callback<ov::pass::ConvertNMS9ToNMSIEInternal>(nmsCallback);
        pass_config->set_callback<ov::pass::ConvertMulticlassNmsToMulticlassNmsIE>(nmsCallback);
        pass_config->set_callback<ov::pass::ConvertMatrixNmsToMatrixNmsIE>(nmsCallback);
    }

    // List of enabled/disabled transformations

    // Allow FP16 Converts to be folded and FP16 constants to be upgraded to FP32 data type
    pass_config->disable<ov::pass::DisableDecompressionConvertConstantFolding>();
    pass_config->disable<ov::pass::ConvertCompressedOnlyToLegacy>();
    pass_config->disable<ov::pass::EyeDecomposition>();

    pass_config->disable<ov::pass::ConvertGELU>();
    pass_config->disable<ov::pass::ConvertShuffleChannels3>();
    pass_config->disable<ov::pass::Gelu7Downgrade>();
    pass_config->disable<ov::pass::HSwishDecomposition>();
    pass_config->disable<ov::pass::ReduceL1Decomposition>();
    pass_config->disable<ov::pass::ReduceL2Decomposition>();
    pass_config->disable<ov::pass::SoftPlusDecomposition>();
    pass_config->disable<ov::pass::HSigmoidDecomposition>();
    pass_config->disable<ov::pass::ConvertMod>();
    pass_config->disable<ov::pass::ConvertShuffleChannels3>();
    pass_config->disable<ov::pass::WeightsDequantizeToFakeQuantize>();
    pass_config->disable<ov::pass::SimplifyCTCGreedyDecoderSeqLen>();
    pass_config->disable<ov::pass::ConvertGather7ToGather1>();
    pass_config->disable<ov::pass::ConvertGather8ToGather7>();
    pass_config->disable<ov::pass::ConvertMinimum>();
    pass_config->disable<ov::pass::ConvertBroadcastToTiles>();
    pass_config->disable<ov::pass::ConvertReduceMeanToPooling>();
    pass_config->disable<ov::pass::ConvertReduceMaxToPooling>();
    pass_config->disable<ov::pass::ConvertReduceSumToPooling>();
    pass_config->disable<ov::pass::SliceToStridedSlice>();
    pass_config->disable<ov::pass::ConvertDetectionOutput8ToDetectionOutput1>();
    pass_config->disable<ov::pass::ConvertROIAlign9To3>();
    pass_config->disable<ov::pass::SoftSignDecomposition>();
    pass_config->disable<ov::pass::UniqueDecomposition>();

    pass_config->enable<ov::pass::NormalizeL2Decomposition>();
    pass_config->enable<ov::pass::ConvertInterpolate1ToInterpolate4>();
    pass_config->enable<ov::pass::ConvertGather1ToGather7>();
    pass_config->enable<ov::pass::ConvertDetectionOutput1ToDetectionOutput8>();
    pass_config->enable<ov::pass::ConvertROIAlign3To9>();

    if (useLpt) {
        CPU_LPT_SCOPE(LowPrecisionTransformations_Part3);
        pass_config->set_callback<ov::pass::AddFakeQuantizeFusion,
                                  ov::pass::MulFakeQuantizeFusion,
                                  ov::pass::FakeQuantizeMulFusion>(
                                      [](const_node_ptr &node) -> bool {
                                          std::string errMsg;
                                          return !node::FakeQuantize::isSupportedOperation(node, errMsg);
                                      });

        pass_config->set_callback<ov::pass::ConvertQuantizeDequantize>([&defaultPrecisions](const_node_ptr &node) -> bool {
            return ngraph::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForMultiply(node, defaultPrecisions);
        });
    }

    manager.run_passes(model);
}

void Transformations::Lpt(const bool hasINT16orINT32Levels, const std::vector<ov::element::Type>& defaultPrecisions) {
    CPU_DEBUG_CAP_TRANSFORMATION_SCOPE(this, Lpt);

    using namespace ngraph::pass::low_precision;
    CPU_LPT_SCOPE(LowPrecisionTransformations_Part4);
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "LowPrecisionTransformations");
    //Only enable conv/group conv signed input on AMX platform.
    std::vector<ov::element::Type> input0LowPrecisionList;
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx)) {
        input0LowPrecisionList = {ov::element::u8, ov::element::i8};
    } else {
        input0LowPrecisionList = {ov::element::u8};
    }
    auto supportedPrecisions = std::vector<PrecisionsRestriction>({
            PrecisionsRestriction::create<ov::opset1::Convolution>({
                    {{0}, input0LowPrecisionList},
                    {{1}, {ov::element::i8}},
                }),
            PrecisionsRestriction::create<ov::opset1::ConvolutionBackpropData>({
                    {{0}, {ov::element::u8, ov::element::i8}},
                    {{1}, {ov::element::i8}}
                }),
            PrecisionsRestriction::create<ov::opset1::GroupConvolution>({
                    {{0}, input0LowPrecisionList},
                    {{1}, {ov::element::i8}}
                }),
            PrecisionsRestriction::create<ov::opset1::Multiply>({
                    {{0}, {ov::element::u8}},
                    {{1}, {ov::element::i8}},
                }),
            PrecisionsRestriction::create<ov::opset1::MatMul>({
                    {{0}, {ov::element::u8, ov::element::i8}},
                    {{1}, {ov::element::i8}}
                }),
            PrecisionsRestriction::create<ov::opset5::LSTMSequence>({
                    {{0, 1}, {ov::element::u8, ov::element::i8}},
                }),
            PrecisionsRestriction::create<ov::opset6::GRUSequence>({
                    {{0, 1}, {ov::element::u8, ov::element::i8}},
                }),
        });

    auto quantizationRestrictions = std::vector<QuantizationGranularityRestriction>({
            QuantizationGranularityRestriction::create<ov::opset1::Convolution>({0}),
            QuantizationGranularityRestriction::create<ov::opset1::ConvolutionBackpropData>({0})
        });

    // for GNA networks reference execution
    bool updatePrecision = true;
    if (hasINT16orINT32Levels) {
        updatePrecision = false;
        supportedPrecisions = std::vector<PrecisionsRestriction>({});
    }

    ov::pass::Manager lptManager;
    lptManager.register_pass<ngraph::pass::low_precision::LowPrecision>(
        supportedPrecisions,
        quantizationRestrictions,
        LayerTransformation::Params(updatePrecision, ov::element::f32, defaultPrecisions));
    lptManager.get_pass_config()->set_callback<ngraph::pass::low_precision::MarkupPrecisions>([](const_node_ptr& node) -> bool {
        if (const auto mulitply = std::dynamic_pointer_cast<const ov::opset1::Multiply>(node)) {
            return !MultiplyToGroupConvolutionTransformation::canBeTransformedToGroupConvolution(mulitply);
        }
        return false;
    });
    lptManager.get_pass_config()->set_callback<ngraph::pass::low_precision::ConvolutionBackpropDataTransformation>(
        [&defaultPrecisions](const_node_ptr& node) -> bool {
            return LayerTransformation::isAsymmetricQuantization(node, defaultPrecisions) ||
                WeightableLayerTransformation::isAsymmetricOnWeights(node, defaultPrecisions);
        });

    lptManager.get_pass_config()->disable<ngraph::pass::low_precision::MultiplyToGroupConvolutionTransformation>();

    lptManager.run_passes(model);
}

void Transformations::PostLpt() {
    CPU_DEBUG_CAP_TRANSFORMATION_SCOPE(this, PostLpt);

    ov::pass::Manager postLPTPassManager;
    postLPTPassManager.set_per_pass_validation(false);
    postLPTPassManager.register_pass<ov::pass::UnrollTensorIterator>();
    postLPTPassManager.register_pass<ov::pass::ReshapePRelu>();
    postLPTPassManager.get_pass_config()->set_callback<ov::pass::UnrollTensorIterator>([](const_node_ptr &node) -> bool {
        // UnrollTI transformation is disabled by default, is turned on by LowLatency transformation
        return node->get_rt_info().count("UNROLL_TI") == 0;
    });
    postLPTPassManager.register_pass<MoveEltwiseUpThroughDataMov>();
    postLPTPassManager.get_pass_config()->set_callback<MoveEltwiseUpThroughDataMov>([](const std::shared_ptr<const ov::Node>& node) -> bool {
        if (node->get_input_size() >= 2) {
            return node->get_input_element_type(1) == ov::element::i8 || node->get_input_element_type(1) == ov::element::u8;
        }
        return false;
    });

    postLPTPassManager.register_pass<ov::pass::ConstantFolding>();

    // Snippets may brake MHA patterns so the fusion has to performed before
    postLPTPassManager.register_pass<MHAFusion>();
    postLPTPassManager.register_pass<FuseFQtoInteraction>();
    postLPTPassManager.get_pass_config()->set_callback<MHAFloatFusion, MHAFloatFusion2,
                                                       MHAQuantFusion, MHAQuantFusion2>
        ([this](const std::shared_ptr<const ov::Node>& n) -> bool {
            std::string errorMessage;

            if (!node::MHA::isSupportedOperation(n, errorMessage))
                return true;

            // Implementation calls AMX BF16 brgemm only for tensors with K and N aligned on 2, otherwise fallbacks on vector impl
            // Vector madd BF16 instruction on SPR has reduced performance on HW level, which results in overall perf degradation
            size_t bf16Factor = 2;
            if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16_amx_bf16) &&
                (n->get_input_element_type(0) == element::bf16 || (n->get_input_element_type(0) == element::f32 && enableBF16)) &&
                (n->get_input_shape(0)[3] % bf16Factor != 0 || n->get_input_shape(1)[1] % bf16Factor != 0 || n->get_input_shape(3)[3] % bf16Factor != 0)) {
                return true;
            }

            return false;
        });

    // Float MHA is supported by snippets now
    if (!enableBF16) {
        postLPTPassManager.get_pass_config()->disable<MHAFloatFusion>();
        postLPTPassManager.get_pass_config()->disable<MHAFloatFusion2>();
    }

    // Execute before snippets. Otherwise FQ will be converted to Subgraph
    postLPTPassManager.register_pass<ConvertFqRnnToQuantizedRnn>();
    postLPTPassManager.run_passes(model);
}

void Transformations::MainSnippets(void) {
    if (snippetsMode == Config::SnippetsMode::Disable ||
        !dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) // snippets are implemeted only for relevant platforms (avx2+ extentions)
        return;

    ngraph::pass::Manager snippetsManager;
    snippetsManager.set_per_pass_validation(false);
    if (snippetsMode != Config::SnippetsMode::IgnoreCallback)
        snippetsManager.register_pass<SnippetsMarkSkipped>(enableBF16);
    snippetsManager.register_pass<ngraph::snippets::pass::SnippetsTokenization>();

    const bool isMHASupported =
            !enableBF16 &&  // TODO: Need to add BF16 support for MHA in Snippets
            dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core);  // MHA has BRGEMM that is supported only on AVX512 platforms
    if (!isMHASupported) {
        snippetsManager.get_pass_config()->disable<ngraph::snippets::pass::TokenizeMHASnippets>();
    }
    if (snippetsMode != Config::SnippetsMode::IgnoreCallback) {
        snippetsManager.get_pass_config()->set_callback<ngraph::snippets::pass::TokenizeMHASnippets>(
                [](const std::shared_ptr<const ov::Node>& n) -> bool {
                    const auto pshape = n->get_output_partial_shape(0);
                    const auto shape = pshape.get_shape();
                    const auto parallel_work_amount =
                            std::accumulate(shape.rbegin() + 2, shape.rend(), 1, std::multiplies<size_t>());
                    const auto kernel_buffer_size =
                            std::accumulate(shape.rbegin(), shape.rbegin() + 2, 1, std::multiplies<size_t>()) *
                            n->get_output_element_type(0).size();
                    // Heuristic values:
                    //    parallelism work amount - not enough work amount for parallelism
                    //    kernel work amount - large shape for kernel execution, not cache-local
                    // TODO: The heuristics will be removed after
                    //       - loop blocking support on code generation level
                    //       - parallelism support on JIT level
                    const auto needed_num_of_threads = 12lu;
                    const auto l2_cache_size = dnnl::utils::get_cache_size(2, true);
                    const auto is_unsupported_parallel_work_amount = parallel_get_num_threads() / 2 > parallel_work_amount &&
                                                                     parallel_work_amount < needed_num_of_threads;
                    const auto is_unsupported_kernel_work_amount = kernel_buffer_size > l2_cache_size;
                    return is_unsupported_parallel_work_amount || is_unsupported_kernel_work_amount;
                });
        snippetsManager.get_pass_config()->set_callback<ngraph::snippets::pass::TokenizeSnippets>(
                [](const std::shared_ptr<const ov::Node>& n) -> bool {
                    // CPU Plugin support Swish in Subgraph via conversion to SwichCPU which assumes second input to be constant
                    const bool is_unsupported_swish =
                            ov::is_type<const ov::op::v4::Swish>(n) && n->inputs().size() > 1 &&
                            !ov::is_type<const ov::op::v0::Constant>(n->get_input_node_shared_ptr(1));
                    // todo: general tokenization flow is not currently supported for these operations.
                    //  they can be tokenized only as a part of complex patterns
                    const bool is_disabled_tokenization = (ov::is_type<const ov::op::v1::Softmax>(n) ||
                                                           ov::is_type<const ov::op::v8::Softmax>(n) ||
                                                           ov::is_type<const ov::op::v0::MatMul>(n) ||
                                                           ov::is_type<const ov::op::v1::Transpose>(n) ||
                                                           ov::is_type<const ov::op::v1::Broadcast>(n) ||
                                                           ov::is_type<const ov::op::v3::Broadcast>(n));
                    const auto& inputs = n->inputs();
                    // todo: clarify whether we can evaluate snippets on const paths
                    const bool has_only_const_inputs = std::all_of(inputs.begin(), inputs.end(),
                                                                   [](const ov::Input<const ov::Node>& in) {
                                                                       return ov::is_type<ov::op::v0::Constant>(
                                                                               in.get_source_output().get_node_shared_ptr());
                                                                   });
                    // todo: clarify whether we can evaluate snippets on inputs with larger ranks
                    auto rank_is_too_large = [](const ov::descriptor::Tensor& t) {
                        // callback is called has_supported_in_out(), so it's safe to assume that the shapes are static
                        return t.get_partial_shape().rank().get_length() > 6;
                    };
                    const bool bad_input_rank = std::any_of(inputs.begin(), inputs.end(),
                                                            [&](const ov::Input<const ov::Node>& in) {
                                                                return rank_is_too_large(in.get_tensor());
                                                            });
                    const auto& outputs = n->outputs();
                    const bool bad_output_rank = std::any_of(outputs.begin(), outputs.end(),
                                                             [&](const ov::Output<const ov::Node>& out) {
                                                                 return rank_is_too_large(out.get_tensor());
                                                             });
                    return has_only_const_inputs || bad_input_rank || bad_output_rank || is_unsupported_swish ||
                           is_disabled_tokenization;
                });
    }
    snippetsManager.run_passes(model);
}

void Transformations::PostSnippets(void) {
    ov::pass::Manager postSnippetsManager;
    postSnippetsManager.set_per_pass_validation(false);
    postSnippetsManager.register_pass<ov::pass::FakeQuantizeDecomposition>();
    postSnippetsManager.get_pass_config()->set_callback<ov::pass::FakeQuantizeDecomposition>([](const_node_ptr& node) -> bool {
        std::string errMsg;
        return node::FakeQuantize::isSupportedOperation(node, errMsg);
    });
    postSnippetsManager.register_pass<ov::pass::ConstantFolding>();
    postSnippetsManager.run_passes(model);
}

void Transformations::Snippets(void) {
    CPU_DEBUG_CAP_TRANSFORMATION_SCOPE(this, Snippets);

    MainSnippets();
    PostSnippets();
}

}   // namespace intel_cpu
}   // namespace ov
