// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "intel_gpu/plugin/transformations_pipeline.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "low_precision/convolution.hpp"
#include "low_precision/convolution_backprop_data.hpp"
#include "low_precision/group_convolution.hpp"
#include "low_precision/low_precision.hpp"
#include "low_precision/mat_mul.hpp"
#include "low_precision/multiply_to_group_convolution.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/pull_reshape_through_dequantization.hpp"
#include "low_precision/pull_transpose_through_dequantization.hpp"
#include "low_precision/recurrent_cell.hpp"
#include "low_precision/strided_slice.hpp"
#include "openvino/core/deprecated.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/gru_cell.hpp"
#include "openvino/op/gru_sequence.hpp"
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/rnn_cell.hpp"
#include "openvino/op/rnn_sequence.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "plugin/transformations/binary_conv_to_conv.hpp"
#include "plugin/transformations/clamp_fp16_output.hpp"
#include "plugin/transformations/convert_fc_to_compressed.hpp"
#include "plugin/transformations/convert_gather_to_compressed.hpp"
#include "plugin/transformations/convert_matmul_to_fc.hpp"
#include "plugin/transformations/fc_convert_fusion.hpp"
#include "plugin/transformations/kv_cache_fusion.hpp"
#include "plugin/transformations/move_convert_after_gather.hpp"
#include "plugin/transformations/move_fc_reshape_to_weights.hpp"
#include "plugin/transformations/rms_fusion.hpp"
#include "plugin/transformations/swiglu_fusion.hpp"
#include "plugin/transformations/transpose_matmul_fusion.hpp"
#include "plugin/transformations/indirect_kv_cache.hpp"
#include "transformations/common_optimizations/broadcast_elementwise_fusion.hpp"
#include "transformations/common_optimizations/broadcast_transition.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/common_optimizations/convert_quantize_dequantize.hpp"
#include "transformations/common_optimizations/lin_op_sequence_fusion.hpp"
#include "transformations/common_optimizations/lstm_cell_fusion.hpp"
#include "transformations/common_optimizations/mvn_fusion.hpp"
#include "transformations/common_optimizations/softmax_fusion.hpp"
#include "transformations/common_optimizations/transpose_sinking.hpp"
#include "transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp"
#include "transformations/common_optimizations/wrap_interpolate_into_transposes.hpp"
#include "transformations/control_flow/unroll_tensor_iterator.hpp"
#include "transformations/convert_pooling_to_reduce.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/convert_shapeof.hpp"
#include "transformations/decompose_reduce_for_false_keepdims.hpp"
#include "transformations/einsum_decomposition.hpp"
#include "transformations/fp16_compression/convert_compression_only_to_legacy.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/low_precision/mark_dequantization_subgraph.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_batch_to_space.hpp"
#include "transformations/op_conversions/convert_broadcast3.hpp"
#include "transformations/op_conversions/convert_deformable_conv_v8_to_v1.hpp"
#include "transformations/op_conversions/convert_depth_to_space.hpp"
#include "transformations/op_conversions/convert_gather_0d.hpp"
#include "transformations/op_conversions/convert_gather_downgrade.hpp"
#include "transformations/op_conversions/convert_gelu.hpp"
#include "transformations/op_conversions/convert_gp9_to_gp_ie_internal.hpp"
#include "transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp"
#include "transformations/op_conversions/convert_matrix_nms_to_matrix_nms_ie.hpp"
#include "transformations/op_conversions/convert_mod.hpp"
#include "transformations/op_conversions/convert_multiclass_nms_to_multiclass_nms_ie.hpp"
#include "transformations/op_conversions/convert_nms9_to_nms_ie_internal.hpp"
#include "transformations/op_conversions/convert_nms_rotated_to_nms_ie_internal.hpp"
#include "transformations/op_conversions/convert_pad12_downgrade.hpp"
#include "transformations/op_conversions/convert_pad_to_group_conv.hpp"
#include "transformations/op_conversions/convert_previous_nms_to_nms_9.hpp"
#include "transformations/op_conversions/convert_prior_box_v8_to_v0.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"
#include "transformations/op_conversions/convert_reduce_to_reshape.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"
#include "transformations/op_conversions/convert_shapeof3.hpp"
#include "transformations/op_conversions/convert_shuffle_channels3.hpp"
#include "transformations/op_conversions/convert_softmax_downgrade.hpp"
#include "transformations/op_conversions/convert_space_to_batch.hpp"
#include "transformations/op_conversions/convert_space_to_depth.hpp"
#include "transformations/op_conversions/convert_subtract.hpp"
#include "transformations/op_conversions/convert_ti_to_sequences.hpp"
#include "transformations/op_conversions/convert_topk11_downgrade.hpp"
#include "transformations/op_conversions/eye_decomposition.hpp"
#include "transformations/op_conversions/gelu7_downgrade.hpp"
#include "transformations/op_conversions/gru_cell_decomposition.hpp"
#include "transformations/op_conversions/hsigmoid_decomposition.hpp"
#include "transformations/op_conversions/hswish_decomposition.hpp"
#include "transformations/op_conversions/log_softmax_decomposition.hpp"
#include "transformations/op_conversions/lstm_cell_decomposition.hpp"
#include "transformations/op_conversions/mvn6_decomposition.hpp"
#include "transformations/op_conversions/normalize_l2_decomposition.hpp"
#include "transformations/op_conversions/reduce_l1_decomposition.hpp"
#include "transformations/op_conversions/reduce_l2_decomposition.hpp"
#include "transformations/op_conversions/rnn_cell_decomposition.hpp"
#include "transformations/op_conversions/simplify_ctc_greedy_decoder_seq_len.hpp"
#include "transformations/op_conversions/softmax_decomposition.hpp"
#include "transformations/op_conversions/softplus_decomposition.hpp"
#include "transformations/opset_conversions/convert_opset2_to_opset1.hpp"
#include "transformations/opset_conversions/convert_opset3_to_opset2.hpp"
#include "transformations/resolve_names_collisions.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/smart_reshape/matmul_sr.hpp"

namespace {
template<typename T>
static bool disable_reduce_decomposition(const std::shared_ptr<const ov::Node> node) {
    if (auto op = std::dynamic_pointer_cast<const T>(node)) {
        if (op->input(0).get_partial_shape()[0].is_static()) {
            bool fp16_batch_not_1 = op->get_element_type() == ov::element::f16 && op->input(0).get_partial_shape()[0] != 1;
            return !fp16_batch_not_1;
        }
    }
    return false;
}

static bool is_non_supported_decompression_op(const std::shared_ptr<const ov::Node> node) {
    auto get_single_consumer = [](const std::shared_ptr<const ov::Node> node) -> std::shared_ptr<ov::Node> {
        const auto consumers = node->get_output_target_inputs(0);
        if (consumers.size() != 1)
            return nullptr;
        return consumers.begin()->get_node()->shared_from_this();
    };

    auto consumer = get_single_consumer(node);
    if (!consumer)
        return true;

    if (ov::is_type<ov::opset1::MatMul>(consumer) || ov::is_type<ov::op::v8::Gather>(consumer)) {
        return false;
    } else if (ov::is_type<ov::opset1::Reshape>(consumer)) {
        consumer = get_single_consumer(consumer);
        if (consumer != nullptr && (ov::is_type<ov::opset1::MatMul>(consumer) || ov::is_type<ov::op::v8::Gather>(consumer))) {
            return false;
        }
    }
    if (consumer != nullptr && ov::is_type<ov::opset1::Convert>(consumer)) {
        consumer = get_single_consumer(consumer);
        if (consumer != nullptr && (ov::is_type<ov::opset1::MatMul>(consumer) || ov::is_type<ov::op::v8::Gather>(consumer))) {
            return false;
        }
    }
    return true;
}
}  // namespace

namespace ov {
namespace intel_gpu {

void TransformationsPipeline::apply(std::shared_ptr<ov::Model> func) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "TransformationsPipeline::apply");
    using const_node_ptr = const std::shared_ptr<const ov::Node>;

    const auto& defaultPrecisions = ov::pass::low_precision::precision_set::get_int8_support();
    bool enableInt8;
    bool unroll_loop = config.get_property(ov::intel_gpu::enable_loop_unrolling);
    {
        ov::pass::Manager manager;
        auto pass_config = manager.get_pass_config();
        manager.set_per_pass_validation(false);

        // Temporary solution, global rt info cleanup is needed
        for (auto& node : func->get_ops()) {
            ov::enable_constant_folding(node);
            ov::disable_keep_const_precision(node);
        }

        auto is_model_quantized = ov::pass::low_precision::LowPrecision::isFunctionQuantized(func);
        enableInt8 = config.get_property(ov::intel_gpu::enable_lp_transformations) && is_model_quantized;
        if (enableInt8) {
            manager.register_pass<ov::pass::MarkDequantizationSubgraph>(
                std::vector<ov::element::Type>{ ov::element::i8, ov::element::u8, ov::element::i4, ov::element::u4 });
        }

        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<EinsumDecomposition>();

        precisions_map fp_convert_precision_map = {
                {ov::element::f64, ov::element::f32}
        };

        // call conversion of float types with keep_precision_sensitive_in_fp32 = true
        auto fp_precision_supported = [&](ov::element::Type e) -> bool {
            switch (e) {
                case ov::element::f16: return device_info.supports_fp16;
                case ov::element::f32: return true; // assume that all GPUs support f32 data type
                case ov::element::f64: return device_info.supports_fp64;
                case ov::element::bf16: return false;
                default: return false;
            }
            return false;
        };

        const auto fallback_precision = ov::element::f32;
        std::vector<ov::element::Type> fp_element_types = {
                ov::element::f32,
                ov::element::f16,
                ov::element::bf16
        };

        // Add conversion from FP data types to infer precision if it's specified
        auto infer_precision = config.get_property(ov::hint::inference_precision);
        if (infer_precision != ov::element::undefined) {
            if (!fp_precision_supported(infer_precision))
                infer_precision = fallback_precision;

            for (auto& et : fp_element_types) {
                if (et != infer_precision) {
                    fp_convert_precision_map.insert({et, infer_precision});
                }
            }
        }

        // Add conversion from unsupported FP data types to f32 if we don't have a conversion to something valid already in the list
        for (auto& et : fp_element_types) {
            if (!fp_precision_supported(et)) {
                bool has_valid_conversion = fp_convert_precision_map.count(et) && fp_precision_supported(fp_convert_precision_map[et]);
                if (!has_valid_conversion) {
                    fp_convert_precision_map.insert(std::make_pair(et, fallback_precision));
                }
            }
        }

        type_to_fuse_map empty_fuse_map = {};
        manager.register_pass<ov::pass::Validate>();

        // fuse softmax, MVN patterns, so that they will not be marked as precision sensitive in ConvertPrecision
        manager.register_pass<ov::pass::SoftmaxFusion>();
        manager.register_pass<ov::pass::MVNFusion>();
        // decompose MVNs that sre not supported in GPU, so that they will be marked as precision sensitive in ConvertPrecision
        manager.register_pass<ov::pass::MVN6Decomposition>();
        // Run these broadcast optimizations earlier to ensure that those are executed before NopElimination/ConstantFolding
        manager.register_pass<ov::pass::BroadcastElementwiseFusion>();
        manager.register_pass<ov::pass::BroadcastTransition>();

        manager.register_pass<ov::pass::KeepConstantsPrecisionAndAddConverts>();
        pass_config->set_callback<ov::pass::KeepConstantsPrecisionAndAddConverts>(
            [](const_node_ptr& node) -> bool {
                auto next_node = node->get_output_target_inputs(0).begin()->get_node();
                if (is_type<ov::op::v0::Convert>(next_node)) {
                    next_node = next_node->get_output_target_inputs(0).begin()->get_node();
                }
                return !is_type<ov::op::v0::MatMul>(next_node);
            });

        manager.register_pass<ov::pass::MarkDequantizationSubgraph>(ov::element::TypeVector{ov::element::u8, ov::element::u4, ov::element::i4}, true);
        // Need to check if transfomrations work correctly for mixed models with both compression and quantization at the same time.
        if (!is_model_quantized)
            pass_config->set_callback<ov::pass::MarkDequantizationSubgraph>(is_non_supported_decompression_op);

        manager.register_pass<ov::intel_gpu::MoveConvertAfterGather>();

        const bool keep_precision_sensitive_in_fp32_1 = true;
        const bool convert_input_output_precision = false;
        const bool store_original_precision_as_rt_attribute = true;
        manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map,
                                                          empty_fuse_map,
                                                          keep_precision_sensitive_in_fp32_1,
                                                          convert_input_output_precision,
                                                          store_original_precision_as_rt_attribute);

        manager.register_pass<ov::pass::CommonOptimizations>();

        manager.register_pass<ov::pass::WrapInterpolateIntoTransposes>();
        manager.register_pass<ov::pass::TransposeSinking>();

        if (!unroll_loop) {
            manager.register_pass<ov::pass::BidirectionalLSTMSequenceDecomposition>();
            manager.register_pass<ov::pass::BidirectionalGRUSequenceDecomposition>();
            manager.register_pass<ov::pass::BidirectionalRNNSequenceDecomposition>();
        }

        manager.register_pass<ov::intel_gpu::ConvertBinaryConvolutionToConvolution>();
        manager.register_pass<ov::pass::ConvertSequenceToTensorIterator>();
        manager.register_pass<ov::pass::ConvertOpSet3ToOpSet2>();
        manager.register_pass<ov::pass::ConvertOpSet2ToOpSet1>();

        manager.register_pass<ov::pass::LSTMCellDecomposition>();
        manager.register_pass<ov::pass::GRUCellDecomposition>();
        manager.register_pass<ov::pass::RNNCellDecomposition>();

        if (unroll_loop) {
            manager.register_pass<ov::pass::BidirectionalLSTMSequenceDecomposition>();
            manager.register_pass<ov::pass::BidirectionalGRUSequenceDecomposition>();
            manager.register_pass<ov::pass::BidirectionalRNNSequenceDecomposition>();
        }

        manager.register_pass<ConvertShapeOf1To3>();
        manager.register_pass<ov::pass::ConvertNMS1ToNMS9>();
        manager.register_pass<ov::pass::ConvertNMS3ToNMS9>();
        manager.register_pass<ov::pass::ConvertNMS4ToNMS9>();
        manager.register_pass<ov::pass::ConvertNMS5ToNMS9>();
        manager.register_pass<ov::pass::ConvertNMS9ToNMSIEInternal>();
        manager.register_pass<ov::pass::ConvertNMSRotatedToNMSIEInternal>();
        manager.register_pass<ov::pass::ConvertGP9ToGPIEInternal>();
        manager.register_pass<ov::pass::ConvertMatrixNmsToMatrixNmsIE>();
        manager.register_pass<ov::pass::ConvertGather0D>();
        manager.register_pass<ov::pass::ConvertPriorBox8To0, false>();
        manager.register_pass<ov::pass::ConvertMulticlassNmsToMulticlassNmsIE>();
        manager.register_pass<ov::pass::TransposeMatMul>();
        manager.register_pass<ov::pass::ConvertPad12ToPad1, false>();

        precisions_map int_convert_precision_map {
                {ov::element::i64, ov::element::i32},
                {ov::element::u64, ov::element::i32},
                {ov::element::u16, ov::element::i32},
                {ov::element::u32, ov::element::i32},
                {ov::element::boolean, ov::element::u8},
                {ov::element::i4, ov::element::i8},
                {ov::element::u4, ov::element::u8},
        };

        manager.register_pass<ov::pass::Validate>();
        const bool keep_precision_sensitive_in_fp32_2 = true;
        manager.register_pass<ov::pass::ConvertPrecision>(int_convert_precision_map,
                                                          empty_fuse_map,
                                                          keep_precision_sensitive_in_fp32_2,
                                                          convert_input_output_precision);

        pass_config->disable<ov::pass::EyeDecomposition>();

        // disable conversion to legacy and use the new mixed precision
        // in which precision sensitive nodes are kept in FP32
        pass_config->disable<ov::pass::ConvertCompressedOnlyToLegacy>();

        // SpaceToDepth/DepthToSpace node implementation supports only equal input/output tensors with rank <= 5
        pass_config->set_callback<ov::pass::ConvertSpaceToDepth,
                                  ov::pass::ConvertDepthToSpace>(
                [](const_node_ptr &node) -> bool {
                    return node->input_value(0).get_partial_shape().size() <= 5lu &&
                        node->input_value(0).get_partial_shape().size() == node->get_output_partial_shape(0).size();
                });

        pass_config->set_callback<ov::pass::ConvertBatchToSpace,
                                  ov::pass::ConvertSpaceToBatch>(
                [](const_node_ptr &node) -> bool {
                    const auto& rank = node->input(0).get_partial_shape().rank().get_length();
                    return rank <= 5;
                });

        // Convert reduce to reshape expected to be optimized out
        manager.register_pass<ov::pass::ConvertReduceToReshape>();

        if (device_info.supports_immad) {
            // oneDNN reduction is used
            pass_config->disable<ov::pass::ConvertReduceSumToPooling>();
            pass_config->disable<ov::pass::ConvertReduceMeanToPooling>();
            pass_config->disable<ov::pass::ConvertReduceMaxToPooling>();
            manager.register_pass<ConvertAvgPoolingToReduce>();
            manager.register_pass<DecomposeReduceForFalseKeepDims>();
        } else {
            pass_config->set_callback<ov::pass::ConvertReduceSumToPooling>(
            [](const_node_ptr &node) -> bool {
                return disable_reduce_decomposition<ov::op::v1::ReduceSum>(node);
            });

            pass_config->set_callback<ov::pass::ConvertReduceMeanToPooling>(
            [](const_node_ptr &node) -> bool {
                return disable_reduce_decomposition<ov::op::v1::ReduceMean>(node);
            });

            pass_config->set_callback<ov::pass::ConvertReduceMaxToPooling>(
            [](const_node_ptr &node) -> bool {
                return disable_reduce_decomposition<ov::op::v1::ReduceMax>(node);
            });
        }

        auto isCellPrimitiveSupported = [](const_node_ptr &node) -> bool {
            if (std::dynamic_pointer_cast<const ov::op::v0::RNNCell>(node)) {
                return false;
            } else if (std::dynamic_pointer_cast<const ov::op::v3::GRUCell>(node)) {
                return false;
            } else if (const auto &lstm_cell = std::dynamic_pointer_cast<const ov::op::v4::LSTMCell>(node)) {
                return lstm_cell->get_clip() == 0.0f && lstm_cell->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"};
            } else if (const auto &lstm_cell_v1 = std::dynamic_pointer_cast<const ov::op::v0::LSTMCell>(node)) {
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
            if (std::dynamic_pointer_cast<const ov::op::v5::RNNSequence>(node)) {
                return false;
            } else if (std::dynamic_pointer_cast<const ov::op::v5::GRUSequence>(node)) {
                return false;
            } else if (const auto &lstm_seq = std::dynamic_pointer_cast<const ov::op::v5::LSTMSequence>(node)) {
                return lstm_seq->get_clip() == 0.0f &&
                       lstm_seq->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"} &&
                       max_seq_len < 16 &&
                       !ov::op::util::is_seq_len_provided(lstm_seq->get_input_node_shared_ptr(0),
                                                          lstm_seq->get_input_node_shared_ptr(3));
            }
            return false;
        };

        pass_config->set_callback<ov::pass::RNNCellDecomposition,
                                  ov::pass::GRUCellDecomposition,
                                  ov::pass::LSTMCellDecomposition>(
            [isCellPrimitiveSupported](const_node_ptr &node) -> bool {
                return isCellPrimitiveSupported(node);
            });

        pass_config->set_callback<ov::pass::LSTMCellFusion>(
            [isCellPrimitiveSupported](const_node_ptr &node) -> bool {
                return !isCellPrimitiveSupported(node);
            });

        if (unroll_loop) {
            pass_config->set_callback<ov::pass::ConvertRNNSequenceToTensorIterator,
                    ov::pass::ConvertGRUSequenceToTensorIterator,
                    ov::pass::ConvertLSTMSequenceToTensorIterator>(
                    [isSequencePrimitiveSupported](const_node_ptr &node) -> bool {
                        return isSequencePrimitiveSupported(node);
                    });
        }

        pass_config->set_callback<ov::pass::ConvertLoopToLSTMSequence,
                                  ov::pass::FuseReverseLSTMSequence,
                                  ov::pass::FuseLSTMSequencesToBidirectionalLSTMSequence>(
                [isSequencePrimitiveSupported](const_node_ptr &node) -> bool {
                    return !isSequencePrimitiveSupported(node);
                });

        pass_config->set_callback<ov::pass::MVN6Decomposition>(
            [](const_node_ptr &node) -> bool {
                const auto mvn = std::dynamic_pointer_cast<const ov::op::v6::MVN>(node);
                if (mvn != nullptr && node->get_input_size() == 2) {
                    if (auto axes_node = dynamic_cast<ov::op::v0::Constant*>(mvn->get_input_node_ptr(1))) {
                        auto mvn_axes = axes_node->cast_vector<int64_t>();
                        auto out_rank = mvn->get_output_partial_shape(0).size();
                        ov::util::normalize_axes(mvn.get(), out_rank, mvn_axes);

                        std::sort(mvn_axes.begin(), mvn_axes.end());

                        // Supported cases:
                        // 2 <= out_rank <= 5
                        // axes set: [out_rank - 1, out_rank - 2, ... r] where r > 1
                        // basically impl supports cases when tensor can be reshaped to [d1, d2]
                        // so that d2 is set of dimensions for normalization

                        // Skip unsupported ranks
                        if (out_rank == 1 || out_rank > 5)
                            return false;

                        // check axes set
                        for (size_t i = 0; i < mvn_axes.size(); i++) {
                            auto axis = mvn_axes[mvn_axes.size() - i - 1];
                            if (axis != static_cast<int64_t>(out_rank - i - 1) || axis == 0) {
                                  return false;
                            }
                        }
                        return true;
                    }
                }
                return false;
            });

        pass_config->enable<ov::pass::NormalizeL2Decomposition>();
        pass_config->set_callback<ov::pass::NormalizeL2Decomposition>(
            [](const_node_ptr &node) -> bool {
            // Condition to filter out axes such as [0, 1, 2] which is not supported currently.
            const auto norm = ov::as_type_ptr<const ov::op::v0::NormalizeL2>(node);
            const auto inputRank = norm->get_input_partial_shape(0).size();
            auto axesNode = ov::as_type_ptr<const ov::op::v0::Constant>(norm->get_input_node_shared_ptr(1));
            const auto axes = axesNode->cast_vector<size_t>();
            const auto isSupportedAxes = [](const std::vector<size_t> &axes, const size_t inputRank) {
                if (axes.size() == 1 && axes[0] == 1) {
                    return true;
                } else if (axes.size() == inputRank - 1) {
                    auto sortAxes = axes;
                    std::sort(sortAxes.begin(), sortAxes.end());
                    for (size_t i = 0; i < sortAxes.size(); i++) {
                        if (sortAxes[i] != i + 1)
                            return false;
                    }
                    return true;
                }
                return false;
            };

            if (!isSupportedAxes(axes, inputRank) && ov::shape_size(axesNode->get_shape()) != 0) {
                return false;
            }
            return true;
            });

        pass_config->enable<ov::pass::SoftmaxDecomposition>();
        pass_config->set_callback<ov::pass::SoftmaxDecomposition>(
            [](const_node_ptr &node) -> bool {
                return node->input_value(0).get_partial_shape().rank().get_length() <= 5;
            });

        // List of enabled/disabled transformations
        pass_config->disable<ov::pass::ConvertGELU>();
        pass_config->disable<ov::pass::Gelu7Downgrade>();
        pass_config->disable<ov::pass::ConvertMod>();
        pass_config->disable<ov::pass::ConvertShuffleChannels3>();
        pass_config->disable<ov::pass::HSwishDecomposition>();
        pass_config->disable<ov::pass::HSigmoidDecomposition>();
        pass_config->disable<ov::pass::ReduceL1Decomposition>();
        pass_config->disable<ov::pass::ReduceL2Decomposition>();
        pass_config->disable<ov::pass::SoftPlusDecomposition>();
        pass_config->disable<ov::pass::LogSoftmaxDecomposition>();
        pass_config->disable<ov::pass::ConvertBroadcast3>();
        pass_config->disable<ov::pass::WeightsDequantizeToFakeQuantize>();
        pass_config->disable<ov::pass::SimplifyCTCGreedyDecoderSeqLen>();
        pass_config->disable<ov::pass::ConvertSoftMax8ToSoftMax1>();
        pass_config->disable<ov::pass::ConvertShapeOf3>();
        pass_config->disable<ov::pass::ConvertGather8ToGather7>();
        pass_config->disable<ov::pass::ConvertGather7ToGather1>();
        pass_config->disable<ov::pass::ConvertTopK11ToTopK3>();

        pass_config->enable<ov::pass::ConvertInterpolate1ToInterpolate4>();

        if (enableInt8) {
            pass_config->set_callback<ov::pass::ConvertQuantizeDequantize>([&](const_node_ptr &node) -> bool {
                return ov::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForMultiply(node, defaultPrecisions);
            });
        }

        manager.run_passes(func);
    }

    if (enableInt8) {
        OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "TransformationsPipeline::apply::lpt");
        using namespace ov::pass::low_precision;

        auto supportedPrecisions = std::vector<PrecisionsRestriction>({
            PrecisionsRestriction::create<ov::op::v1::Convolution>({
                {{0}, {ov::element::u8, ov::element::i8}},
                {{1}, {ov::element::i8}},
            }),
            PrecisionsRestriction::create<ov::op::v1::ConvolutionBackpropData>({
                {{0}, {ov::element::u8, ov::element::i8}},
                {{1}, {ov::element::i8}}
            }),
            PrecisionsRestriction::create<ov::op::v1::GroupConvolution>({
                {{0}, {ov::element::u8, ov::element::i8}},
                {{1}, {ov::element::i8}}
            }),
            PrecisionsRestriction::create<ov::op::v5::LSTMSequence>(PrecisionsRestriction::PrecisionsByPorts{}),
            PrecisionsRestriction::create<ov::op::v5::GRUSequence>(PrecisionsRestriction::PrecisionsByPorts{})
        });

        auto perTensorQuantization = std::vector<QuantizationGranularityRestriction>({
            QuantizationGranularityRestriction::create<ov::op::v1::Convolution>({0}),
            QuantizationGranularityRestriction::create<ov::op::v1::ConvolutionBackpropData>({0}),
        });

        ov::pass::Manager lptManager;

        auto lptPassConfig = lptManager.get_pass_config();
        // quantized LSTMSequence / GPUSequence are not supported yet. Avoid extra transformation
        lptPassConfig->disable<ov::pass::low_precision::RecurrentCellTransformation>();
        lptPassConfig->set_callback<ov::pass::low_precision::MarkupPrecisions>([](const_node_ptr& node) -> bool {
            if (const auto mulitply = std::dynamic_pointer_cast<const ov::op::v1::Multiply>(node)) {
                return !MultiplyToGroupConvolutionTransformation::canBeTransformedToGroupConvolution(mulitply);
            }
            return false;
        });
        lptPassConfig->set_callback<ConvolutionBackpropDataTransformation>([func, defaultPrecisions](const_node_ptr& node) -> bool {
            auto fillStaticChannel = [func](const ov::PartialShape& shape, size_t& channel) -> bool {
                const auto rank = shape.rank();
                if (rank.is_dynamic()) {
                    return false;
                }
                if (rank.get_length() < 2l) {
                    return false;
                }
                const auto& dimension = shape[1];
                if (dimension.is_dynamic()) {
                    return false;
                }
                channel = dimension.get_length();
                return true;
            };

            size_t inputChannels = 0;
            if (!fillStaticChannel(node->get_input_partial_shape(0), inputChannels)) {
                return true;
            }

            size_t outputChannels = 0;
            if (!fillStaticChannel(node->get_output_partial_shape(0), outputChannels)) {
                return true;
            }


            if ((inputChannels % 4 != 0) || (outputChannels % 16 != 0)) {
                return true;
            }

            return LayerTransformation::isAsymmetricQuantization(node, defaultPrecisions)
                || WeightableLayerTransformation::isAsymmetricOnWeights(node, defaultPrecisions);
        });

        lptPassConfig->set_callback<MultiplyToGroupConvolutionTransformation>([&](const_node_ptr& node) -> bool {
            // disable MultiplyToGroupConvolution if Multiply with Constant can be fused

            const auto dequantization = NetworkHelper::getDequantization(node, defaultPrecisions, 0, true);
            std::shared_ptr<ov::Node> parent = dequantization.empty() ? nullptr : dequantization.data.get_node()->shared_from_this();
            if (parent == nullptr) {
                const auto constantNode = NetworkHelper::getConstantInput(node);
                const auto constant = constantNode == nullptr ? nullptr : ov::as_type_ptr<ov::op::v0::Constant>(constantNode);
                if (constant != nullptr) {
                    auto parent = node->get_input_node_shared_ptr(0);
                    if (parent == constant) {
                        parent = node->get_input_node_shared_ptr(1);
                    }
                }
            }

            if (parent != nullptr) {
                const auto parentHasOneConsumer = parent->get_output_target_inputs(0).size() == 1ul;
                if (parentHasOneConsumer) {
                    return true;
                }
            }

            // disable MultiplyToGroupConvolution for Multiply with scalar

            if (MultiplyToGroupConvolutionTransformation::isDynamicOrScalar(node)) {
                return true;
            }

            return false;
        });

        bool reshapeIgnorePerTensorQuantizationCheck = false;
        if (device_info.supports_immad) // Disable reshape transform until onednn i8 fc is optimized
            reshapeIgnorePerTensorQuantizationCheck = true;
        auto params = LayerTransformation::Params(true, element::f32, defaultPrecisions, reshapeIgnorePerTensorQuantizationCheck);
        lptManager.register_pass<LowPrecision>(supportedPrecisions, perTensorQuantization, params);
        lptManager.run_passes(func);
    }

    {
        OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "TransformationsPipeline::apply::run_passes");
        ov::pass::Manager manager;
        // This ConstantFolding pass is added to fold reshapes added for constant inputs on NMS internal operation which prevents upper-bound calculation
        // TODO: check why we have these reshapes
        manager.register_pass<ov::pass::ConstantFolding>();

        manager.register_pass<ov::pass::UnrollTensorIterator>();
        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ov::pass::UnrollTensorIterator>(
            [unroll_loop](const std::shared_ptr<const ov::Node> &node) -> bool {
                auto sub_graph_op = std::dynamic_pointer_cast<const ov::op::util::SubGraphOp>(node);
                int64_t num_iter = sub_graph_op->get_num_iterations();
                if (!unroll_loop)
                    return num_iter != 1;
                return num_iter >= 16;
            });

        manager.run_passes(func);
    }

    {
        ov::pass::Manager manager;
        manager.register_pass<ov::intel_gpu::ClampFP16Output>();
        manager.register_pass<ov::intel_gpu::ConvertMatMulToFullyConnected>();
        manager.register_pass<ov::intel_gpu::MoveFCReshapeToWeights>();
        manager.register_pass<ov::intel_gpu::ConvertFullyConnectedToFullyConnectedCompressed>();
        manager.register_pass<ov::intel_gpu::ConvertGatherToGatherCompressed>();
        manager.register_pass<ov::intel_gpu::RMSFusion>(device_info.max_work_group_size);
        manager.register_pass<ov::intel_gpu::KVCacheFusion>();
        manager.register_pass<ov::intel_gpu::FullyConnectedConvertFusion>();
        if (!device_info.supports_immad)
            manager.register_pass<ov::intel_gpu::TransposeMatMulFusion>();
        manager.register_pass<ov::intel_gpu::SwiGLUFusion>();

        manager.register_pass<ov::intel_gpu::IndirectKVCache>();
        // This is supposed to be the last pass to ensure that we don't have name collisions until
        // GPU plugin stops using friendly names for program creation
        manager.register_pass<ov::pass::ResolveNameCollisions>(true);

        manager.run_passes(func);
    }
}
}  // namespace intel_gpu
}  // namespace ov
