// Copyright (C) 2018-2025 Intel Corporation
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

#include "openvino/opsets/opset10.hpp"
#include "intel_gpu/plugin/transformations_pipeline.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "low_precision/add.hpp"
#include "low_precision/concat.hpp"
#include "low_precision/convolution.hpp"
#include "low_precision/convolution_backprop_data.hpp"
#include "low_precision/fold_convert.hpp"
#include "low_precision/fuse_convert.hpp"
#include "low_precision/group_convolution.hpp"
#include "low_precision/low_precision.hpp"
#include "low_precision/mat_mul.hpp"
#include "low_precision/multiply_to_group_convolution.hpp"
#include "low_precision/mvn.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/pull_reshape_through_dequantization.hpp"
#include "low_precision/pull_transpose_through_dequantization.hpp"
#include "low_precision/recurrent_cell.hpp"
#include "low_precision/rt_info/bias_attribute.hpp"
#include "low_precision/strided_slice.hpp"
#include "low_precision/transpose.hpp"
#include "openvino/core/deprecated.hpp"
#include "openvino/core/type/element_type.hpp"
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
#include "openvino/op/reshape.hpp"
#include "openvino/op/rnn_cell.hpp"
#include "openvino/op/rnn_sequence.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "plugin/transformations/binary_conv_to_conv.hpp"
#include "plugin/transformations/clamp_fp16_output.hpp"
#include "plugin/transformations/convert_fc_to_compressed.hpp"
#include "plugin/transformations/convert_matmul_to_fc.hpp"
#include "plugin/transformations/convert_stridedslices_to_variadicsplit.hpp"
#include "plugin/transformations/decompose_reduce_scalar_output.hpp"
#include "plugin/transformations/fc_convert_fusion.hpp"
#include "plugin/transformations/fc_horizontal_fusion.hpp"
#include "plugin/transformations/kv_cache_fusion.hpp"
#include "plugin/transformations/move_fc_reshape_to_weights.hpp"
#include "plugin/transformations/bcast_and_pad_zp_buffers.hpp"
#include "plugin/transformations/print_model_statistics.hpp"
#include "plugin/transformations/fc_per_layer_scaling.hpp"
#include "plugin/transformations/transpose_fusion.hpp"
#include "plugin/transformations/indirect_kv_cache.hpp"
#include "plugin/transformations/kv_cache_compression.hpp"
#include "plugin/transformations/convert_convolution.hpp"
#include "plugin/transformations/unsqueeze_broadcast_reshape_matmul_fusion.hpp"
#include "plugin/transformations/unsqueeze_broadcast_reshape_sdpa_fusion.hpp"
#include "plugin/transformations/increase_position_ids_precision.hpp"
#include "plugin/transformations/dynamic_quantize_fully_connected.hpp"
#include "plugin/transformations/optimize_subsequent_reshapes.hpp"
#include "plugin/transformations/lora_horizontal_fusion.hpp"
#include "plugin/transformations/sink_reshape.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/rms_fusion.hpp"
#include "transformations/common_optimizations/broadcast_elementwise_fusion.hpp"
#include "transformations/common_optimizations/broadcast_transition.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/common_optimizations/convert_quantize_dequantize.hpp"
#include "transformations/common_optimizations/group_normalization_fusion.hpp"
#include "transformations/common_optimizations/lin_op_sequence_fusion.hpp"
#include "transformations/common_optimizations/lstm_cell_fusion.hpp"
#include "transformations/common_optimizations/move_eltwise_up_data_movement.hpp"
#include "transformations/common_optimizations/mvn_fusion.hpp"
#include "transformations/common_optimizations/sdpa_scale_fusion.hpp"
#include "transformations/common_optimizations/activations_scaling.hpp"
#include "transformations/common_optimizations/softmax_fusion.hpp"
#include "transformations/common_optimizations/glu_fusion.hpp"
#include "transformations/common_optimizations/shared_ops_optimization.hpp"
#include "transformations/common_optimizations/transpose_sinking.hpp"
#include "transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp"
#include "transformations/common_optimizations/wrap_interpolate_into_transposes.hpp"
#include "transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp"
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
#include "transformations/op_conversions/convert_divide.hpp"
#include "transformations/op_conversions/convert_gather_0d.hpp"
#include "transformations/op_conversions/convert_gather_downgrade.hpp"
#include "transformations/op_conversions/convert_gelu.hpp"
#include "transformations/op_conversions/convert_gather_to_compressed.hpp"
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
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"
#include "transformations/op_conversions/group_normalization_decomposition.hpp"
#include "transformations/opset_conversions/convert_opset2_to_opset1.hpp"
#include "transformations/opset_conversions/convert_opset3_to_opset2.hpp"
#include "transformations/resolve_names_collisions.hpp"
#include "transformations/rt_info/dequantization_node.hpp"
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

static bool is_decompression_multiply(const std::shared_ptr<const ov::Node> node, bool supports_immad) {
    std::vector<ov::DiscreteTypeInfo> target_consumers = { ov::opset1::MatMul::get_type_info_static(),
                                                           ov::op::v8::Gather::get_type_info_static(),
                                                           ov::op::v1::Convolution::get_type_info_static(),
                                                           ov::opset1::Convolution::get_type_info_static(),
                                                           ov::op::v1::ConvolutionBackpropData::get_type_info_static(),
                                                           ov::opset1::ConvolutionBackpropData::get_type_info_static(),
                                                           ov::opset1::GroupConvolution::get_type_info_static() };

    std::vector<ov::DiscreteTypeInfo> convolutions = { ov::op::v1::Convolution::get_type_info_static(),
                                                       ov::opset1::Convolution::get_type_info_static(),
                                                       ov::op::v1::ConvolutionBackpropData::get_type_info_static(),
                                                       ov::opset1::ConvolutionBackpropData::get_type_info_static(),
                                                       ov::opset1::GroupConvolution::get_type_info_static() };

    auto all_has_types = [](const std::set<ov::Input<ov::Node>>& consumers, const std::vector<ov::DiscreteTypeInfo>& types) {
        return std::all_of(consumers.begin(), consumers.end(), [&types](const ov::Input<ov::Node>& input) {
            return cldnn::one_of(input.get_node()->get_type_info(), types);
        });
    };

    const auto consumers = node->get_output_target_inputs(0);

    for (const auto& consumer : consumers) {
        const auto& type_info = consumer.get_node()->get_type_info();
        if (cldnn::one_of(type_info, target_consumers)) {
            if (cldnn::one_of(type_info, convolutions) && consumer.get_node()->input_value(0).get_partial_shape().is_dynamic()) {
                return false;
            }
            return true;
        }
    }

    auto are_multiply_from_decompression = [&](const ov::Input<ov::Node> consumer) {
        if (!cldnn::one_of(consumer.get_node()->get_type_info(), { ov::op::v1::Multiply::get_type_info_static() }))
            return false;
        const auto child_consumers = consumer.get_node()->get_output_target_inputs(0);

        for (const auto& child_consumer : child_consumers) {
            const auto& type_info = child_consumer.get_node()->get_type_info();
            if (cldnn::one_of(type_info, target_consumers)) {
                if (cldnn::one_of(type_info, convolutions) && child_consumer.get_node()->input_value(0).get_partial_shape().is_dynamic()) {
                    return false;
                }
                return true;
            }
        }
        return false;
    };

    auto are_converts_from_decompression = [&](const std::set<ov::Input<ov::Node>>& consumers) {
        if (!all_has_types(consumers, { ov::opset1::Convert::get_type_info_static() }))
            return false;
        for (const auto& consumer : consumers) {
            const auto child_consumers = consumer.get_node()->get_output_target_inputs(0);
            for (const auto& child_consumer : child_consumers) {
                const auto& type_info = child_consumer.get_node()->get_type_info();
                if (cldnn::one_of(type_info, target_consumers)) {
                    if (cldnn::one_of(type_info, convolutions) && child_consumer.get_node()->input_value(0).get_partial_shape().is_dynamic()) {
                        return false;
                    }
                    return true;
                }
                if (are_multiply_from_decompression(child_consumer)) {
                    continue;
                }
                return false;
            }
        }
        return true;
    };

    if (all_has_types(consumers, { ov::opset1::Reshape::get_type_info_static() })) {
        for (const auto& consumer : consumers) {
            const auto child_consumers = consumer.get_node()->get_output_target_inputs(0);
            for (const auto& child_consumer : child_consumers) {
                const auto& type_info = child_consumer.get_node()->get_type_info();
                if (cldnn::one_of(type_info, target_consumers)) {
                    if (cldnn::one_of(type_info, convolutions) && child_consumer.get_node()->input_value(0).get_partial_shape().is_dynamic()) {
                        return false;
                    }
                    return true;
                } else if (are_converts_from_decompression(child_consumers)) {
                    return true;
                }
            }
        }
    }
    return are_converts_from_decompression(consumers);
}
}  // namespace

namespace cldnn {
extern bool query_microkernels_supported(cldnn::engine& e, const cldnn::ExecutionConfig& config);
}  // namespace cldnn

namespace ov::intel_gpu {

bool TransformationsPipeline::fuse_type_to_convert(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    auto convert = ov::as_type_ptr<ov::opset10::Convert>(node);
    if (!convert)
        return false;
    const auto& from = node->get_output_element_type(0);
    auto it = precisions.find(from);
    if (it == precisions.end())
        return false;
    const auto& to = it->second;

    if (convert->get_convert_element_type() == ov::element::boolean && to.is_integral_number()) {
        // For Convert node, converting precision from numerical data types to boolean will lead to mathematical
        // error, because here the output precision boolean is replaced by u8:
        //  - floating point value 0.01 is converted to be 1 for boolean, but 0 for u8 - need to insert Ceil.
        //  - either float or int values should be clipped with the interval [0; 1] to mimic bool cast behavior, i.e.
        //  0 - is false, 1 - is true
        //  - to perform clamping correctly an Abs op should be inserted before Clamp
        // Thus an Abs, Ceil and Clamp nodes should be added before the Convert node for this scenario.
        ov::pass::NodeRegistry reg;
        const auto& in_prec = convert->get_input_element_type(0);
        auto parent_node = convert->input_value(0).get_node_shared_ptr();
        auto item = precisions.find(in_prec);
        if (item != precisions.end()) {
            // Add convert node for unsupported precision, such as FP64 or INT64
            parent_node = reg.make<ov::opset10::Convert>(parent_node, item->second);
        }
        if (in_prec.is_signed()) {
            parent_node = reg.make<ov::opset10::Abs>(parent_node);
        }
        if (in_prec.is_real()) {
            parent_node = reg.make<ov::opset10::Ceiling>(parent_node);
        }
        parent_node = reg.make<ov::opset10::Clamp>(parent_node, 0, 1);
        const auto new_convert = reg.make<ov::opset10::Convert>(parent_node, to);
        new_convert->set_friendly_name(convert->get_friendly_name());
        ov::copy_runtime_info(convert, reg.get());
        ov::replace_node(convert, new_convert);
        return true;
    }
    convert->set_convert_element_type(to);
    return true;
}

void TransformationsPipeline::apply(std::shared_ptr<ov::Model> func) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "TransformationsPipeline::apply");
    using const_node_ptr = const std::shared_ptr<const ov::Node>;

    const auto& defaultPrecisions = ov::pass::low_precision::precision_set::get_int8_support();
    const ov::element::TypeVector supported_woq_types = {ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4};
    bool enableInt8;
    ov::element::Type infer_precision = ov::element::undefined;
    bool unroll_loop = config.get_property(ov::intel_gpu::enable_loop_unrolling);
    {
        ov::pass::Manager manager("Plugin:GPU");
        auto pass_config = manager.get_pass_config();
        manager.set_per_pass_validation(false);

        // Temporary solution, global rt info cleanup is needed
        for (auto& node : func->get_ops()) {
            ov::enable_constant_folding(node);
            ov::disable_keep_const_precision(node);
        }

        auto is_model_quantized = ov::pass::low_precision::LowPrecision::isFunctionQuantized(func);
        enableInt8 = config.get_property(ov::intel_gpu::enable_lp_transformations) && is_model_quantized;

        manager.register_pass<ov::pass::MarkDequantization>(
            std::vector<ov::element::Type>{ ov::element::i8, ov::element::u8, ov::element::i4, ov::element::u4 },
            !device_info.supports_immad);

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
        infer_precision = config.get_property(ov::hint::inference_precision);
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
        // fuse following ops into GroupNormalization:
        // group_norm_gamma * (instance_norm_gamma * MVN(x) + instance_norm_beta) + group_norm_beta
        // note that instance norm related parameters are optional:
        // - instance_norm_gamma is assumed to be filled with ones if not present in the graph
        // - instance_norm_beta is assumed to be filled with zeros if not present in the graph
        manager.register_pass<ov::pass::GroupNormalizationFusion>();
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

        // Disable subtract folding only for the dGPUs to meet the requirements of oneDNN:
        // it expects to have the same data type for weights and zero points (apply it only for u8 data type, since other compression
        // types are not supported by oneDNN)
        manager.register_pass<ov::pass::KeepConstPrecision>(supported_woq_types, !device_info.supports_immad);
        pass_config->set_callback<ov::pass::MarkDequantization,
                ov::pass::KeepConstPrecision>([&](const std::shared_ptr<const ov::Node> node) {
            return !is_decompression_multiply(node, device_info.supports_immad);
        });

        pass_config->set_callback<ov::pass::RMSFusion>([=](const_node_ptr& root) -> bool {
            if (!root->get_input_partial_shape(0).is_static()) {
                return false;
            }
            const auto& gamma_shape = root->get_input_partial_shape(0).to_shape();
            const int32_t vec_size = 8;
            return static_cast<int32_t>((gamma_shape.back() / vec_size)) > static_cast<int32_t>(device_info.max_work_group_size);
        });
        manager.register_pass<ov::pass::RMSFusion>(false);

        const bool keep_precision_sensitive_in_fp32_1 = true;
        const bool convert_input_output_precision = false;
        const bool store_original_precision_as_rt_attribute = true;

        manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map,
                                                          empty_fuse_map,
                                                          keep_precision_sensitive_in_fp32_1,
                                                          convert_input_output_precision,
                                                          store_original_precision_as_rt_attribute);

        manager.register_pass<ov::pass::CommonOptimizations>();

        pass_config->set_callback<ov::pass::ScaledDotProductAttentionDecomposition>([&](const std::shared_ptr<const ov::Node> node){
            GPU_DEBUG_IF(cldnn::debug_configuration::get_instance()->enable_sdpa != -1) {
                GPU_DEBUG_CODE(return cldnn::debug_configuration::get_instance()->enable_sdpa == 1);
            }

            if (!config.get_property(ov::intel_gpu::hint::enable_sdpa_optimization))
                return false;

            auto sdpa = ov::as_type_ptr<const ov::op::v13::ScaledDotProductAttention>(node);
            const auto& query_ps = sdpa->get_input_partial_shape(0);
            const auto& key_ps = sdpa->get_input_partial_shape(1);
            const auto& value_ps = sdpa->get_input_partial_shape(2);

            // Known limitations:
            // - The data type of SDPA should be fp16
            if (sdpa->get_output_element_type(0) != ov::element::f16)
                return false;

            // - The number of dimensions for each input is expected to be 4
            if (query_ps.size() != 4 || key_ps.size() != 4 || value_ps.size() != 4) {
                return false;
            }

            // - The head size of all Q, K, and V inputs should be the same static value
            if (query_ps[query_ps.size() - 1].is_dynamic() || key_ps[key_ps.size() - 1].is_dynamic() || value_ps[value_ps.size() - 1].is_dynamic()) {
                return false;
            }

            if (query_ps[query_ps.size() - 1].get_length() != key_ps[key_ps.size() - 1].get_length() ||
                query_ps[query_ps.size() - 1].get_length() != value_ps[value_ps.size() - 1].get_length()) {
                return false;
            }

            const auto head_size = query_ps[query_ps.size() - 1].get_length();
            if (device_info.supports_immad && cldnn::query_microkernels_supported(m_context->get_engine(), config) && head_size <= 256)
                return true;

            // - Head size should be 128 for any model type; or should be in the range of 64 to 256 for stateful LLMs because of performance reasons.
            //   This limitations is recommended to prevent performance drop in models with small head size, such as SD,
            //   until the SDPA operation is optimized for these cases
            const auto optimal_subgroup_size = 16;
            bool valid_head_size = head_size % optimal_subgroup_size == 0;
            valid_head_size &= (head_size >= 64 && head_size <= 256);
            if (!valid_head_size) {
                return false;
            }

            return true;
        });

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
        manager.register_pass<DecomposeReduceForScalarOutput>();

        precisions_map int_convert_precision_map{
            {ov::element::i64, ov::element::i32},
            {ov::element::u64, ov::element::i32},
            {ov::element::i16, ov::element::i32},
            {ov::element::u16, ov::element::i32},
            {ov::element::u32, ov::element::i32},
            {ov::element::boolean, ov::element::u8},
            {ov::element::i4, ov::element::i8},
            {ov::element::u4, ov::element::u8},
        };

        manager.register_pass<ov::pass::Validate>();
        const bool keep_precision_sensitive_in_fp32_2 = true;

        // To convert to f16 input to boolean which is converted to u8, add abs + ceiling + clamp before convert.
        type_to_fuse_map type_to_fuse = {{ov::opset10::Convert::get_type_info_static(), fuse_type_to_convert}};
        manager.register_pass<ov::pass::ConvertPrecision>(int_convert_precision_map,
                                                          type_to_fuse,
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
            if (ov::as_type_ptr<const ov::op::v0::RNNCell>(node)) {
                return false;
            } else if (ov::as_type_ptr<const ov::op::v3::GRUCell>(node)) {
                return false;
            } else if (const auto &lstm_cell = ov::as_type_ptr<const ov::op::v4::LSTMCell>(node)) {
                return false;
            } else if (const auto &lstm_cell_v1 = ov::as_type_ptr<const ov::op::v0::LSTMCell>(node)) {
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
            auto max_seq_len = data_pshape[1];
            if (data_pshape.rank().is_static() && data_pshape.rank().get_length() > 1 && !data_pshape[1].is_static())
                return false;
            if (ov::as_type_ptr<const ov::op::v5::RNNSequence>(node)) {
                return false;
            } else if (ov::as_type_ptr<const ov::op::v5::GRUSequence>(node)) {
                return false;
            } else if (const auto &lstm_seq = ov::as_type_ptr<const ov::op::v5::LSTMSequence>(node)) {
                return lstm_seq->get_clip() == 0.0f &&
                       lstm_seq->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"} &&
                       max_seq_len != 1 &&
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
                const auto mvn = ov::as_type_ptr<const ov::op::v6::MVN>(node);
                if (mvn != nullptr && node->get_input_size() == 2) {
                    if (auto axes_node = ov::as_type<ov::op::v0::Constant>(mvn->get_input_node_ptr(1))) {
                        auto mvn_axes = axes_node->cast_vector<int64_t>();
                        auto out_rank = mvn->get_output_partial_shape(0).size();
                        ov::util::try_normalize_axes(mvn_axes, out_rank, *mvn);

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
            [&](const_node_ptr &node) -> bool {
                OPENVINO_ASSERT(node->input_value(0).get_partial_shape().rank().is_static(),
                    node->get_friendly_name() + " has dynamic rank!");
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
        pass_config->disable<ov::pass::GroupNormalizationDecomposition>();

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

        ov::pass::Manager lptManager("GPU:LPT");

        auto lptPassConfig = lptManager.get_pass_config();
        // quantized LSTMSequence / GPUSequence are not supported yet. Avoid extra transformation
        lptPassConfig->disable<ov::pass::low_precision::RecurrentCellTransformation>();
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

        lptPassConfig->set_callback<TransposeTransformation>([&](const_node_ptr& node) -> bool {
            for (auto& user : node->get_users()) {
                if (ov::is_type<ov::op::v13::ScaledDotProductAttention>(user))
                    return true;
            }

            return false;
        });

        lptPassConfig->set_callback<MarkupPrecisions>([](const_node_ptr& node) -> bool {
            return ov::is_type<ov::opset1::Multiply>(node) && !MultiplyToGroupConvolutionTransformation::canBeTransformedToGroupConvolution(node);
        });

        lptPassConfig->set_callback<FoldConvertTransformation>([&](const_node_ptr& node) -> bool {
            const auto& consumers = node->get_output_target_inputs(0);
            if (consumers.size() == 1) {
                const auto consumer = consumers.begin()->get_node()->shared_from_this();
                return ov::is_type<ov::opset1::Multiply>(consumer) && is_decompression_multiply(consumer, device_info.supports_immad);
            }
            return false;
        });
        lptPassConfig->set_callback<FuseConvertTransformation>([&](const_node_ptr& node) -> bool {
            if (ov::is_type<ov::opset1::Multiply>(node)) {
                return ov::is_type<ov::opset1::Multiply>(node) && is_decompression_multiply(node, device_info.supports_immad);
            } else if (ov::is_type<ov::opset1::Subtract>(node)) {
                const auto& consumers = node->get_output_target_inputs(0);
                if (consumers.size() == 1) {
                    const auto consumer = consumers.begin()->get_node()->shared_from_this();
                    return ov::is_type<ov::opset1::Multiply>(consumer) && is_decompression_multiply(consumer, device_info.supports_immad);
                }
            }
            return false;
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
        ov::pass::Manager manager("GPU:UnrollTensorIterator");
        // This ConstantFolding pass is added to fold reshapes added for constant inputs on NMS internal operation which prevents upper-bound calculation
        // TODO: check why we have these reshapes
        manager.register_pass<ov::pass::ConstantFolding>();

        manager.register_pass<ov::pass::UnrollTensorIterator>();
        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ov::pass::UnrollTensorIterator>(
            [unroll_loop](const std::shared_ptr<const ov::Node> &node) -> bool {
                auto sub_graph_op = ov::as_type_ptr<const ov::op::util::SubGraphOp>(node);
                int64_t num_iter = sub_graph_op->get_num_iterations();
                if (!unroll_loop)
                    return num_iter != 1;
                return num_iter >= 16;
            });

        manager.run_passes(func);
    }

    {
        OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "TransformationsPipeline::apply::activations_scaling");
        ov::pass::Manager manager("GPU:ActivationsScaling");
        manager.set_per_pass_validation(false);
        auto pass_config = manager.get_pass_config();

        // Other ops support eltwise fusions
        const std::vector<DiscreteTypeInfo> allowed_data_movement_ops = {
            ov::op::v1::Reshape::get_type_info_static(),
            ov::op::v0::Squeeze::get_type_info_static(),
            ov::op::v0::Unsqueeze::get_type_info_static(),
            ov::op::v0::ShuffleChannels::get_type_info_static(),
            ov::op::v7::Roll::get_type_info_static(),
            ov::op::v0::ReverseSequence::get_type_info_static(),
            ov::op::v1::Broadcast::get_type_info_static(),
            ov::op::v3::Broadcast::get_type_info_static(),
        };
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMovScalar>(allowed_data_movement_ops);
        // FIXME (151111): this Validate is added as a workaround for resolving element
        // types after MoveEltwiseUpThroughDataMovScalar. It has to be removed
        // after 141764 is fixed as there's a clear issue with Validate passes
        // not working properly.
        manager.register_pass<ov::pass::Validate>();

        manager.register_pass<ov::pass::RoPEFusion>(true);
        pass_config->disable<ov::pass::RoPEFusionGPTJ>();
        pass_config->disable<ov::pass::RoPEFusionIOSlicing>();
        pass_config->disable<ov::pass::RoPEShareCosSin>();

        manager.register_pass<ov::intel_gpu::IncreasePositionIdsPrecision>();
        // This Validate is needed for proper data type propagation after applying IncreasePositionIdsPrecision pass
        manager.register_pass<ov::pass::Validate>();

        float activations_scale_factor = config.get_property(ov::hint::activations_scale_factor);

        if (activations_scale_factor > 0.f && infer_precision == ov::element::f16) {
            using namespace ov::pass::low_precision;

            auto supportedPrecisions = std::vector<PrecisionsRestriction>({});
            auto perTensorQuantization = std::vector<QuantizationGranularityRestriction>({});

            pass_config->disable<ov::pass::AddMultiplyFusion>();
            pass_config->disable<RecurrentCellTransformation>();
            pass_config->disable<MultiplyToGroupConvolutionTransformation>();
            pass_config->disable<ConvolutionTransformation>();
            pass_config->disable<ConvolutionBackpropDataTransformation>();
            pass_config->disable<GroupConvolutionTransformation>();
            pass_config->disable<MatMulTransformation>();
            pass_config->disable<MVNTransformation>();

            pass_config->set_callback<FoldConvertTransformation>(
                [](const std::shared_ptr<const ov::Node> &node) -> bool {
                    return ov::is_dequantization_node(node);
                });

            pass_config->set_callback<FuseConvertTransformation>(
                [](const std::shared_ptr<const ov::Node> &node) -> bool {
                    return (ov::is_dequantization_node(node) || ov::is_type<ov::opset1::FakeQuantize>(node));
                });

            manager.register_pass<ov::pass::activations_scaling::ScaleDownSingleLayer>(activations_scale_factor, infer_precision);
            manager.register_pass<ov::pass::SharedOpOptimization>();

            pass_config->set_callback<ov::pass::activations_scaling::ScaleDownSingleLayer>(
                [&infer_precision](const std::shared_ptr<const ov::Node> &node) -> bool {
                    return (node->input(0).get_element_type() != infer_precision);
                });

            // Move down scalar-multiply layers as much as possible
            auto params = LayerTransformation::Params(false, infer_precision, {infer_precision}, true, true);
            auto lpt_pass = manager.register_pass<LowPrecision>(supportedPrecisions, perTensorQuantization, params);
            lpt_pass->add_main<ov::pass::activations_scaling::EliminateScalarMul>();
            lpt_pass->add_main<ov::pass::activations_scaling::MoveDownScalarMul>();

            // Move up remained scalar-multiply layers
            manager.register_pass<ov::pass::EliminateEltwise>();
            manager.register_pass<ov::pass::activations_scaling::MulShareTransformation>();

            const std::vector<DiscreteTypeInfo> allowed_data_movement_ops = {
                ov::op::v1::Reshape::get_type_info_static(),
                ov::op::v1::Transpose::get_type_info_static(),
            };
            manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMovScalar>(allowed_data_movement_ops);
            manager.register_pass<ov::pass::Validate>();
        }

        manager.run_passes(func);
    }

    {
        ov::pass::Manager manager("GPU:PostLPT");
        manager.set_per_pass_validation(false);

        manager.register_pass<ov::intel_gpu::ClampFP16Output>();
        manager.register_pass<ov::intel_gpu::ConvertMatMulToFullyConnected>();
        manager.register_pass<ov::intel_gpu::MoveFCReshapeToWeights>();
        manager.register_pass<ov::intel_gpu::ConvertFullyConnectedToFullyConnectedCompressed>();

        bool disable_horizontal_fc_fusion = false;
        bool disable_fc_swiglu_fusion = false;
        GPU_DEBUG_GET_INSTANCE(debug_config);
        GPU_DEBUG_IF(debug_config->disable_horizontal_fc_fusion == 1)
            disable_horizontal_fc_fusion = true;
        GPU_DEBUG_IF(debug_config->disable_fc_swiglu_fusion == 1)
            disable_fc_swiglu_fusion = true;
        // mlp fusion is only supported for cldnn on high performant GPUis
        bool fuse_mlp_swiglu = !device_info.supports_immad &&
                               device_info.execution_units_count >= 128 &&
                               !disable_fc_swiglu_fusion;
        if (!disable_horizontal_fc_fusion) {
            manager.register_pass<ov::intel_gpu::FullyConnectedHorizontalFusion>(fuse_mlp_swiglu);
            // Temporary disabling for BMG due to regression
            if (device_info.arch != cldnn::gpu_arch::xe2) {
                manager.register_pass<ov::intel_gpu::LoRAHorizontalFusion>();
            }
        }

        // ZP should not be folded for FC. But still, ZP should be folded for Gather.
        // Therefore, run MarkDequantization again to fold ZP constant.
        manager.register_pass<ov::pass::MarkDequantization>(supported_woq_types, true);
        if (device_info.supports_immad) {
            if (disable_horizontal_fc_fusion)
                manager.register_pass<ov::pass::ConstantFolding>();
        }
        if (!disable_horizontal_fc_fusion)
            manager.register_pass<ov::pass::ConstantFolding>();

        manager.register_pass<ov::pass::SDPAScaleFusion>();
        manager.register_pass<ov::pass::ConvertGatherToGatherCompressed>();
        auto pass_config = manager.get_pass_config();
        manager.register_pass<ov::intel_gpu::KVCacheFusion>();
        manager.register_pass<ov::intel_gpu::FullyConnectedConvertFusion>();
        manager.register_pass<ov::intel_gpu::TransposeFusion>(device_info.supports_immad);

        if (!device_info.supports_immad) {
            manager.register_pass<ov::intel_gpu::UnsqueezeBroadcastReshapeMatmulFusion>();
        }
        manager.register_pass<ov::intel_gpu::UnsqueezeBroadcastReshapeSDPAFusion>();

        manager.register_pass<ov::pass::GLUFusion>();
        manager.register_pass<ov::intel_gpu::IndirectKVCache>();

        auto kv_cache_compression_dt = config.get_property(ov::hint::kv_cache_precision);
        manager.register_pass<ov::intel_gpu::KVCacheCompression>(kv_cache_compression_dt, device_info.supports_immad);

        manager.register_pass<ov::intel_gpu::ConvertConvolutionToInternal>();

        // This pass should be done after asymmetric quantization matching as it can move zp subtraction upper in the graph
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMovPerChannel>();

        manager.register_pass<ov::intel_gpu::ConvertStridedSlicesToVariadicSplit>();

        const size_t zp_pad_size = device_info.supports_immad ? 16 : 32;
        manager.register_pass<ov::intel_gpu::BroadcastAndPadZeroPointBuffers>(zp_pad_size, device_info.supports_immad);

        manager.register_pass<ov::intel_gpu::OptimizeSubsequentReshapes>();

        manager.register_pass<ov::intel_gpu::SinkReshape>();

        if (device_info.supports_immad) {
            auto dynamic_quantization_group_size = config.get_property(ov::hint::dynamic_quantization_group_size);
            pass_config->set_callback<ov::intel_gpu::DynamicQuantizeFullyConnected>([=](const_node_ptr& root) -> bool {
                for (size_t i = 0 ; i < root->get_input_node_shared_ptr(0)->get_output_size(); ++i) {
                    if (root->get_input_node_shared_ptr(0)->get_output_element_type(i) == ov::element::Type_t::f32) {
                        GPU_DEBUG_TRACE << root->get_friendly_name() << "  dyn_quan is turned off: input type is not supported" << std::endl;
                        return true;
                    }
                }

                auto weight_shape = root->get_input_partial_shape(1);
                const size_t innermost_size = weight_shape[weight_shape.size() - 1].get_length();
                if (innermost_size < 32) {
                    GPU_DEBUG_TRACE << root->get_friendly_name() << "  dyn_quan is turned off: shape is too small - " << innermost_size << std::endl;
                    return true;
                }

                // AZP does not support 8bit weight
                // XXX: This is currently wrapped as GPU_DEBUG_IF as dynamic_quantize_asym is not exposed through public API.
                GPU_DEBUG_IF(debug_config->dynamic_quantize_asym
                    && (root->get_input_element_type(1) == ov::element::i8 || root->get_input_element_type(1) == ov::element::u8)) {
                    GPU_DEBUG_TRACE << root->get_friendly_name() << "  dyn_quan is turned off: asym quantization does not support 8bit weight" << std::endl;
                    return true;
                }

                // AZP does not support grouped size dyn-quan
                GPU_DEBUG_IF(debug_config->dynamic_quantize_asym && (dynamic_quantization_group_size != UINT64_MAX)) {
                    GPU_DEBUG_TRACE << root->get_friendly_name() << "  dyn_quan is turned off: asym quantization does not support grouped quantization" <<
                                                                   " ('DynamicQuantizeAsym' is enabled with grouped size dyn-quan)" << std::endl;
                    return true;
                }

                bool has_wzp = root->get_input_size() > 4;
                if ((root->get_input_element_type(1) == ov::element::i8 || root->get_input_element_type(1) == ov::element::u8)
                    && has_wzp
                    && dynamic_quantization_group_size != UINT64_MAX) {
                    GPU_DEBUG_TRACE << root->get_friendly_name() << "  dyn_quan is turned off:"
                                                                    " asym 8bit weight does not support grouped quantization" << std::endl;
                    return true;
                }

                return false;
            });
            manager.register_pass<ov::intel_gpu::DynamicQuantizeFullyConnected>(dynamic_quantization_group_size);
        }

        // Remove Pad in front of MaxPool if both the pads_begin and pads_end are zero.
        manager.register_pass<ov::pass::EliminatePad>();

        // This is supposed to be the last pass to ensure that we don't have name collisions until
        // GPU plugin stops using friendly names for program creation
        manager.register_pass<ov::pass::ResolveNameCollisions>(true);
        GPU_DEBUG_IF(cldnn::debug_configuration::get_instance()->verbose >= 1) {
            manager.register_pass<ov::intel_gpu::PrintModelStatistics>();
        }
        manager.run_passes(func);
    }
}
}  // namespace ov::intel_gpu
