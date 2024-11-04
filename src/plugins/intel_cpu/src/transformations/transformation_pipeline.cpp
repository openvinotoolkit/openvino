// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformation_pipeline.h"
#include "defs.hpp"

// Operations
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset2.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/op/paged_attention.hpp"
#include <ov_ops/augru_cell.hpp>
#include <ov_ops/augru_sequence.hpp>
#include <ov_ops/gather_compressed.hpp>

// Common transformations
#include "transformations/common_optimizations/mark_precision_sensitive_shapeof_subgraphs.hpp"
#include "transformations/common_optimizations/add_fake_quantize_fusion.hpp"
#include "transformations/fp16_compression/convert_compression_only_to_legacy.hpp"
#include "transformations/common_optimizations/convert_quantize_dequantize.hpp"
#include "transformations/common_optimizations/lstm_cell_fusion.hpp"
#include "transformations/common_optimizations/fq_mul_fusion.hpp"
#include "transformations/common_optimizations/mul_fake_quantize_fusion.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/transpose_sinking.hpp"
#include "transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp"
#include "transformations/common_optimizations/augru_cell_fusion.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/common_optimizations/wrap_interpolate_into_transposes.hpp"
#include "transformations/common_optimizations/matmul_const_transposes_extraction.hpp"
#include "transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp"
#include "transformations/common_optimizations/move_eltwise_up_data_movement.hpp"
#include "transformations/common_optimizations/mark_rope_input_to_keep_in_mixed_precision.hpp"
#include "transformations/common_optimizations/rms_fusion.hpp"
#include "transformations/common_optimizations/lora_subgraph_fusion.hpp"
#include "transformations/control_flow/unroll_tensor_iterator.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/fp16_compression/mark_floatpoint_range.hpp"
#include "transformations/op_conversions/convert_avgpool_downgrade.hpp"
#include "transformations/op_conversions/convert_batch_to_space.hpp"
#include "transformations/op_conversions/convert_bitwise_to_logical_bool.hpp"
#include "transformations/op_conversions/convert_broadcast_to_tiles.hpp"
#include "transformations/op_conversions/convert_depth_to_space.hpp"
#include "transformations/op_conversions/convert_gather_downgrade.hpp"
#include "transformations/op_conversions/convert_gather_upgrade.hpp"
#include "transformations/op_conversions/convert_gather_to_compressed.hpp"
#include "transformations/op_conversions/convert_gelu.hpp"
#include "transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp"
#include "transformations/op_conversions/convert_matrix_nms_to_matrix_nms_ie.hpp"
#include "transformations/op_conversions/convert_maxpool_downgrade.hpp"
#include "transformations/op_conversions/convert_minimum_to_power_and_max.hpp"
#include "transformations/op_conversions/convert_mod.hpp"
#include "transformations/op_conversions/convert_multiclass_nms_to_multiclass_nms_ie.hpp"
#include "transformations/op_conversions/convert_nms9_to_nms_ie_internal.hpp"
#include "transformations/op_conversions/convert_previous_nms_to_nms_9.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"
#include "transformations/op_conversions/convert_roi_align_v3_to_v9.hpp"
#include "transformations/op_conversions/convert_roi_align_v9_to_v3.hpp"
#include "transformations/op_conversions/convert_scatter_nd_update15_downgrade.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"
#include "transformations/op_conversions/convert_shuffle_channels3.hpp"
#include "transformations/op_conversions/convert_slicescatter.hpp"
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
#include "transformations/op_conversions/hard_sigmoid_decomposition.hpp"
#include "transformations/op_conversions/hsigmoid_decomposition.hpp"
#include "transformations/op_conversions/hswish_decomposition.hpp"
#include "transformations/op_conversions/gru_cell_decomposition.hpp"
#include "transformations/op_conversions/lstm_cell_decomposition.hpp"
#include "transformations/op_conversions/group_normalization_decomposition.hpp"
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
#include "transformations/op_conversions/convert_topk3.hpp"
#include "transformations/op_conversions/convert_topk11_downgrade.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"
#include "transformations/opset_conversions/convert_opset2_to_opset1.hpp"
#include "transformations/opset_conversions/convert_opset3_to_opset2.hpp"
#include "transformations/smart_reshape/matmul_sr.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/transpose_sinking/ts_shape_of.hpp"
#include "utils/ngraph_transformation.hpp"
#include "transformations/utils/print_model.hpp"

// LPT transformations
#include "low_precision/add.hpp"
#include "low_precision/convert_subtract_constant.hpp"
#include "low_precision/convolution_backprop_data.hpp"
#include "low_precision/fold_convert.hpp"
#include "low_precision/fuse_convert.hpp"
#include "low_precision/group_convolution.hpp"
#include "low_precision/multiply_to_group_convolution.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/recurrent_cell.hpp"
#include "low_precision/rt_info/bias_attribute.hpp"
#include "transformations/low_precision/mark_dequantization_subgraph.hpp"

// CPU specific transformations
#include "transformations/cpu_opset/convert_to_cpu_specific_opset.hpp"
#if defined(OPENVINO_ARCH_ARM64)
#include "transformations/snippets/aarch64/pass/snippets_mark_skipped.hpp"
#else
#include "transformations/snippets/x64/pass/snippets_mark_skipped.hpp"
#endif
#include "transformations/cpu_opset/x64/pass/convert_to_interaction.hpp"
#include "transformations/cpu_opset/x64/pass/mlp_fusion.hpp"
#include "transformations/cpu_opset/x64/pass/qkv_proj_fusion.hpp"
#include "transformations/cpu_opset/arm/pass/convert_group_conv.hpp"
#include "transformations/cpu_opset/arm/pass/convert_group_conv1d.hpp"
#include "transformations/cpu_opset/arm/pass/convert_reduce_multi_axis.hpp"
#include "transformations/cpu_opset/arm/pass/mish_decomposition.hpp"
#include "transformations/cpu_opset/arm/pass/convert_reduce_no_keep_dims.hpp"
#include "transformations/cpu_opset/common/pass/decompose_integer_divide.hpp"
#include "transformations/cpu_opset/common/pass/decompose_rms_norm.hpp"
#include "transformations/cpu_opset/common/pass/convert_fq_rnn_to_quantized_rnn.hpp"
#include "transformations/cpu_opset/common/pass/insert_convert_after_extension.hpp"
#include "transformations/cpu_opset/common/pass/ngram_fusion.hpp"
#include "transformations/cpu_opset/common/pass/permute_slice_n_interpolation.hpp"
#include "transformations/cpu_opset/common/pass/swap_convert_transpose.hpp"
#include "transformations/cpu_opset/common/pass/causal_mask_preprocess_fusion.hpp"
#include "transformations/cpu_opset/common/pass/stateful_sdpa_fusion.hpp"
#include "transformations/cpu_opset/x64/pass/sdpa_fuse_transpose_reshape.hpp"

// Snippets
#include "snippets/pass/tokenization.hpp"
#include "snippets/pass/mha_tokenization.hpp"
#include "snippets/pass/fc_tokenization.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "snippets/pass/common_optimizations.hpp"
#include "snippets/pass/split_dimension_m.hpp"
#include "snippets/pass/extract_reshapes_from_mha.hpp"
#include "snippets/pass/explicit_transpose_matmul_inputs.hpp"
#if defined(SNIPPETS_LIBXSMM_TPP)
#include "transformations/tpp/x64/pass/brgemm_to_brgemm_tpp.hpp"
#endif

// Misc
#include "nodes/mvn.h"
#include "nodes/normalize.h"
#include "nodes/fake_quantize.h"
#include "nodes/mha.h"
#include "nodes/rnn.h"
#include "nodes/scaled_attn.h"
#include "nodes/llm_mlp.h"
#include "nodes/qkv_proj.h"
#include "nodes/rms_norm.h"
#include "dnnl.hpp"
#if defined(OPENVINO_ARCH_ARM64)
#include "cpu/aarch64/cpu_isa_traits.hpp"
#else
#include "cpu/x64/cpu_isa_traits.hpp"
#endif
#include "openvino/core/validation_util.hpp"

namespace ov {
namespace intel_cpu {

using const_node_ptr = const std::shared_ptr<const ov::Node>;

bool Transformations::is_decompression_multiply(const_node_ptr& node) const {
    auto all_has_type = [](const std::set<ov::Input<ov::Node>>& consumers, const ov::DiscreteTypeInfo& type) {
        return std::all_of(consumers.begin(), consumers.end(), [&type](const ov::Input<ov::Node>& input) {
            return input.get_node()->get_type_info() == type;
        });
    };

    const auto consumers = node->get_output_target_inputs(0);
    if (all_has_type(consumers, ov::opset1::MatMul::get_type_info_static()))
        return true;

    auto are_converts_from_decompression = [&all_has_type](const std::set<ov::Input<ov::Node>>& consumers) {
        if (!all_has_type(consumers, ov::opset1::Convert::get_type_info_static()))
            return false;
        for (const auto& consumer : consumers) {
            const auto child_consumers = consumer.get_node()->get_output_target_inputs(0);
            if (!all_has_type(child_consumers, ov::opset1::MatMul::get_type_info_static()))
                return false;
        }
        return true;
    };

    if (all_has_type(consumers, ov::opset1::Reshape::get_type_info_static())) {
        for (const auto& consumer : consumers) {
            const auto child_consumers = consumer.get_node()->get_output_target_inputs(0);
            if (all_has_type(child_consumers, ov::opset1::MatMul::get_type_info_static()) ||
                are_converts_from_decompression(child_consumers)) {
                return true;
            }
        }
    }
    return are_converts_from_decompression(consumers);
}

bool Transformations::fuse_type_to_fq(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    auto fq = ov::as_type_ptr<ov::opset1::FakeQuantize>(node);
    if (!fq)
        return false;
    const auto& from = node->get_output_element_type(0);
    auto it = precisions.find(from);
    if (it == precisions.end())
        return false;
    const auto& to = it->second;

    for (size_t i = 0; i < node->get_input_size(); ++i) {
        auto convert_before = std::make_shared<opset4::Convert>(node->input_value(i), from);
        node->input(i).replace_source_output(convert_before);
    }

    auto consumers = node->output(0).get_target_inputs();
    for (auto& input : consumers) {
        const auto consumer = input.get_node();
        if (ov::is_type<ov::op::v0::Result>(consumer) || ov::is_type<ov::op::v0::Convert>(consumer)) {
            continue;
        }
        auto convert_after = std::make_shared<opset4::Convert>(node, to);
        input.replace_source_output(convert_after);
    }

    return true;
}

bool Transformations::fuse_type_to_pa(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    auto pa = ov::as_type_ptr<ov::op::PagedAttentionExtension>(node);
    if (!pa)
        return false;
    // PagedAttentionExtension's 2nd output type should be kept f32.
    // The reason is that the pagedattention node in CPU plugin hardcodes 2nd output type as f32.
    // So, set f32 to the 2nd output type, which can avoid extra data type conversion during transformation.
    pa->set_out_type(1, ov::element::f32);
    return true;
}

bool Transformations::fuse_type_to_convert(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
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

void Transformations::UpToLpt() {
    using namespace ov::pass::low_precision;
    static const std::set<levels>& supported_fq_levels = {
        levels::int4,
        levels::int4_narrow_range,
        levels::int8,
        levels::int8_narrow_range
    };

    const bool useLpt = config.lpTransformsMode == Config::LPTransformsMode::On &&
        LowPrecision::isFunctionQuantized(model, supported_fq_levels) &&
        CPU_DEBUG_CAP_IS_TRANSFORMATION_ENABLED(config.debugCaps, Lpt);

    const auto defaultPrecisions = useLpt ? precision_set::get_int8_support() : std::vector<ov::element::Type>{};

    PreLpt(defaultPrecisions);

    if (useLpt)
        Lpt(defaultPrecisions);
}

void Transformations::CpuSpecificOpSet(void) {
    CPU_DEBUG_CAP_TRANSFORMATION_SCOPE(this, Specific);

    ConvertToCPUSpecificOpset(model);
}

void Transformations::PreLpt(const std::vector<ov::element::Type>& defaultPrecisions) {
    CPU_DEBUG_CAP_TRANSFORMATION_SCOPE(this, PreLpt);

    // Decompression handling related transformations must be run separately from common preLPT pipeline
    // since there is used the same transformations as in LPT related transformations, but with the specific settings.
    // This must be done in order to keep compressed MatMul weights with decompression operations as is
    ov::pass::Manager decompression_handling_manager("CPU:DecompressionHandling");
    decompression_handling_manager.set_per_pass_validation(false);
    CPU_REGISTER_PASS_COMMON(decompression_handling_manager, ov::pass::InitNodeInfo);
    const bool useLpt = !defaultPrecisions.empty();
    CPU_REGISTER_PASS_COMMON(decompression_handling_manager, ov::pass::ConvertGatherToGatherCompressed);
    CPU_REGISTER_PASS_COMMON(decompression_handling_manager, ov::pass::MarkShapeOfSubgraphs);
    // We need to fuse Transpose to MatMul to have a simpler callback for the next transformation
    CPU_REGISTER_PASS_X64(decompression_handling_manager, ov::pass::TransposeMatMul);
    ov::element::TypeVector decompression_precisions{ov::element::u8,
                                                     ov::element::i8,
                                                     ov::element::u4,
                                                     ov::element::i4,
                                                     ov::element::nf4,
                                                     ov::element::f4e2m1};
    CPU_REGISTER_PASS_X64(decompression_handling_manager, ov::pass::MarkDequantizationSubgraph, decompression_precisions, false, true);
    CPU_SET_CALLBACK_X64(decompression_handling_manager, [&](const_node_ptr &node) -> bool {
        return !is_decompression_multiply(node);
    }, ov::pass::MarkDequantizationSubgraph);

    CPU_SET_CALLBACK_COMMON(
        decompression_handling_manager,
        [&](const_node_ptr& node) -> bool {
            if (ov::is_type<ov::op::internal::GatherCompressed>(node)) {
                // It is necessary to avoid precision conversion for constant node(compressed weights)
                ov::enable_keep_const_precision(node->get_input_node_shared_ptr(0));

                // Prioritize LPT pipeline to handle dequantization part for quantized models as it more optimal in
                // general case
                if (ov::intel_cpu::one_of(node->get_input_node_shared_ptr(0)->get_element_type(),
                                          ov::element::u8,
                                          ov::element::i8) &&
                    useLpt) {
                    return true;
                }
            }
            return false;
        },
        ov::pass::ConvertGatherToGatherCompressed);
    decompression_handling_manager.run_passes(model);

    ov::pass::Manager manager("Plugin:CPU");
    manager.set_per_pass_validation(false);
    if (useLpt)
        CPU_REGISTER_PASS_COMMON(manager, ov::pass::MarkDequantizationSubgraph, defaultPrecisions);

    auto get_convert_precisions = [&]() {
        precisions_map map = {
            {ov::element::i64,     ov::element::i32},
            {ov::element::u64,     ov::element::i32},
            {ov::element::i16,     ov::element::i32},
            {ov::element::u16,     ov::element::i32},
            {ov::element::u32,     ov::element::i32},
            {ov::element::f64,     ov::element::f32},
            {ov::element::boolean, ov::element::u8},
            {ov::element::i4,      ov::element::i8},
            {ov::element::u4,      ov::element::u8}
        };

        // @todo should we always convert to f32 regardless of hardware support, as it is done for f16?
        if (!hasHardwareSupport(ov::element::bf16))
            map.insert({ov::element::bf16, ov::element::f32});
        // TODO: Remove 'hasHardwareSupport' when all nodes are able to handle f16 properly.
        if (!one_of(config.inferencePrecision, element::f16, element::undefined) || !hasHardwareSupport(element::f16)) {
            map.insert({ov::element::f16, ov::element::f32});
        }
        return map;
    };

    type_to_fuse_map type_to_fuse = {{ov::opset10::Convert::get_type_info_static(), fuse_type_to_convert}};

    // It cannot be static data, because it may be difference for different inferencePrecision
    const auto precisions = get_convert_precisions();
    if (config.inferencePrecision == ov::element::f16) {
        precisions_map fp_convert_precision_map = {{ov::element::f32, ov::element::f16}};
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
        type_to_fuse_map fuse_map = {{ov::opset1::FakeQuantize::get_type_info_static(), fuse_type_to_fq}};
#else
        type_to_fuse_map fuse_map = {{ov::op::PagedAttentionExtension::get_type_info_static(), fuse_type_to_pa}};
#endif
        const bool keep_precision_sensitive_in_fp32 = true;
        CPU_REGISTER_PASS_COMMON(manager,
                                 ov::pass::ConvertPrecision,
                                 fp_convert_precision_map,
                                 fuse_map,
                                 keep_precision_sensitive_in_fp32,
                                 false);
    }
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::KeepConstAndDecompression);
    CPU_SET_CALLBACK_COMMON(manager,
        [](const_node_ptr &node) -> bool {
            const auto consumers = node->get_output_target_inputs(0);
            return std::all_of(consumers.begin(), consumers.end(), [](const ov::Input<ov::Node>& consumer) {
                return !ov::is_type<ov::op::v0::MatMul>(consumer.get_node());
            });
        },
        ov::pass::KeepConstAndDecompression);

    CPU_REGISTER_PASS_COMMON(manager, ov::pass::AUGRUCellFusion);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::CommonOptimizations);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::WrapInterpolateIntoTransposes);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::TransposeSinking);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConvertSequenceToTensorIterator);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConvertOpSet3ToOpSet2);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConvertOpSet2ToOpSet1);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::LSTMCellDecomposition);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::GRUCellDecomposition);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::RNNCellDecomposition);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConvertNMS1ToNMS9);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConvertNMS3ToNMS9);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConvertNMS4ToNMS9);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConvertNMS5ToNMS9);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConvertNMS9ToNMSIEInternal);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::Validate);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConvertMulticlassNmsToMulticlassNmsIE);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::Validate);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConvertMatrixNmsToMatrixNmsIE);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::Validate);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::TransposeMatMul);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConstantFolding);
    CPU_REGISTER_PASS_ARM64(manager, ov::pass::HardSigmoidDecomposition);

    if (useLpt) {
        CPU_LPT_SCOPE(LowPrecisionTransformations_Part2);
        CPU_REGISTER_PASS_COMMON(manager, ov::pass::low_precision::ConvertSubtractConstant, defaultPrecisions);
    }
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::Validate);
    // Common ConvertPrecision pass handles only a limited set of opevino operations to match the list of precisions supported by the plugin.
    // However, if the extension operation produces an output precision that is not natively supported, this may lead to inconsistency during
    // element type propagation. This transformation is called before the ConvertPrecision pass to align the actual precisions with the list of supported ones.
    constexpr bool convert_input_output_precision = false;
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::InsertConvertAfterExtension, convert_input_output_precision);
    // Do not insert pass::Validate between pass::InsertConvertAfterExtension and pass::ConvertPrecision.
    // This may result in the loss of the original Element type of the Output .
    // element type convert is disabled.
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConvertPrecision, precisions, type_to_fuse, false, convert_input_output_precision);

    CPU_REGISTER_PASS_COMMON(manager, ov::pass::EliminateConvert);
    CPU_REGISTER_PASS_COMMON(manager, SwapConvertTranspose);
    CPU_REGISTER_PASS_X64(manager, ConvertToInteraction);
    CPU_REGISTER_PASS_X64(manager, ConvertInteractionInt8);
    CPU_REGISTER_PASS_ARM(manager, ConvertReduceNoKeepDims);
    CPU_REGISTER_PASS_ARM(manager, ConvertReduceMultiAxis);
    CPU_REGISTER_PASS_ARM32(manager, MishDecomposition);
    CPU_REGISTER_PASS_ARM(manager, ConvertConv1D);
    CPU_REGISTER_PASS_ARM(manager, ConvertGroupConv1D);
    CPU_REGISTER_PASS_ARM(manager, ConvertGroupConvolution);
    // The plugin computes Divide in floating point precision.
    // To preserve correct math for integer division we need to insert explicit Floor operation.
    CPU_REGISTER_PASS_ARM(manager, DecomposeIntegerDivide);
    CPU_REGISTER_PASS_X86(manager, DecomposeIntegerDivide);

    CPU_REGISTER_PASS_COMMON(manager, PermuteSliceAndInterpolation);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::Validate);

    // SpaceToDepth/ DepthToSpace node implementation supports only equal input/output tensors with rank <= 5
    CPU_SET_CALLBACK_COMMON(manager,
        [](const_node_ptr &node) -> bool {
            return node->input_value(0).get_shape().size() <= 5lu &&
                node->input_value(0).get_shape().size() == node->get_output_shape(0).size();
        },
        ov::pass::ConvertSpaceToDepth, ov::pass::ConvertDepthToSpace);

    CPU_SET_CALLBACK_COMMON(manager,
        [](const_node_ptr &node) -> bool {
            const auto & rank = node->input(0).get_partial_shape().rank().get_length();
            return rank == 4lu || rank == 5lu;
        },
        ov::pass::ConvertBatchToSpace, ov::pass::ConvertSpaceToBatch);

    CPU_SET_CALLBACK_COMMON(
        manager,
        [](const_node_ptr& node) -> bool {
            const auto maxpool = std::dynamic_pointer_cast<const ov::op::v14::MaxPool>(node);
            return !maxpool || maxpool->get_rounding_type() == ov::op::RoundingType::CEIL_TORCH;
        },
        ov::pass::ConvertMaxPool14ToMaxPool8);

    CPU_SET_CALLBACK_COMMON(
        manager,
        [](const_node_ptr& node) -> bool {
            const auto avgpool = std::dynamic_pointer_cast<const ov::op::v14::AvgPool>(node);
            return !avgpool || avgpool->get_rounding_type() == ov::op::RoundingType::CEIL_TORCH;
        },
        ov::pass::ConvertAvgPool14ToAvgPool1);

    CPU_SET_CALLBACK_COMMON(manager,
        [](const_node_ptr &node) -> bool {
            std::string msg;
            return node::RNN::isSupportedOperation(node, msg);
        },
        ov::pass::ConvertRNNSequenceToTensorIterator,
        ov::pass::ConvertGRUSequenceToTensorIterator,
        ov::pass::ConvertLSTMSequenceToTensorIterator);

    CPU_SET_CALLBACK_COMMON(manager,
        [](const_node_ptr &node) -> bool {
            std::string msg;
            return !node::RNN::isSupportedOperation(node, msg);
        },
        ov::pass::ConvertLoopToLSTMSequence,
        ov::pass::FuseReverseLSTMSequence,
        ov::pass::FuseLSTMSequencesToBidirectionalLSTMSequence);

    CPU_SET_CALLBACK_COMMON(manager,
        [](const_node_ptr &node) -> bool {
            std::string msg;
            return node::RNN::isSupportedOperation(node, msg);
        },
        ov::pass::RNNCellDecomposition,
        ov::pass::GRUCellDecomposition,
        ov::pass::LSTMCellDecomposition);

    CPU_SET_CALLBACK_COMMON(manager,
        [](const_node_ptr &node) -> bool {
            std::string msg;
            return !node::RNN::isSupportedOperation(node, msg);
        },
        ov::pass::LSTMCellFusion);

    CPU_SET_CALLBACK_COMMON(manager,
        [](const_node_ptr &node) -> bool {
            std::string errorMessage;
            return node::MVN::isSupportedOperation(node, errorMessage);
        },
        ov::pass::MVN6Decomposition);

    CPU_SET_CALLBACK_COMMON(manager,
        [](const_node_ptr &node) -> bool {
            std::string errorMsg;
            return node::NormalizeL2::isSupportedOperation(node, errorMsg);
        },
        ov::pass::NormalizeL2Decomposition);

    if (!useLpt) {
        CPU_SET_CALLBACK_X64(manager,
            [this](const_node_ptr &node) -> bool {
                // This is a callback from snippets. If GroupNorm node is appropriate for snippets execution with higher perf,
                // then it will not be decomposed to mvn+reshape+eltwises, support it with snippets instead.
                // Callback is used here, why not call GroupNormalizationDecomposition after snippets transformation pipeline is because
                // 1. If GN is not tokenized conditionally in snippets transformation pipeline, GroupNormalizationDecomposition will produce
                //    "reshpae + mvn + reshape + mul + add". these simple ops will not tokenized into subgraph, lead to suboptimal perf.
                // 2. GroupNormalizationDecomposition produce MVN, and MVN have a conditional pass MVN6Decomposition. If call MVN6Decomposition again after
                //    snippets pipeline as well, where MVN is decomposed to simple ops, these simple ops will not tokenized into subgraph again.
                // CVS-134277 to fully enable GN as snippets to disable this GroupNormalizationDecomposition entirly.
                if (node->is_dynamic() || !one_of(config.inferencePrecision, element::f32, element::undefined) ||
                    config.snippetsMode == Config::SnippetsMode::Disable)
                    return false;
                if (config.snippetsMode != Config::SnippetsMode::IgnoreCallback) {
                    const auto group_norm = ov::as_type_ptr<const ov::op::v12::GroupNormalization>(node);
                    if (!group_norm || !implication(config.inferencePrecision == element::undefined, group_norm->get_element_type() == element::f32))
                        return false;
                    const auto num_groups = static_cast<size_t>(group_norm->get_num_groups());
                    const auto shape = group_norm->get_input_partial_shape(0).to_shape();
                    size_t snippets_work_amount = shape[0] * num_groups;
                    size_t concurrency = parallel_get_max_threads();
                    if (concurrency > snippets_work_amount)
                        return false;
                    size_t spatial_dim = 1;
                    for (size_t i = 2; i < shape.size(); ++i)
                        spatial_dim = spatial_dim * shape[i];
                    size_t snippets_tensor_size = spatial_dim * shape[1] / num_groups * node->get_element_type().size();
                    size_t cache_size_l1 = dnnl::utils::get_cache_size(1, true);
                    if (snippets_tensor_size > cache_size_l1) {
                        return false;
                    }
                }

                return true;
            },
            ov::pass::GroupNormalizationDecomposition);
    }

    CPU_ENABLE_PASS_COMMON(manager, ov::pass::SoftmaxDecomposition);
    CPU_SET_CALLBACK_COMMON(manager,
            [](const_node_ptr &node) -> bool {
                return node->input_value(0).get_partial_shape().rank().get_length() <= 5;
            },
        ov::pass::SoftmaxDecomposition);

    // NMS-alike nodes are always transformed to NMSIEInternal node in case of legacy api, for compatibility.
    // And on the other hand in case of api 2.0, keep them internal dynamic for better performance and functionality.
    auto nmsCallback = [](const_node_ptr &node) -> bool {
        // TODO: remove nmsCallback at all
        const bool isLegacyApi = false;
        return isLegacyApi ?  false : true;
    };

    CPU_SET_CALLBACK_COMMON(manager, nmsCallback, ov::pass::ConvertNMS9ToNMSIEInternal);
    CPU_SET_CALLBACK_COMMON(manager, nmsCallback, ov::pass::ConvertMulticlassNmsToMulticlassNmsIE);
    CPU_SET_CALLBACK_COMMON(manager, nmsCallback, ov::pass::ConvertMatrixNmsToMatrixNmsIE);
    CPU_SET_CALLBACK_COMMON(manager,
        [this](const_node_ptr &node) -> bool {
            std::string errorMsg;
            // Current SDPA impl is optimized only for LLM models, so we decompose it for others to avoid perf regression.
            // Matching the pattern is a little complicated, so we just check if there is any state nodes.
            return node::ScaledDotProductAttention::isSupportedOperation(node, errorMsg) && model->get_variables().size() > 0;
        },
        ov::pass::ScaledDotProductAttentionDecomposition);

    // List of enabled/disabled transformations

    // Allow FP16 Converts to be folded and FP16 constants to be upgraded to FP32 data type
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::DisableDecompressionConvertConstantFolding);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertCompressedOnlyToLegacy);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::EyeDecomposition);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertGELU);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertShuffleChannels3);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::Gelu7Downgrade);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::SoftPlusDecomposition);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertMod);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertShuffleChannels3);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::WeightsDequantizeToFakeQuantize);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::SimplifyCTCGreedyDecoderSeqLen);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertGather7ToGather1);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertGather8ToGather7);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertMinimum);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertBroadcastToTiles);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertReduceMeanToPooling);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertReduceMaxToPooling);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertReduceSumToPooling);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::SliceToStridedSlice);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertDetectionOutput8ToDetectionOutput1);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertROIAlign9To3);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::SoftSignDecomposition);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::UniqueDecomposition);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertTopK3);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertTopK11ToTopK3);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::HSwishDecomposition);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::MatMulConstTransposesExtraction);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertScatterNDUpdate15ToScatterNDUpdate3);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertSliceScatter);
    CPU_DISABLE_PASS_X64(manager, ov::pass::HSigmoidDecomposition);

    CPU_DISABLE_PASS_X64(manager, ov::pass::ReduceL1Decomposition);
    CPU_DISABLE_PASS_X64(manager, ov::pass::ReduceL2Decomposition);

    CPU_ENABLE_PASS_COMMON(manager, ov::pass::NormalizeL2Decomposition);
    CPU_ENABLE_PASS_COMMON(manager, ov::pass::ConvertInterpolate1ToInterpolate4);
    CPU_ENABLE_PASS_COMMON(manager, ov::pass::ConvertGather1ToGather7);
    CPU_ENABLE_PASS_COMMON(manager, ov::pass::ConvertDetectionOutput1ToDetectionOutput8);
    CPU_ENABLE_PASS_COMMON(manager, ov::pass::ConvertROIAlign3To9);

    if (useLpt) {
        CPU_LPT_SCOPE(LowPrecisionTransformations_Part3);
        CPU_SET_CALLBACK_COMMON(manager,
            [](const_node_ptr &node) -> bool {
                std::string errMsg;
                return !node::FakeQuantize::isSupportedOperation(node, errMsg);
            },
            ov::pass::AddFakeQuantizeFusion,
            ov::pass::MulFakeQuantizeFusion,
            ov::pass::FakeQuantizeMulFusion);

        CPU_SET_CALLBACK_COMMON(manager,
            [&defaultPrecisions](const_node_ptr &node) -> bool {
                return ov::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForMultiply(node, defaultPrecisions);
            },
            ov::pass::ConvertQuantizeDequantize);
    }

    /* In some cases, during the transformation pipeline, some MatMul nodes can be transformed into other nodes. For example, they can become part of
       AUGRUCell node (see AUGRUCellFusion pass). In such cases, some constant paths will be unfolded, which can lead to crashes in the plugin. To avoid this,
       we re-mark decompression converts again and finally do CF for those constant paths that are not inputs to MatMul node */
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::EnableDecompressionConvertConstantFolding);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::KeepConstAndDecompression);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConstantFolding);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::LoraSubgraphFusion);

    manager.run_passes(model);
}

void Transformations::Lpt(const std::vector<ov::element::Type>& defaultPrecisions) {
    CPU_DEBUG_CAP_TRANSFORMATION_SCOPE(this, Lpt);

    using namespace ov::pass::low_precision;
    CPU_LPT_SCOPE(LowPrecisionTransformations_Part4);
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "LowPrecisionTransformations");
    // Only enable conv/group conv signed input on AMX and avx2_vnni_2 platform.
    std::vector<ov::element::Type> input0LowPrecisionList;
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx) ||
        dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni_2)) {
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
            PrecisionsRestriction::create<ov::opset1::GroupConvolution>([input0LowPrecisionList](const std::shared_ptr<ov::Node>& node){
                const auto& input_partial_shape = node->get_input_partial_shape(0);
                const auto& rank = input_partial_shape.rank();
                if (rank.is_static() && (rank.get_length() == 5)) {
                    return PrecisionsRestriction::PrecisionsByPorts{
                        {{0}, {ov::element::u8, ov::element::i8}},
                        {{1}, {ov::element::i8}}};
                }

                return PrecisionsRestriction::PrecisionsByPorts{
                    {{0}, input0LowPrecisionList},
                    {{1}, {ov::element::i8}}
                };
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
                    {{0, 1}, {ov::element::u8}}
                }),
            PrecisionsRestriction::create<ov::opset6::GRUSequence>({
                    {{0, 1}, {ov::element::u8}}
                }),
        });

    auto quantizationRestrictions = std::vector<QuantizationGranularityRestriction>({
            QuantizationGranularityRestriction::create<ov::opset1::Convolution>({0}),
            QuantizationGranularityRestriction::create<ov::opset1::ConvolutionBackpropData>({0})
        });

    ov::pass::Manager lptManager("CPU:LPT");
    CPU_REGISTER_PASS_COMMON(lptManager, LowPrecision,
        supportedPrecisions,
        quantizationRestrictions,
        LayerTransformation::Params(true, ov::element::f32, defaultPrecisions));

    CPU_SET_CALLBACK_COMMON(lptManager, [](const_node_ptr& node) -> bool {
        return ov::is_type<ov::opset1::Multiply>(node) &&
               !MultiplyToGroupConvolutionTransformation::canBeTransformedToGroupConvolution(node);
    }, MarkupPrecisions);
    CPU_SET_CALLBACK_COMMON(lptManager, [&defaultPrecisions](const_node_ptr& node) -> bool {
        return LayerTransformation::isAsymmetricQuantization(node, defaultPrecisions) ||
               WeightableLayerTransformation::isAsymmetricOnWeights(node, defaultPrecisions);
    }, ConvolutionBackpropDataTransformation);
    CPU_SET_CALLBACK_COMMON(lptManager, [](const_node_ptr& node) -> bool {
        return ov::marked_as_bias(node);
    }, AddTransformation);

    CPU_SET_CALLBACK_X64(lptManager, [&](const_node_ptr& node) -> bool {
        const auto& consumers = node->get_output_target_inputs(0);
        if (consumers.size() == 1) {
            const auto consumer = consumers.begin()->get_node()->shared_from_this();
            return ov::is_type<ov::opset1::Multiply>(consumer) && is_decompression_multiply(consumer);
        }
        return false;
    }, FoldConvertTransformation);

    CPU_SET_CALLBACK_X64(lptManager, [&](const_node_ptr& node) -> bool {
        if (ov::is_type<ov::opset1::Multiply>(node)) {
            return ov::is_type<ov::opset1::Multiply>(node) && is_decompression_multiply(node);
        } else if (ov::is_type<ov::opset1::Subtract>(node)) {
            const auto& consumers = node->get_output_target_inputs(0);
            if (consumers.size() == 1) {
                const auto consumer = consumers.begin()->get_node()->shared_from_this();
                return ov::is_type<ov::opset1::Multiply>(consumer) && is_decompression_multiply(consumer);
            }
        }
        return false;
    }, FuseConvertTransformation);

    CPU_DISABLE_PASS_ARM(lptManager, RecurrentCellTransformation);
    CPU_DISABLE_PASS_COMMON(lptManager, MultiplyToGroupConvolutionTransformation);

    lptManager.run_passes(model);
}

void Transformations::PostLpt() {
    CPU_DEBUG_CAP_TRANSFORMATION_SCOPE(this, PostLpt);

    ov::pass::Manager postLPTPassManager("CPU:PostLPT");
    postLPTPassManager.set_per_pass_validation(false);
    CPU_REGISTER_PASS_COMMON(postLPTPassManager, ov::pass::UnrollTensorIterator);
    CPU_REGISTER_PASS_COMMON(postLPTPassManager, ov::pass::ReshapePRelu);
    CPU_SET_CALLBACK_COMMON(postLPTPassManager,
        [](const_node_ptr &node) -> bool {
            // UnrollTI transformation is disabled by default, is turned on by LowLatency transformation
            return node->get_rt_info().count("UNROLL_TI") == 0;
        },
        ov::pass::UnrollTensorIterator);
    CPU_REGISTER_PASS_COMMON(postLPTPassManager, ov::pass::MoveEltwiseUpThroughDataMov);
    CPU_DISABLE_PASS_COMMON(postLPTPassManager, ov::pass::MoveEltwiseUpThroughDataMovPerChannel);
    CPU_SET_CALLBACK_COMMON(postLPTPassManager,
        [](const std::shared_ptr<const ov::Node>& node) -> bool {
            if (!ov::is_type<const ov::op::v0::FakeQuantize>(node) && node->get_output_element_type(0) != node->get_input_element_type(0))
                return true;
            if (node->get_input_size() >= 2) {
                return node->get_input_element_type(1) == ov::element::i8 ||
                       node->get_input_element_type(1) == ov::element::u8 ||
                       (ov::is_type<const ov::op::v0::FakeQuantize>(node) && !ov::is_type<const ov::op::v1::Transpose>(node->get_input_node_shared_ptr(0)));
            }
            return false;
        },
        ov::pass::MoveEltwiseUpThroughDataMovScalar);
    CPU_REGISTER_PASS_COMMON(postLPTPassManager, ov::pass::Validate);

    CPU_REGISTER_PASS_COMMON(postLPTPassManager, ov::pass::ConstantFolding);

    CPU_REGISTER_PASS_X64(postLPTPassManager, FuseFQtoInteraction);

    // Execute before snippets. Otherwise FQ will be converted to Subgraph
    CPU_REGISTER_PASS_X64(postLPTPassManager, ConvertFqRnnToQuantizedRnn);

    CPU_REGISTER_PASS_X64(postLPTPassManager, ov::pass::RoPEFusion, true);
    CPU_REGISTER_PASS_ARM64(postLPTPassManager, ov::pass::RoPEFusion, true);
    CPU_REGISTER_PASS_X64(postLPTPassManager, CausalMaskPreprocessFusion);

#if defined(OPENVINO_ARCH_X86_64)
    // MLP & QKV fusion optimizations is focused on throughput, only enabled on AMX-bf16 & LLM serving use cases.
    auto can_use_amx_bf16_int8 = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx) && (config.inferencePrecision == element::bf16);
    auto can_use_amx_fp16 = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx_fp16) && (config.inferencePrecision == element::f16);

    if (can_use_amx_bf16_int8 || can_use_amx_fp16) {
        const auto fcDynamicQuantizationGroupSize = config.fcDynamicQuantizationGroupSize;
        CPU_REGISTER_PASS_X64(postLPTPassManager, MLPFusion);
        CPU_SET_CALLBACK_X64(postLPTPassManager,
            [fcDynamicQuantizationGroupSize](const_node_ptr &node) -> bool {
                std::string errorMsg;
                return node::LLMMLP::isSupportedOperation(node, errorMsg, fcDynamicQuantizationGroupSize);
            },
            MLPFusion);

        size_t concurrency = config.streamExecutorConfig.get_threads_per_stream();
        if (concurrency == 0)
            concurrency = parallel_get_max_threads();

        CPU_REGISTER_PASS_X64(postLPTPassManager, QKVProjFusion);
        CPU_SET_CALLBACK_X64(postLPTPassManager,
            [=](const_node_ptr &node) -> bool {
                std::string errorMsg;
                return node::QKVProjection::isSupportedOperation(node, errorMsg, concurrency, fcDynamicQuantizationGroupSize);
            },
            QKVProjFusion);

        CPU_REGISTER_PASS_X64(postLPTPassManager, QKVProjFusion2);
        CPU_SET_CALLBACK_X64(postLPTPassManager,
            [=](const_node_ptr &node) -> bool {
                std::string errorMsg;
                return node::QKVProjection::isSupportedOperation(node, errorMsg, concurrency, fcDynamicQuantizationGroupSize);
            },
            QKVProjFusion2);
    }
#endif // OPENVINO_ARCH_X86_64

    CPU_REGISTER_PASS_COMMON(postLPTPassManager, ov::pass::transpose_sinking::TSShapeOfForward);
    CPU_REGISTER_PASS_COMMON(postLPTPassManager, StatefulSDPAFusion);
    CPU_REGISTER_PASS_X64(postLPTPassManager, ov::intel_cpu::SDPAFuseTransposeReshape);
    CPU_REGISTER_PASS_X64(postLPTPassManager, ov::pass::RMSFusion, false);
    CPU_REGISTER_PASS_X64(postLPTPassManager, ov::intel_cpu::DecomposeRMSNorm);
    CPU_SET_CALLBACK_X64(postLPTPassManager,
        [](const std::shared_ptr<const ov::Node>& node) -> bool {
            std::string errorMsg;
            return node::RMSNorm::isSupportedOperation(node, errorMsg);
        },
        ov::intel_cpu::DecomposeRMSNorm);

    // markup Rope Input when BF16/F16 inference.
    if (one_of(config.inferencePrecision, ov::element::bf16, ov::element::f16)) {
        CPU_REGISTER_PASS_COMMON(postLPTPassManager, ov::pass::MarkRopeInputsToKeepInMixedPrecision);
        CPU_REGISTER_PASS_COMMON(postLPTPassManager, ov::pass::MarkFloatingPointRange);
    }

    // Should be before Snippets pipeline because Ngram pattern contains eltwise nodes that can be tokenized by Snippets.
    auto symbolic_pipeline = CPU_REGISTER_PASS_COMMON(postLPTPassManager, ov::pass::SymbolicOptimizations, false);
    symbolic_pipeline->get_manager()->register_pass<NgramFusion>();

    postLPTPassManager.run_passes(model);
}

void Transformations::MainSnippets(void) {
    auto is_supported_isa = [](){
#if defined(OPENVINO_ARCH_X86_64)
        return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2);
#elif defined(OPENVINO_ARCH_ARM64)
        return dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::asimd);
#endif
        return false;
    };

    if (config.snippetsMode == Config::SnippetsMode::Disable || !is_supported_isa())
        return;

    // TODO [123659] Implement common logic to split optimization and limitation conditions
    const auto ignoreCallback = config.snippetsMode == Config::SnippetsMode::IgnoreCallback;

    // [111813]: At the moment Snippets supports Transpose on output of MHA pattern only if it is an one node between MatMul and Result.
    // However there may be Convert [f32->bf16] before Result since:
    //  - bf16 Brgemm has f32 output;
    //  - CPU Node Subgraph requires bf16 on output when inference precision is bf16.
    // To avoid sitations when Transpose is not alone node between MatMul and Result,
    // Plugin disables Transpose tokenization on output
    bool mha_token_enable_transpose_on_output = one_of(config.inferencePrecision, element::f32, element::undefined);
    size_t concurrency = config.streamExecutorConfig.get_threads_per_stream();
    if (concurrency == 0)
        concurrency = parallel_get_max_threads();

    // Runtime caching should be enabled in case of dynamic Subgraphs in CPU Plugin: to reduce overheads of ShapeInference and CodeGeneration
    // If runtime cache capacity is zero, it means that rtCache won't be used and
    // we shouldn't tokenize dynamic Subgraphs - it will lead to performance degradations
    bool is_dynamic_mha_token_enabled = config.rtCacheCapacity != 0;
#if defined(OPENVINO_ARCH_ARM64)
    // ARM has 32 gprs. After excluding 2 registers for work amounts, 1 register for runtime parameters, 1 platform register,
    // 3 registers for temporary use, and 2 stack related registers, it has 23 remaining registers.
    size_t data_ptr_gpr_count = 23;
    // ARM doesn't even support MHA yet
    is_dynamic_mha_token_enabled = false;
#else
    // X64 has 16 gprs. After excluding 2 registers for work amounts, 1 register for runtime parameters,
    // and 2 stack related registers, it has 11 remaining registers.
    size_t data_ptr_gpr_count = 11;
#endif
    // The optimization "SplitDimensionM" depends on target machine (thread count).
    // To avoid uncontrolled behavior in tests, we disabled the optimization when there is Config::SnippetsMode::IgnoreCallback
    bool split_m_dimension = !ignoreCallback;
    // [122706] Some 3D MHA Patterns have perf regressions when Transpose op is tokenized
    std::set<size_t> mha_supported_transpose_ranks = { 4 };
    snippets::pass::SnippetsTokenization::Config tokenization_config(concurrency, data_ptr_gpr_count, split_m_dimension,
                                                                     mha_token_enable_transpose_on_output, is_dynamic_mha_token_enabled,
                                                                     mha_supported_transpose_ranks);

    ov::pass::Manager snippetsManager("CPU:Snippets");
    snippetsManager.set_per_pass_validation(false);
    if (!ignoreCallback) {
#if defined(OPENVINO_ARCH_ARM64)
        CPU_REGISTER_PASS_ARM(snippetsManager, SnippetsMarkSkipped);
#else
        CPU_REGISTER_PASS_X64(snippetsManager, SnippetsMarkSkipped, config.inferencePrecision == ov::element::bf16);
#endif
        CPU_DISABLE_PASS_COMMON(snippetsManager, snippets::pass::TokenizeFCSnippets);
    }
    CPU_REGISTER_PASS_COMMON(snippetsManager, snippets::pass::SnippetsTokenization, tokenization_config);

    // - MHA has BRGEMM that is supported only on AVX512 platforms
    // - CPU Plugin Subgraph supports only f32, bf16 (and quantized) BRGEMM
    //   [122494] Need to add support of f16
    const bool isMHASupported =
#if defined(OPENVINO_ARCH_ARM64)
            false;
#else
            (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2) &&
             one_of(config.inferencePrecision, ov::element::f32, element::undefined)) ||
            (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) &&
             one_of(config.inferencePrecision, ov::element::bf16, ov::element::f32, element::undefined));
#endif
    if (!isMHASupported) {
        CPU_DISABLE_PASS_COMMON(snippetsManager, snippets::pass::TokenizeMHASnippets);
        CPU_DISABLE_PASS_COMMON(snippetsManager, snippets::pass::ExtractReshapesFromMHA);
    }

#if defined(OPENVINO_ARCH_X86_64)
    auto is_supported_matmul = [this](const std::shared_ptr<const ov::Node>& n) {
        const auto matmul = ov::as_type_ptr<const ov::op::v0::MatMul>(n);
        if (!matmul)
            return false;
        const auto in_type0 = matmul->get_input_element_type(0);
        const auto in_type1 = matmul->get_input_element_type(1);
        const auto is_fp32 = (in_type0 == ov::element::f32 && in_type1 == ov::element::f32 &&
                              one_of(config.inferencePrecision, element::f32, element::undefined));
        const auto is_fp16 = (in_type0 == ov::element::f16 || in_type1 == ov::element::f16);
        const auto is_bf16 = (in_type0 == ov::element::bf16 && in_type1 == ov::element::bf16) ||
                             ((in_type0 == element::f32 && in_type1 == ov::element::f32 && config.inferencePrecision == ov::element::bf16));
        const auto is_int8 = in_type0 == ov::element::i8;
        if (is_fp16)
            return false;
        if (is_fp32)
            return true;
        // Only FP32 dynamic MHA is supported
        if (matmul->is_dynamic())
            return false;
        // [114487] brgemm kernel in oneDNN requires brgemm_copy_b kernel if MatMul node has transposed_b=True
        // The current solution with ExtractExplicitMatMulTranspose pass is slower for non-f32 cases than using of brgemm_copy_b kernel
        if (matmul->get_transpose_a() || matmul->get_transpose_b())
            return false;
        // [150842] The execution of Brgemm INT8/BF16 on AMX platforms depends on the value of "K % VNNIFactor".
        //          For more details, please teake a look at the ticket 150842
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx)) {
            const auto& b_shape = matmul->get_input_partial_shape(1);
            const auto K = matmul->get_transpose_b() ? *b_shape.rbegin() : *++b_shape.rbegin();
            if (is_bf16) return K.is_static() && (K.get_length() % 2 == 0);
            if (is_int8) return K.is_static();
        }
        if (is_int8)
            return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_vnni) ||
                   dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni);
        if (is_bf16)
            return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16);
        return true;
    };
    auto is_unsupported_parallel_work_amount = [&](const std::shared_ptr<const ov::Node>& n, const ov::PartialShape& shape) {
        // SplitDimensionM transformation doesn't support dynamic shapes, so M dim is split in runtime configurator
        if (shape.is_dynamic())
            return false;
        const auto parallel_work_amount = std::accumulate(shape.rbegin() + 2, shape.rend(), ov::Dimension(1), std::multiplies<ov::Dimension>());
        const auto is_unsupported_parallel_work_amount =
            static_cast<size_t>(parallel_work_amount.get_length()) < tokenization_config.get_concurrency() &&
            !ov::snippets::pass::SplitDimensionM::can_be_optimized(n, tokenization_config.get_concurrency());
        return is_unsupported_parallel_work_amount;
    };
#endif // OPENVINO_ARCH_X86_64

    auto is_supported_op = [](const std::shared_ptr<const ov::Node> &n) -> bool {
#if defined(OPENVINO_ARCH_ARM64)
        return (ov::is_type<ov::op::v0::Abs>(n) ||
                ov::is_type<ov::op::v1::Add>(n) ||
                ov::is_type<ov::op::v0::Clamp>(n) ||
                ov::is_type<ov::op::v0::Convert>(n) ||
                ov::is_type<ov::op::v1::Divide>(n) ||
                ov::is_type<ov::op::v0::Elu>(n) ||
                ov::is_type<ov::op::v0::Exp>(n) ||
                ov::is_type<ov::op::v0::Floor>(n) ||
                ov::is_type<ov::op::v0::Gelu>(n) ||
                ov::is_type<ov::op::v7::Gelu>(n) ||
                ov::is_type<ov::op::v4::HSwish>(n) ||
                ov::is_type<ov::op::v1::Maximum>(n) ||
                ov::is_type<ov::op::v1::Minimum>(n) ||
                ov::is_type<ov::op::v4::Mish>(n) ||
                ov::is_type<ov::op::v1::Mod>(n) ||
                ov::is_type<ov::op::v1::Multiply>(n) ||
                ov::is_type<ov::op::v0::Relu>(n) ||
                ov::is_type<ov::op::v0::Sigmoid>(n) ||
                ov::is_type<ov::op::v1::Subtract>(n) ||
                ov::is_type<ov::op::v4::Swish>(n) ||
                ov::is_type<ov::op::v0::Tanh>(n));
#else
        // CPU Plugin support Swish in Subgraph via conversion to SwichCPU which assumes second input to be constant,
        // and CPU Plugin does not support Mish for x64
        auto is_unsupported = [](const std::shared_ptr<const ov::Node> &n) {
            return (ov::is_type<const ov::op::v4::Swish>(n) && n->inputs().size() > 1 &&
                   !ov::is_type<const ov::op::v0::Constant>(n->get_input_node_shared_ptr(1))) ||
                   ov::is_type<const ov::op::v4::Mish>(n);
        };
        // todo: general tokenization flow is not currently supported for these operations.
        // they can be tokenized only as a part of complex patterns
        auto is_unsupported_by_common_tokenization = [](const std::shared_ptr<const ov::Node> &n) {
            return (ov::is_type<const ov::op::v1::Softmax>(n) ||
                    ov::is_type<const ov::op::v8::Softmax>(n) ||
                    ov::is_type<const ov::op::v0::MatMul>(n) ||
                    ov::is_type<const ov::op::v1::Transpose>(n) ||
                    ov::is_type<const ov::op::v1::Broadcast>(n) ||
                    ov::is_type<const ov::op::v3::Broadcast>(n) ||
                    ov::is_type<const ov::op::v1::ReduceMax>(n) ||
                    ov::is_type<const ov::op::v1::ReduceSum>(n));
        };
        return !is_unsupported(n) && !is_unsupported_by_common_tokenization(n);
#endif
    };

    auto has_supported_tensors = [ignoreCallback](const std::shared_ptr<const ov::Node> &n) -> bool {
        // Check for supported precision
        auto is_supported_tensor = [&n, ignoreCallback](descriptor::Tensor& t, bool is_input) -> bool {
            // TODO [105804] int32 isn't supported in general because i32 emitters are required for bit-exact i32 calculations in some cases
            // So i32 is supported exclusively for transposes and broadcast
            static const std::set<ov::element::Type> supported_element_types =
#if defined(OPENVINO_ARCH_ARM64)
                {ov::element::f32, ov::element::f16, ov::element::i8, ov::element::u8};
#else
                {ov::element::f32, ov::element::bf16, ov::element::f16, ov::element::i8, ov::element::u8};
#endif

            if (!ignoreCallback) {
                // Check for supported ranks
                // todo: clarify whether we can evaluate snippets on inputs with larger ranks
                if (t.get_partial_shape().rank().get_length() > 6)
                    return false;
            }

            return supported_element_types.count(t.get_element_type()) != 0 || (is_input &&
                   t.get_element_type() == ov::element::i32 &&
                   (ov::is_type<const opset1::Transpose>(n) ||
                    ov::is_type<const opset1::Broadcast>(n) ||
                    ov::is_type<const opset1::ReduceMax>(n) ||
                    ov::is_type<const opset1::ReduceSum>(n)));
        };

        const auto& inputs = n->inputs();
        const auto& outputs = n->outputs();
        return std::all_of(inputs.begin(), inputs.end(), [&](const Input<const ov::Node>& in) {return  is_supported_tensor(in.get_tensor(), true);}) &&
               std::all_of(outputs.begin(), outputs.end(), [&](const Output<const ov::Node>& out) {return  is_supported_tensor(out.get_tensor(), false);});
    };

    if (!ignoreCallback) {
        CPU_SET_CALLBACK_X64(snippetsManager, [&](const std::shared_ptr<const ov::Node>& n) -> bool {
            // Tranformation callback is called on MatMul0
            if (!is_supported_matmul(n))
                return true;
            // Search for MatMul1
            auto child = n->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
            while (!ov::is_type<const ov::op::v0::MatMul>(child)) {
                child = child->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
            }
            if (!is_supported_matmul(child))
                return true;

            const auto& pshape = child->get_input_partial_shape(0);
            return is_unsupported_parallel_work_amount(n, pshape);
        }, snippets::pass::TokenizeMHASnippets);
        CPU_SET_CALLBACK_X64(snippetsManager, [&](const std::shared_ptr<const ov::Node>& n) -> bool {
            return !is_supported_matmul(n) || is_unsupported_parallel_work_amount(n, n->get_output_partial_shape(0));
        }, snippets::pass::ExtractReshapesFromMHA);
    }

    CPU_SET_CALLBACK_COMMON(snippetsManager,
    [&](const std::shared_ptr<const ov::Node>& n) -> bool {
        if (!ignoreCallback) {
            if (n->is_dynamic() || !is_supported_op(n))
                return true;
        }

        const auto& inputs = n->inputs();
        // todo: clarify whether we can evaluate snippets on const paths
        const bool has_only_const_inputs = std::all_of(inputs.begin(), inputs.end(),
                                                        [](const ov::Input<const ov::Node>& in) {
                                                            return ov::is_type<ov::op::v0::Constant>(
                                                                    in.get_source_output().get_node_shared_ptr());
                                                        });
        if (has_only_const_inputs)
            return true;
        if (!has_supported_tensors(n))
            return true;
        return false;
    },
    snippets::pass::TokenizeSnippets);

    auto mm_supports_transpose_b = [this, ignoreCallback](const std::shared_ptr<const ov::Node>& n) {
        MAYBE_UNUSED(config.inferencePrecision);
        if (!ignoreCallback)
            return false;
        // Note: BrgemmTPP doesn't support transposed KN natively
        // so we should extract transposes for the corresponding matmul nodes
#if defined(SNIPPETS_LIBXSMM_TPP)
        // TPP doesn't support dynamic shapes -> there will be BrgemmCPU node
        if (n->is_dynamic())
            return true;
        std::vector<std::vector<size_t>> layouts(3);
        const auto matmul = ov::as_type_ptr<const ov::op::v0::MatMul>(n);
        OPENVINO_ASSERT(matmul, "ExplicitTransposeMatMulInputs callback must be called for matmul node");
        if (matmul->get_transpose_b()) {
            std::vector<size_t> transposed_layout(n->get_input_partial_shape(1).size());
            std::iota(transposed_layout.begin(), transposed_layout.end(), 0);
            std::swap(*transposed_layout.rbegin(), *(transposed_layout.rbegin() + 1));
            layouts[1] = std::move(transposed_layout);
        }
        ov::element::TypeVector precisions;
        auto push_precision = [&](const ov::element::Type& precision) {
            if (config.inferencePrecision == ov::element::bf16 && precision == ov::element::f32)
                precisions.push_back(ov::element::bf16);
            else
                precisions.push_back(precision);
        };
        push_precision(n->get_input_element_type(0));
        push_precision(n->get_input_element_type(1));
        push_precision(n->get_output_element_type(0));
        if (ov::intel_cpu::tpp::pass::BrgemmToBrgemmTPP::is_supported_brgemm_configuration(layouts, precisions))
            return false;
#endif
        return true;
    };

    CPU_SET_CALLBACK_COMMON(snippetsManager,
    [&mm_supports_transpose_b](const std::shared_ptr<const ov::Node>& n) {
        return mm_supports_transpose_b(n);
    },
    snippets::pass::ExplicitTransposeMatMulInputs);

    snippetsManager.run_passes(model);
}

void Transformations::PostSnippets(void) {
    ov::pass::Manager postSnippetsManager("CPU:PostSnippets");
    postSnippetsManager.set_per_pass_validation(false);
    CPU_REGISTER_PASS_COMMON(postSnippetsManager, ov::pass::FakeQuantizeDecomposition);
    CPU_SET_CALLBACK_COMMON(postSnippetsManager,
        [](const_node_ptr& node) -> bool {
            std::string errMsg;
            return node::FakeQuantize::isSupportedOperation(node, errMsg);
        },
        ov::pass::FakeQuantizeDecomposition);
    CPU_REGISTER_PASS_COMMON(postSnippetsManager, ov::pass::ConstantFolding);
    postSnippetsManager.run_passes(model);
}

void Transformations::Snippets(void) {
    const bool useSnippets = config.snippetsMode != Config::SnippetsMode::Disable &&
        CPU_DEBUG_CAP_IS_TRANSFORMATION_ENABLED(config.debugCaps, Snippets);
    if (!useSnippets)
        return;

    CPU_DEBUG_CAP_TRANSFORMATION_SCOPE(this, Snippets);
    MainSnippets();
    PostSnippets();
}

}   // namespace intel_cpu
}   // namespace ov
