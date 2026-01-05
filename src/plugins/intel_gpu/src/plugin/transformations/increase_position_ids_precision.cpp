// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "increase_position_ids_precision.hpp"

#include "intel_gpu/op/gemm.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/variadic_split.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/core/graph_util.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

namespace ov::intel_gpu {

IncreasePositionIdsPrecisionForRoPE::IncreasePositionIdsPrecisionForRoPE() {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;

    auto gemm_or_matmul = wrap_type<ov::intel_gpu::op::Gemm, ov::op::v0::MatMul>();
    auto transpose_m = wrap_type<ov::op::v1::Transpose>({gemm_or_matmul, any_input()});
    auto reshape_m = wrap_type<ov::op::v1::Reshape>({gemm_or_matmul, any_input()});
    auto concat_input = std::make_shared<Or>(OutputVector{gemm_or_matmul, transpose_m, reshape_m});
    auto concat = wrap_type<ov::op::v0::Concat>({concat_input, concat_input});
    auto sin = wrap_type<ov::op::v0::Sin>({concat});
    auto cos = wrap_type<ov::op::v0::Cos>({concat});

    auto sin_reshape = wrap_type<ov::op::v1::Reshape>({sin, wrap_type<ov::op::v0::Constant>()});
    auto sin_squeeze = wrap_type<ov::op::v0::Squeeze>({sin, wrap_type<ov::op::v0::Constant>()});
    auto sin_unsqueeze = wrap_type<ov::op::v0::Unsqueeze>({sin, wrap_type<ov::op::v0::Constant>()});
    // Adjust scale factor to positional embedding for LongRoPE
    auto sin_multiply = wrap_type<ov::op::v1::Multiply>({sin, wrap_type<ov::op::v0::Constant>()});
    auto sin_multiply_reshape = wrap_type<ov::op::v1::Reshape>({sin_multiply, wrap_type<ov::op::v0::Constant>()});

    auto cos_reshape = wrap_type<ov::op::v1::Reshape>({cos, wrap_type<ov::op::v0::Constant>()});
    auto cos_squeeze = wrap_type<ov::op::v0::Squeeze>({cos, wrap_type<ov::op::v0::Constant>()});
    auto cos_unsqueeze = wrap_type<ov::op::v0::Unsqueeze>({cos, wrap_type<ov::op::v0::Constant>()});
    // Adjust scale factor to positional embedding for LongRoPE
    auto cos_multiply = wrap_type<ov::op::v1::Multiply>({cos, wrap_type<ov::op::v0::Constant>()});
    auto cos_multiply_reshape = wrap_type<ov::op::v1::Reshape>({cos_multiply, wrap_type<ov::op::v0::Constant>()});

    auto sin_slice = wrap_type<ov::op::v1::StridedSlice>({sin, any_input(), any_input(), any_input()});
    auto sin_gather = wrap_type<ov::op::v8::Gather>({sin_slice, any_input(), any_input()});
    auto sin_unsqueeze2 = wrap_type<ov::op::v0::Unsqueeze>({sin_gather, wrap_type<ov::op::v0::Constant>()});

    auto cos_slice = wrap_type<ov::op::v1::StridedSlice>({cos, any_input(), any_input(), any_input()});
    auto cos_gather = wrap_type<ov::op::v8::Gather>({cos_slice, any_input(), any_input()});
    auto cos_unsqueeze2 = wrap_type<ov::op::v0::Unsqueeze>({cos_gather, wrap_type<ov::op::v0::Constant>()});

    auto rope_sin_input = std::make_shared<Or>(OutputVector{sin_reshape, sin_squeeze, sin_unsqueeze, sin_unsqueeze2, sin_multiply_reshape, sin});
    auto rope_cos_input = std::make_shared<Or>(OutputVector{cos_reshape, cos_squeeze, cos_unsqueeze, cos_unsqueeze2, cos_multiply_reshape, cos});

    auto rope = wrap_type<ov::op::internal::RoPE>({any_input(), rope_cos_input, rope_sin_input});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto matmul_node = ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(gemm_or_matmul).get_node_shared_ptr());
        auto cos_node = ov::as_type_ptr<ov::op::v0::Cos>(pattern_map.at(cos).get_node_shared_ptr());
        auto sin_node = ov::as_type_ptr<ov::op::v0::Sin>(pattern_map.at(sin).get_node_shared_ptr());

        if (!matmul_node || transformation_callback(matmul_node))
            return false;

        const auto desired_et = ov::element::f32;
        const auto original_et = matmul_node->get_output_element_type(0);
        if (original_et == desired_et)
            return false;

        size_t input_idx = 0;
        bool is_changed = insert_converts_before_if_needed(matmul_node, desired_et, input_idx);

        if (is_changed) {
            size_t output_idx = 0;
            insert_converts_after_if_needed(cos_node, original_et, output_idx);
            insert_converts_after_if_needed(sin_node, original_et, output_idx);
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(rope, "IncreasePositionIdsPrecisionForRoPE");
    this->register_matcher(m, callback);
}

IncreasePositionIdsPrecisionForQwen25VL::IncreasePositionIdsPrecisionForQwen25VL() {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;

    // Qwen2.5-VL RoPE pattern:
    // position_ids -> Convert_to_i32 -> Unsqueeze -> Convert_to_f16 -> MatMul -> Transpose -> Concat -> Sin/Cos -> ... -> RoPE
    auto position_ids = any_input();
    auto convert_to_i32 = wrap_type<ov::op::v0::Convert>({position_ids});
    auto unsqueeze = wrap_type<ov::op::v0::Unsqueeze>({convert_to_i32, any_input()});
    auto convert_to_f16 = wrap_type<ov::op::v0::Convert>({unsqueeze});

    auto broadcast_freq = wrap_type<ov::op::v3::Broadcast>({any_input(), any_input()});
    auto matmul = wrap_type<ov::op::v0::MatMul>({broadcast_freq, convert_to_f16});
    auto transpose = wrap_type<ov::op::v1::Transpose>({matmul, any_input()});
    auto concat = wrap_type<ov::op::v0::Concat>({transpose, transpose});

    auto sin = wrap_type<ov::op::v0::Sin>({concat});
    auto cos = wrap_type<ov::op::v0::Cos>({concat});
    auto sin_split = wrap_type<ov::op::v1::VariadicSplit>({sin, wrap_type<ov::op::v0::Constant>(), wrap_type<ov::op::v0::Constant>()});
    auto cos_split = wrap_type<ov::op::v1::VariadicSplit>({cos, wrap_type<ov::op::v0::Constant>(), wrap_type<ov::op::v0::Constant>()});
    auto sin_gather = wrap_type<ov::op::v8::Gather>({sin_split, wrap_type<ov::op::v0::Constant>(), wrap_type<ov::op::v0::Constant>()});
    auto cos_gather = wrap_type<ov::op::v8::Gather>({cos_split, wrap_type<ov::op::v0::Constant>(), wrap_type<ov::op::v0::Constant>()});
    auto sin_concat = wrap_type<ov::op::v0::Concat>({sin_gather, sin_gather, sin_gather, sin_gather, sin_gather, sin_gather});
    auto cos_concat = wrap_type<ov::op::v0::Concat>({cos_gather, cos_gather, cos_gather, cos_gather, cos_gather, cos_gather});
    auto sin_unsequeeze = wrap_type<ov::op::v0::Unsqueeze>({sin_concat, wrap_type<ov::op::v0::Constant>()});
    auto cos_unsqueeze = wrap_type<ov::op::v0::Unsqueeze>({cos_concat, wrap_type<ov::op::v0::Constant>()});
    auto rope = wrap_type<ov::op::internal::RoPE>({any_input(), cos_unsqueeze, sin_unsequeeze});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto convert_node = ov::as_type_ptr<ov::op::v0::Convert>(pattern_map.at(convert_to_f16).get_node_shared_ptr());
        auto broadcast_node = pattern_map.at(broadcast_freq).get_node_shared_ptr();
        auto matmul_node = ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(matmul).get_node_shared_ptr());
        auto sin_node = ov::as_type_ptr<ov::op::v0::Sin>(pattern_map.at(sin).get_node_shared_ptr());
        auto cos_node = ov::as_type_ptr<ov::op::v0::Cos>(pattern_map.at(cos).get_node_shared_ptr());

        if (!convert_node || !matmul_node || transformation_callback(convert_node))
            return false;

        const auto desired_et = ov::element::f32;
        const auto original_et = convert_node->get_output_element_type(0);
        if (original_et == desired_et)
            return false;

        // Check if input is integer type (position_ids should be i32 or i64)
        auto input_et = convert_node->input_value(0).get_element_type();
        if (!input_et.is_integral())
            return false;

        // 1. Change Convert output from f16 to f32 (position_ids path)
        auto new_convert = std::make_shared<ov::op::v0::Convert>(convert_node->input_value(0), desired_et);
        new_convert->set_friendly_name(convert_node->get_friendly_name() + "_increase_precision");
        copy_runtime_info(convert_node, new_convert);
        ov::replace_node(convert_node, new_convert);

        // 2. Insert Convert(f16->f32) after Broadcast (freq path) to match MatMul types
        //    MatMul needs both inputs to be the same type (f32)
        if (broadcast_node->get_output_element_type(0) != desired_et) {
            auto broadcast_to_f32 = std::make_shared<ov::op::v0::Convert>(broadcast_node->output(0), desired_et);
            broadcast_to_f32->set_friendly_name(broadcast_node->get_friendly_name() + "_to_f32");
            copy_runtime_info(broadcast_node, broadcast_to_f32);
            // Replace MatMul's input 0 (Broadcast) with the new Convert
            matmul_node->input(0).replace_source_output(broadcast_to_f32->output(0));
        }

        // 3. Insert Convert(f32->f16) after Sin/Cos to restore original precision
        size_t output_idx = 0;
        insert_converts_after_if_needed(sin_node, original_et, output_idx);
        insert_converts_after_if_needed(cos_node, original_et, output_idx);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(rope, "IncreasePositionIdsPrecisionForQwen25VL");
    this->register_matcher(m, callback);
}

IncreasePositionIdsPrecisionForLtxVideo::IncreasePositionIdsPrecisionForLtxVideo() {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;

    // for ltx-video pattern
    auto mul = wrap_type<ov::op::v1::Multiply>({any_input(), any_input()});
    auto add_constant = wrap_type<ov::op::v0::Constant>();
    auto add = wrap_type<ov::op::v1::Add>({mul, add_constant});
    auto transpose = wrap_type<ov::op::v1::Transpose>({add, any_input()});
    auto reshape = wrap_type<ov::op::v1::Reshape>({transpose, any_input()});
    auto sin = wrap_type<ov::op::v0::Sin>({reshape});
    auto cos = wrap_type<ov::op::v0::Cos>({reshape});
    auto gather_1 = wrap_type<ov::op::v8::Gather>({cos, any_input(), {-1}}, {{"batch_dims", 0}});
    auto gather_3 = wrap_type<ov::op::v8::Gather>({sin, any_input(), {-1}}, {{"batch_dims", 0}});
    auto slice = wrap_type<ov::op::v1::StridedSlice>({gather_1, {0, 0, 0}, {0, 0, 2}, {1, 1, 1}});
    auto shape_of = wrap_type<ov::op::v3::ShapeOf>({slice});
    auto broadcast_zero_like = wrap_type<ov::op::v3::Broadcast>({{0}, shape_of});
    auto broadcast_ones_like = wrap_type<ov::op::v3::Broadcast>({{1}, shape_of});
    auto concat = wrap_type<ov::op::v0::Concat>({broadcast_ones_like, gather_1});
    auto concat_1 = wrap_type<ov::op::v0::Concat>({broadcast_zero_like, gather_3});
    auto rms = wrap_type<ov::op::internal::RMS>({any_input(), any_input()});
    auto mul_2 = wrap_type<ov::op::v1::Multiply>({rms, concat});
    auto reshape_2 = wrap_type<ov::op::v1::Reshape>({any_input(), any_input()});
    auto mul_3 = wrap_type<ov::op::v1::Multiply>({reshape_2, concat_1});
    auto add_1 = wrap_type<ov::op::v1::Add>({mul_2, mul_3});
    auto reshape_3 = wrap_type<ov::op::v1::Reshape>({add_1, any_input()});
    auto tranpose_1 = wrap_type<ov::op::v1::Transpose>({reshape_3, any_input()});
    auto sdpa = wrap_type<ov::op::v13::ScaledDotProductAttention>({any_input(), tranpose_1, any_input()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto mul_node = ov::as_type_ptr<ov::op::v1::Multiply>(pattern_map.at(mul).get_node_shared_ptr());
        auto constant_node = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(add_constant).get_node_shared_ptr());
        auto cos_node = pattern_map.count(cos) > 0 ?
                            ov::as_type_ptr<ov::op::v0::Cos>(pattern_map.at(cos).get_node_shared_ptr())
                            : nullptr;
        auto sin_node = pattern_map.count(sin) > 0 ?
                            ov::as_type_ptr<ov::op::v0::Sin>(pattern_map.at(sin).get_node_shared_ptr())
                            : nullptr;

        if (!mul_node || transformation_callback(mul_node))
            return false;

        const auto desired_et = ov::element::f32;
        const auto original_et = mul_node->get_output_element_type(0);
        if (original_et == desired_et)
            return false;

        size_t input_idx = 0;
        bool is_changed = insert_converts_before_if_needed(mul_node, desired_et, input_idx);
        if (is_changed) {
            if (constant_node)
                insert_converts_after_if_needed(constant_node, desired_et, input_idx);
            size_t output_idx = 0;
            if (cos_node)
                insert_converts_after_if_needed(cos_node, original_et, output_idx);
            if (sin_node)
                insert_converts_after_if_needed(sin_node, original_et, output_idx);
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sdpa, "IncreasePositionIdsPrecisionForLtxVideo");
    this->register_matcher(m, callback);
}

IncreasePositionIdsPrecisionForGPTOSS::IncreasePositionIdsPrecisionForGPTOSS() {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;

    auto broadcast_freq = wrap_type<ov::op::v3::Broadcast>({any_input(), any_input()});
    auto convert_broadcast_freq = wrap_type<ov::op::v0::Convert>({broadcast_freq});

    auto convert_pos_id_to_i32 = wrap_type<ov::op::v0::Convert>({any_input()});
    auto unsqueeze_pos_id_1 = wrap_type<ov::op::v0::Unsqueeze>({convert_pos_id_to_i32, any_input()});
    auto unsqueeze_pos_id_2 = wrap_type<ov::op::v1::Reshape>({convert_pos_id_to_i32, any_input()});
    auto unsqueeze_pos_id = std::make_shared<Or>(OutputVector{unsqueeze_pos_id_1, unsqueeze_pos_id_2});
    auto convert_pos_id_to_f16 = wrap_type<ov::op::v0::Convert>({unsqueeze_pos_id});

    auto broadcast_freq_ = std::make_shared<Or>(OutputVector{broadcast_freq, convert_broadcast_freq});

    auto matmul_freq_pos_id = wrap_type<ov::op::v0::MatMul>({broadcast_freq_, convert_pos_id_to_f16});
    auto reshape_matmul = wrap_type<ov::op::v1::Reshape>({matmul_freq_pos_id, any_input()});
    auto transpose_matmul = wrap_type<ov::op::v1::Transpose>({matmul_freq_pos_id, any_input()});
    auto transpose_or_reshape = std::make_shared<Or>(OutputVector{transpose_matmul, reshape_matmul});

    auto sin_ = wrap_type<ov::op::v0::Sin>({transpose_or_reshape});
    auto sin_convert = wrap_type<ov::op::v0::Convert>({sin_});
    auto sin = std::make_shared<Or>(OutputVector{sin_, sin_convert});

    auto cos = wrap_type<ov::op::v0::Cos>({transpose_or_reshape});
    auto cos_convert = wrap_type<ov::op::v0::Convert>({cos});
    auto cos_ = std::make_shared<Or>(OutputVector{cos, cos_convert});

    auto scale_const_sin = wrap_type<ov::op::v0::Constant>();
    auto scale_const_sin_convert = wrap_type<ov::op::v0::Convert>({scale_const_sin});
    auto scale_const_sin_ = std::make_shared<Or>(OutputVector{scale_const_sin, scale_const_sin_convert});
    auto mul_sin_scale = wrap_type<ov::op::v1::Multiply>({sin, scale_const_sin_});

    auto scale_const_cos = wrap_type<ov::op::v0::Constant>();
    auto scale_const_cos_convert = wrap_type<ov::op::v0::Convert>({scale_const_cos});
    auto scale_const_cos_ = std::make_shared<Or>(OutputVector{scale_const_cos, scale_const_cos_convert});
    auto mul_cos_scale = wrap_type<ov::op::v1::Multiply>({cos_, scale_const_cos_});

    auto unsqueeze_mul_sin_scale_ = wrap_type<ov::op::v0::Unsqueeze>({mul_sin_scale, any_input()});
    auto reshape_mul_sin_scale_ = wrap_type<ov::op::v1::Reshape>({mul_sin_scale, any_input()});
    auto reshape_mul_sin_scale = std::make_shared<Or>(OutputVector{unsqueeze_mul_sin_scale_, reshape_mul_sin_scale_});

    auto unsqueeze_mul_cos_scale_ = wrap_type<ov::op::v0::Unsqueeze>({mul_cos_scale, any_input()});
    auto reshape_mul_cos_scale_ = wrap_type<ov::op::v1::Reshape>({mul_cos_scale, any_input()});
    auto reshape_mul_cos_scale = std::make_shared<Or>(OutputVector{unsqueeze_mul_cos_scale_, reshape_mul_cos_scale_});

    auto rope_qk = wrap_type<ov::op::internal::RoPE>({any_input(), reshape_mul_cos_scale, reshape_mul_sin_scale});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        if (transformation_callback(rope_qk))
            return false;

        auto rope_node = ov::as_type_ptr<ov::op::internal::RoPE>(pattern_map.at(rope_qk).get_node_shared_ptr());

        auto matmul_node = ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(matmul_freq_pos_id).get_node_shared_ptr());
        auto mul_node1 = ov::as_type_ptr<ov::op::v1::Multiply>(pattern_map.at(mul_sin_scale).get_node_shared_ptr());
        auto mul_node2 = ov::as_type_ptr<ov::op::v1::Multiply>(pattern_map.at(mul_cos_scale).get_node_shared_ptr());

        const auto desired_et = ov::element::f32;
        const auto original_et = rope_node->get_output_element_type(0);
        if (original_et == desired_et)
            return false;

        size_t idx = 0;
        insert_converts_before_if_needed(matmul_node, desired_et, idx);
        insert_converts_before_if_needed(mul_node1, desired_et, idx);
        insert_converts_before_if_needed(mul_node2, desired_et, idx);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(rope_qk, "IncreasePositionIdsPrecisionForGPTOSS");
    this->register_matcher(m, callback);
}

bool IncreasePositionIdsPrecisionForRoPE::insert_converts_before_if_needed(const std::shared_ptr<ov::Node>& node,
                                                                const ov::element::Type desired_et, size_t& input_idx) {
    bool is_changed = false;
    for (const auto& input : node->inputs()) {
        const auto& incoming_output = input.get_source_output();
        const auto& incoming_node = incoming_output.get_node_shared_ptr();
        const auto input_et = incoming_output.get_element_type();

        if (input_et == desired_et)
            continue;

        auto in_convert = ov::as_type_ptr<ov::op::v0::Convert>(incoming_node);

        if (in_convert && in_convert->get_users().size() == 1 && input_et.bitwidth() <= desired_et.bitwidth()) {
            auto convert = std::make_shared<ov::op::v0::Convert>(incoming_node->input_value(0), desired_et);
            convert->set_friendly_name(in_convert->get_friendly_name() + "_increase_precision_" + std::to_string(input_idx));
            copy_runtime_info(incoming_node, convert);
            ov::replace_node(incoming_node, convert);
        } else {
            auto convert = std::make_shared<ov::op::v0::Convert>(incoming_output, desired_et);
            convert->set_friendly_name(incoming_node->get_friendly_name() + "_increase_precision_" + std::to_string(input_idx));
            copy_runtime_info(incoming_node, convert);
            input.replace_source_output(convert);
        }

        input_idx++;
        is_changed = true;
    }

    return is_changed;
}

void IncreasePositionIdsPrecisionForRoPE::insert_converts_after_if_needed(const std::shared_ptr<ov::Node>& node,
                                                            const ov::element::Type original_et, size_t& output_idx) {
    for (const auto& output : node->outputs()) {
        for (const auto& out_inputs : output.get_target_inputs()) {
            auto out_node = out_inputs.get_node()->shared_from_this();

            auto convert = std::make_shared<ov::op::v0::Convert>(output, original_et);
            auto convert_name = out_node->get_friendly_name() + "_restore_precision_" + std::to_string(output_idx);
            convert->set_friendly_name(convert_name);
            copy_runtime_info(node, convert);
            out_inputs.replace_source_output(convert);
            output_idx++;
        }
    }
}

IncreasePositionIdsPrecision::IncreasePositionIdsPrecision() {}

bool IncreasePositionIdsPrecision::run_on_model(const std::shared_ptr<ov::Model>& model) {
    ov::pass::SymbolicOptimizations symbolic_optimizations(false, get_pass_config());
    auto symbolic_ctx_manager = symbolic_optimizations.get_manager();
    symbolic_ctx_manager->register_pass<IncreasePositionIdsPrecisionForRoPE>();
    symbolic_ctx_manager->register_pass<IncreasePositionIdsPrecisionForQwen25VL>();
    symbolic_ctx_manager->register_pass<IncreasePositionIdsPrecisionForLtxVideo>();
    symbolic_ctx_manager->register_pass<IncreasePositionIdsPrecisionForGPTOSS>();
    return symbolic_optimizations.run_on_model(model);
}

DisableFP16ComForGPTOSSROPEPattern::DisableFP16ComForGPTOSSROPEPattern() {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;

    // for gpt-oss pattern
    auto freq_const = wrap_type<ov::op::v0::Constant>();
    auto broadcast_freq = wrap_type<ov::op::v3::Broadcast>({freq_const, any_input()});

    // position_id
    auto unsqueeze_pos_id = wrap_type<ov::op::v0::Unsqueeze>({any_input(), any_input()});
    auto convert_pos_id_to_f16 = wrap_type<ov::op::v0::Convert>({unsqueeze_pos_id});

    auto matmul_freq_pos_id = wrap_type<ov::op::v0::MatMul>({broadcast_freq, convert_pos_id_to_f16});
    auto transpose = wrap_type<ov::op::v1::Transpose>({matmul_freq_pos_id, any_input()});
    auto sin = wrap_type<ov::op::v0::Sin>({transpose});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto sin_node = ov::as_type_ptr<ov::op::v0::Sin>(pattern_map.at(sin).get_node_shared_ptr());
        if (!sin_node || transformation_callback(sin_node))
            return false;
        auto freq_const_node = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(freq_const).get_node_shared_ptr());
        ov::disable_fp16_compression(freq_const_node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sin, "DisableFP16ComForGPTOSSROPEPattern");
    this->register_matcher(m, callback);
}
}  // namespace ov::intel_gpu
