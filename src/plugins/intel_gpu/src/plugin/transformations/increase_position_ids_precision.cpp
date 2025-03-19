// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "increase_position_ids_precision.hpp"

#include "intel_gpu/op/gemm.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

IncreasePositionIdsPrecision::IncreasePositionIdsPrecision() {
    add_matcher<IncreasePositionIdsPrecisionLRoPEMatcher>();
    add_matcher<IncreasePositionIdsPrecisionMatcher>();
}

IncreasePositionIdsPrecisionMatcher::IncreasePositionIdsPrecisionMatcher() {
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

    auto cos_reshape = wrap_type<ov::op::v1::Reshape>({cos, wrap_type<ov::op::v0::Constant>()});
    auto cos_squeeze = wrap_type<ov::op::v0::Squeeze>({cos, wrap_type<ov::op::v0::Constant>()});
    auto cos_unsqueeze = wrap_type<ov::op::v0::Unsqueeze>({cos, wrap_type<ov::op::v0::Constant>()});

    auto rope_sin_input = std::make_shared<Or>(OutputVector{sin_reshape, sin_squeeze, sin_unsqueeze, sin});
    auto rope_cos_input = std::make_shared<Or>(OutputVector{cos_reshape, cos_squeeze, cos_unsqueeze, cos});

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

        // Insert converts before if needed
        auto input_idx = 0;
        auto insert_converts_before_if_needed = [&](const std::shared_ptr<Node>& node) {
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
        };

        // Insert converts after if needed
        auto output_idx = 0;
        auto insert_converts_after_if_needed = [&](const std::shared_ptr<Node>& node) {
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
        };

        bool is_changed = insert_converts_before_if_needed(matmul_node);

        if (is_changed) {
            insert_converts_after_if_needed(cos_node);
            insert_converts_after_if_needed(sin_node);
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(rope, "IncreasePositionIdsPrecisionMatcher");
    this->register_matcher(m, callback);
}

IncreasePositionIdsPrecisionLRoPEMatcher::IncreasePositionIdsPrecisionLRoPEMatcher() {
    using namespace ov::pass::pattern;

    auto data_const_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto concat_list_m = wrap_type<ov::op::v0::Concat>({any_input(), any_input(), any_input()}, type_matches(element::i32));
    auto broadcast_m = wrap_type<ov::op::v3::Broadcast>({data_const_m, concat_list_m}, type_matches(element::f16));
    auto reshape_m = wrap_type<ov::op::v1::Reshape>({any_input(), wrap_type<ov::op::v0::Constant>()}, type_matches(element::i32));
    auto convert_m = wrap_type<ov::op::v0::Convert>({reshape_m}, type_matches(element::f16));
    auto gemm_m = wrap_type<ov::op::v0::MatMul>({broadcast_m, convert_m});
    auto transpose_m = wrap_type<ov::op::v1::Reshape>({gemm_m, wrap_type<ov::op::v0::Constant>()}, type_matches(element::f16));
    auto concat_m = wrap_type<ov::op::v0::Concat>({transpose_m, transpose_m}, type_matches(element::f16));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto concat = ov::as_type_ptr<ov::op::v0::Concat>(m.get_match_root());
        if (!concat || transformation_callback(concat))
            return false;

        const auto desired_et = ov::element::f32;
        const auto original_et = concat->get_output_element_type(0);
        if (original_et == desired_et)
            return false;

        const auto& pattern_map = m.get_pattern_value_map();
        const auto& data_const = pattern_map.at(data_const_m).get_node_shared_ptr();
        auto broadcast = ov::as_type_ptr<ov::op::v3::Broadcast>(pattern_map.at(broadcast_m).get_node_shared_ptr());
        auto convert =  ov::as_type_ptr<ov::op::v0::Convert>(pattern_map.at(convert_m).get_node_shared_ptr());
        auto data_const_convert = std::make_shared<ov::op::v0::Convert>(data_const, desired_et);
        auto gemm =  ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(gemm_m).get_node_shared_ptr());
        auto transpose =  ov::as_type_ptr<ov::op::v1::Reshape>(pattern_map.at(transpose_m).get_node_shared_ptr());

        data_const_convert->set_friendly_name(data_const->get_friendly_name() + "_increase_precision");
        broadcast->input(0).replace_source_output(data_const_convert);
        broadcast->set_output_type(0, desired_et, broadcast->get_output_partial_shape(0));
        convert->set_destination_type(desired_et);
        gemm->set_output_type(0, desired_et, gemm->get_output_partial_shape(0));
        transpose->set_output_type(0, desired_et, transpose->get_output_partial_shape(0));
        concat->set_output_type(0, desired_et, concat->get_output_partial_shape(0));

        for (auto user : concat->get_users()) {
            if (auto cos = ov::as_type_ptr<ov::op::v0::Cos>(user)) {
                auto target_inputs = cos->get_output_target_inputs(0);
                auto cos_convert = std::make_shared<ov::op::v0::Convert>(cos, original_et);
                cos_convert->set_friendly_name(cos->get_friendly_name() + "_restore_precision");
                ov::copy_runtime_info(cos, cos_convert);
                for (auto& in : target_inputs) {
                    in.replace_source_output(cos_convert);
                }
            }
            if (auto sin = ov::as_type_ptr<ov::op::v0::Sin>(user)) {
                auto target_inputs = sin->get_output_target_inputs(0);
                auto sin_convert = std::make_shared<ov::op::v0::Convert>(sin, original_et);
                sin_convert->set_friendly_name(sin->get_friendly_name() + "_restore_precision");
                ov::copy_runtime_info(sin, sin_convert);
                for (auto& in : target_inputs) {
                    in.replace_source_output(sin_convert);
                }
            }
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(concat_m, "IncreasePositionIdsPrecisionLRoPEMatcher");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
