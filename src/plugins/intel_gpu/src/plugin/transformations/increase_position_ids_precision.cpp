// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "increase_position_ids_precision.hpp"

#include "intel_gpu/op/gemm.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

IncreasePositionIdsPrecision::IncreasePositionIdsPrecision() {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;

    auto gemm_or_matmul = wrap_type<ov::intel_gpu::op::Gemm, ov::op::v0::MatMul>();
    auto concat = wrap_type<ov::op::v0::Concat>({gemm_or_matmul, gemm_or_matmul});
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

        auto matmul_node = std::dynamic_pointer_cast<ov::op::v0::MatMul>(pattern_map.at(gemm_or_matmul).get_node_shared_ptr());
        auto cos_node = std::dynamic_pointer_cast<ov::op::v0::Cos>(pattern_map.at(cos).get_node_shared_ptr());
        auto sin_node = std::dynamic_pointer_cast<ov::op::v0::Sin>(pattern_map.at(sin).get_node_shared_ptr());

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

                auto in_convert = std::dynamic_pointer_cast<ov::op::v0::Convert>(incoming_node);

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

    auto m = std::make_shared<ov::pass::pattern::Matcher>(rope, "IncreasePositionIdsPrecision");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
