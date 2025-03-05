// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "swap_mul_transpose.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/core/type.hpp"

#include <memory>

namespace ov::intel_gpu {

SwapMulTranspose::SwapMulTranspose() {
    using namespace ov::op;
    using namespace ov::pass::pattern;
    using namespace ov::pass::pattern::op;

    auto sdpa_without_attn_mask_m = wrap_type<v13::ScaledDotProductAttention>({ any_input(), any_input(), any_input() });
    auto sdpa_with_attn_mask_m = wrap_type<v13::ScaledDotProductAttention>({ any_input(), any_input(), any_input(), any_input() });
    auto sdpa_with_attn_mask_and_scale_m =
        wrap_type<v13::ScaledDotProductAttention>({ any_input(), any_input(), any_input(), any_input(), any_input() });
    auto sdpa_m = std::make_shared<Or>(OutputVector{sdpa_without_attn_mask_m, sdpa_with_attn_mask_m, sdpa_with_attn_mask_and_scale_m});

    auto scale_const_m = wrap_type<ov::op::v0::Constant>();
    auto multiply_m = wrap_type<v1::Multiply>({sdpa_m, scale_const_m});

    auto transpose_order_m = wrap_type<ov::op::v0::Constant>();
    auto transpose_m = wrap_type<v1::Transpose>({multiply_m, transpose_order_m});


    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto transpose = ov::as_type_ptr<v1::Transpose>(pattern_map.at(transpose_m).get_node_shared_ptr());
        if (transformation_callback(transpose)) {
            return false;
        }
        std::cout << "SwapMulTranspose: " << transpose->get_friendly_name() << std::endl;

        auto multiply = ov::as_type_ptr<v1::Multiply>(pattern_map.at(multiply_m).get_node_shared_ptr());

        auto sdpa_out = multiply->input(0).get_source_output();
        transpose->input(0).replace_source_output(sdpa_out);
        for (auto& target : transpose->get_output_target_inputs(0)) {
            target.replace_source_output(multiply->output(0));
        }
        multiply->input(0).replace_source_output(transpose->output(0));

        // auto matmul = pattern_map.at(matmul_m).get_node_shared_ptr();

        // auto min = static_cast<double>(std::numeric_limits<ov::float16>::lowest());
        // auto max = static_cast<double>(std::numeric_limits<ov::float16>::max());
        // auto clamp = std::make_shared<v0::Clamp>(softmax->get_input_source_output(0), min, max);
        // clamp->set_friendly_name(matmul->get_friendly_name() + "/ClampFP16Output");
        // ov::copy_runtime_info({matmul, softmax}, clamp);

        // softmax->input(0).replace_source_output(clamp);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose_m, "SwapMulTranspose");
    this->register_matcher(m, callback);
}

VariadicSplitMulFusion::VariadicSplitMulFusion() {
    using namespace ov::op;
    using namespace ov::pass::pattern;
    using namespace ov::pass::pattern::op;

    const auto is_scalar_const = [](const ov::Output<ov::Node>& output) -> bool {
        if (!ov::is_type<ov::op::v0::Constant>(output.get_node()))
            return false;
        const auto shape = output.get_partial_shape();
        if (shape.is_dynamic())
            return false;
        return ov::shape_size(shape.to_shape()) == 1;
    };

    auto variadic_split_m = wrap_type<v1::VariadicSplit>({any_input(), any_input(), any_input()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto variadic_split = ov::as_type_ptr<v1::VariadicSplit>(pattern_map.at(variadic_split_m).get_node_shared_ptr());
        
        if (transformation_callback(variadic_split)) {
            return false;
        }

        std::vector<float> const_values;
        bool can_be_merged = true;
        std::shared_ptr<ov::op::v0::Constant> const_node = nullptr;
        for (auto& output : variadic_split->outputs()) {
            if (output.get_target_inputs().size() != 1) {
                can_be_merged = false;
                break;
            }
            auto target_node = output.get_target_inputs().begin()->get_node();
            if (!ov::is_type<ov::op::v1::Multiply>(target_node)) {
                can_be_merged = false;
                break;
            }

            for (auto& input : target_node->inputs()) {
                if (input.get_source_output() != output) {
                    if (is_scalar_const(input.get_source_output())) {
                        const_node = ov::as_type_ptr<ov::op::v0::Constant>(
                            input.get_source_output().get_node_shared_ptr());
                        const_values.emplace_back(const_node->cast_vector<float>()[0]);
                    } else {
                        can_be_merged = false;
                        break;
                    }
                }
            }
        }

        if (!const_values.empty() &&
            !std::equal(const_values.begin() + 1, const_values.end(), const_values.begin())) {
            can_be_merged = false;
        }

        if (can_be_merged) {
            std::cout << "VariadicSplitMulFusion: " << variadic_split->get_friendly_name() << std::endl;

            auto new_mul = std::make_shared<ov::op::v1::Multiply>(
                variadic_split->input(0).get_source_output(), const_node);
            new_mul->set_friendly_name(variadic_split->get_friendly_name() + "_mul");
            ov::NodeVector fused_mul_nodes;
            variadic_split->input(0).replace_source_output(new_mul);
            for (auto& output : variadic_split->outputs()) {
                auto target_node = output.get_target_inputs().begin()->get_node();
                fused_mul_nodes.push_back(target_node->shared_from_this());
                ov::replace_output_update_name(target_node->output(0), output);
            }
            ov::copy_runtime_info(fused_mul_nodes, new_mul);
        }

        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(variadic_split_m, "VariadicSplitMulFusion");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
