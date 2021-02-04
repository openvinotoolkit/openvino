// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mul_fake_quantize_fusion.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "itt.hpp"


NGRAPH_RTTI_DEFINITION(ngraph::pass::MulFakeQuantizeFusion, "MulFakeQuantizeFusion", 0);

ngraph::pass::MulFakeQuantizeFusion::MulFakeQuantizeFusion() {
    MATCHER_SCOPE(MulFakeQuantizeFusion);
    auto mul_pattern = ngraph::pattern::wrap_type<opset5::Multiply>({ngraph::pattern::any_input(), ngraph::pattern::wrap_type<opset5::Constant>()},
                                                                    pattern::consumers_count(1));
    auto fq_pattern = ngraph::pattern::wrap_type<opset5::FakeQuantize>({mul_pattern,
                                                                        ngraph::pattern::any_input(),
                                                                        ngraph::pattern::any_input(),
                                                                        ngraph::pattern::any_input(),
                                                                        ngraph::pattern::any_input()});
    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto fq = std::dynamic_pointer_cast<opset5::FakeQuantize>(m.get_match_root());
        if (!fq)
            return false;
        auto mul = std::dynamic_pointer_cast<opset5::Multiply>(fq->input_value(0).get_node_shared_ptr());
        if (!mul)
            return false;
        auto mul_data = mul->input_value(0).get_node_shared_ptr();
        auto mul_const = std::dynamic_pointer_cast<opset5::Constant>(mul->input_value(1).get_node_shared_ptr());
        if (!mul_const) {
            mul_const = std::dynamic_pointer_cast<opset5::Constant>(mul->input_value(0).get_node_shared_ptr());
            if (!mul_const)
                return false;
            mul_data = mul->input_value(1).get_node_shared_ptr();
        }

        auto shape = mul_const->get_shape();
        size_t const_shape_size = shape_size(shape);
        if (const_shape_size > 1) {
            // disallow constant shapes other than (N, 1, 1, ..., 1) or (1, C, 1, ..., 1)
            if (!(shape[0] > 1 && shape[0] == const_shape_size) &&
                !(shape.size() > 1 && shape[1] == const_shape_size)) {
                return false;
            }
        }

        std::shared_ptr<Node> mul_const_node = mul_const;
        if (const_shape_size > 1 &&
            static_cast<Dimension::value_type>(shape.size()) < fq->get_input_partial_shape(0).rank().get_length()) {
            // Reshape constants like (C, 1, 1) to (1, C, 1, 1)
            shape.insert(shape.begin(), fq->get_input_partial_shape(0).rank().get_length() - shape.size(), 1);
            mul_const_node = std::make_shared<opset5::Reshape>(mul_const_node, op::Constant::create(element::u64, Shape{shape.size()}, shape), false);
        }

        auto new_input_low = std::make_shared<opset5::Divide>(fq->input_value(1), mul_const_node);
        auto new_input_high = std::make_shared<opset5::Divide>(fq->input_value(2), mul_const_node);
        auto new_output_low = fq->input_value(3).get_node_shared_ptr();
        auto new_output_high = fq->input_value(4).get_node_shared_ptr();

        std::shared_ptr<Node> new_fq;
        auto mul_const_value = mul_const->cast_vector<float>();
        if (std::all_of(mul_const_value.begin(), mul_const_value.end(), [] (float f) -> bool { return f < 0.0f; })) {
            new_output_low = fq->input_value(4).get_node_shared_ptr();
            new_output_high = fq->input_value(3).get_node_shared_ptr();
            new_fq = register_new_node<opset5::FakeQuantize>(mul_data, new_input_low, new_input_high,
                    fq->input_value(4), fq->input_value(3), fq->get_levels());
            copy_runtime_info({mul, fq}, {mul_const_node, new_input_low, new_input_high, new_fq});
        } else if (std::any_of(mul_const_value.begin(), mul_const_value.end(), [] (float f) -> bool { return f < 0.0f; })) {
            auto zero = op::Constant::create(element::f32, Shape{}, {0.0f});
            auto minus_one = op::Constant::create(element::f32, Shape{}, {-1.0f});
            auto less_than_zero = std::make_shared<opset5::Less>(mul_const_node, zero);
            auto less_than_zero_convert = std::make_shared<opset5::Convert>(less_than_zero, element::f32);
            auto greater_eq_zero = std::make_shared<opset5::GreaterEqual>(mul_const_node, zero);
            auto greater_eq_zero_convert = std::make_shared<opset5::Convert>(greater_eq_zero, element::f32);
            auto neg_mask = std::make_shared<opset5::Multiply>(minus_one, less_than_zero_convert);
            auto out_low_times_neg_mask = std::make_shared<opset5::Multiply>(neg_mask, fq->input_value(3));
            auto out_low_times_pos_mask = std::make_shared<opset5::Multiply>(greater_eq_zero_convert, fq->input_value(3));
            auto new_output_low = std::make_shared<opset5::Add>(out_low_times_neg_mask, out_low_times_pos_mask);
            auto out_high_times_neg_mask = std::make_shared<opset5::Multiply>(neg_mask, fq->input_value(4));
            auto out_high_times_pos_mask = std::make_shared<opset5::Multiply>(greater_eq_zero_convert, fq->input_value(4));
            auto new_output_high = std::make_shared<opset5::Add>(out_high_times_neg_mask, out_high_times_pos_mask);
            new_fq = register_new_node<opset5::FakeQuantize>(mul_data, new_input_low,
                    new_input_high, new_output_low, new_output_high, fq->get_levels());
            copy_runtime_info({mul, fq},
                              {mul_const_node, new_input_low, new_input_high,
                               less_than_zero, less_than_zero_convert, greater_eq_zero, greater_eq_zero_convert,
                               neg_mask, out_low_times_neg_mask, out_low_times_pos_mask, new_output_low,
                               out_high_times_neg_mask, out_high_times_pos_mask, new_output_high, new_fq});
        } else {
            new_fq = register_new_node<opset5::FakeQuantize>(mul_data, new_input_low, new_input_high,
                    fq->input_value(3), fq->input_value(4), fq->get_levels());
            copy_runtime_info({mul, fq}, {mul_const_node, new_input_low, new_input_high, new_fq});
        }

        new_fq->set_friendly_name(fq->get_friendly_name());
        replace_node(fq, new_fq);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fq_pattern, matcher_name);
    this->register_matcher(m, callback);
}
