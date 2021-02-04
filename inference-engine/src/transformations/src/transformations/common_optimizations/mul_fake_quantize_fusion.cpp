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

        auto const_shape = mul_const->get_shape();
        size_t const_shape_size = shape_size(const_shape);
        if (const_shape_size > 1) {
            // disallow constant shapes other than (N, 1, 1, ..., 1) or (1, C, 1, ..., 1)
            if (!(const_shape[0] > 1 && const_shape[0] == const_shape_size) &&
                !(const_shape.size() > 1 && const_shape[1] == const_shape_size)) {
                return false;
            }
        }

        std::shared_ptr<Node> mul_const_node = mul_const;
        if (const_shape_size > 1 &&
            static_cast<Dimension::value_type>(const_shape.size()) < fq->get_input_partial_shape(0).rank().get_length()) {
            // Reshape constants like (C, 1, 1) to (1, C, 1, 1)
            const_shape.insert(const_shape.begin(), fq->get_input_partial_shape(0).rank().get_length() - const_shape.size(), 1);
            mul_const_node = std::make_shared<opset5::Reshape>(mul_const_node,
                    op::Constant::create(element::u64, Shape{const_shape.size()}, const_shape), false);
        }

        auto new_input_low = std::make_shared<opset5::Divide>(fq->input_value(1), mul_const_node);
        auto new_input_high = std::make_shared<opset5::Divide>(fq->input_value(2), mul_const_node);

        std::shared_ptr<Node> new_fq;
        auto mul_const_value = mul_const->cast_vector<float>();
        if (std::all_of(mul_const_value.begin(), mul_const_value.end(), [] (float f) -> bool { return f < 0.0f; })) {
            new_fq = register_new_node<opset5::FakeQuantize>(mul_data, new_input_low, new_input_high,
                    fq->input_value(4), fq->input_value(3), fq->get_levels());
            copy_runtime_info({mul, fq}, {mul_const_node, new_input_low, new_input_high, new_fq});
        } else if (std::any_of(mul_const_value.begin(), mul_const_value.end(), [] (float f) -> bool { return f < 0.0f; })) {
            const auto& output_low = fq->input_value(3);
            const auto& output_high = fq->input_value(4);
            auto zero = op::Constant::create(element::f32, Shape{}, {0.0f});
            // get the mask of the values from mul_const that are less than zero
            // and greater or equal to zero
            std::vector<float> less_than_zero;
            less_than_zero.reserve(mul_const_value.size());
            std::vector<float> greater_eq_zero;
            greater_eq_zero.reserve(mul_const_value.size());
            for (size_t i = 0; i < mul_const_value.size(); i++) {
                less_than_zero.push_back(mul_const_value[i] < 0);
                greater_eq_zero.push_back(mul_const_value[i] >= 0);
            }
            auto less_const = op::Constant::create(element::f32, const_shape, less_than_zero);
            auto greater_eq_const = op::Constant::create(element::f32, const_shape, greater_eq_zero);
            // new_output_low is defined as follows:
            //   output_low[i],  when mul_const[i] >= 0
            //   output_high[i], when mul_const[i] < 0
            auto new_output_low = std::make_shared<opset5::Add>(
                    std::make_shared<opset5::Multiply>(greater_eq_const, output_low),
                    std::make_shared<opset5::Multiply>(less_const, output_high));
            // new_output_high is defined as follows:
            //   output_high[i], when mul_const[i] >= 0
            //   output_low[i],  when mul_const[i] < 0
            auto new_output_high = std::make_shared<opset5::Add>(
                    std::make_shared<opset5::Multiply>(greater_eq_const, output_high),
                    std::make_shared<opset5::Multiply>(less_const, output_low));
            new_fq = register_new_node<opset5::FakeQuantize>(mul_data, new_input_low,
                    new_input_high, new_output_low, new_output_high, fq->get_levels());
        } else {
            new_fq = register_new_node<opset5::FakeQuantize>(mul_data, new_input_low, new_input_high,
                    fq->input_value(3), fq->input_value(4), fq->get_levels());
        }

        copy_runtime_info({mul, fq}, {mul_const_node, new_input_low, new_input_high, new_fq});
        new_fq->set_friendly_name(fq->get_friendly_name());
        replace_node(fq, new_fq);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fq_pattern, matcher_name);
    this->register_matcher(m, callback);
}
