// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mul_fake_quantize_fusion.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/validation_util.hpp>
#include "itt.hpp"


NGRAPH_RTTI_DEFINITION(ngraph::pass::MulFakeQuantizeFusion, "MulFakeQuantizeFusion", 0);

ngraph::pass::MulFakeQuantizeFusion::MulFakeQuantizeFusion() {
    MATCHER_SCOPE(MulFakeQuantizeFusion);
    auto input_pattern = ngraph::pattern::any_input();
    auto const_pattern = ngraph::pattern::wrap_type<opset5::Constant>();
    auto mul_pattern = ngraph::pattern::wrap_type<opset5::Multiply>({input_pattern, const_pattern},
                                                                    pattern::consumers_count(1));
    auto fq_pattern = ngraph::pattern::wrap_type<opset5::FakeQuantize>({mul_pattern,
                                                                        ngraph::pattern::any_input(),
                                                                        ngraph::pattern::any_input(),
                                                                        ngraph::pattern::any_input(),
                                                                        ngraph::pattern::any_input()});
    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_value_map = m.get_pattern_value_map();
        auto fq = std::dynamic_pointer_cast<opset5::FakeQuantize>(pattern_value_map.at(fq_pattern).get_node_shared_ptr());
        if (!fq)
            return false;
        auto mul_const = std::dynamic_pointer_cast<opset5::Constant>(pattern_value_map.at(const_pattern).get_node_shared_ptr());
        if (!mul_const)
            return false;

        auto mul_const_value = mul_const->cast_vector<float>();
        if (std::any_of(mul_const_value.begin(), mul_const_value.end(), [] (float f) -> bool { return f <= 0.0f; }))
            return false;

        std::shared_ptr<Node> new_const = mul_const;
        auto const_shape = mul_const->get_shape();
        size_t const_shape_size = shape_size(const_shape);
        bool is_single_value = const_shape_size == 1;

        if (!is_single_value) {
            float v;
            is_single_value = op::util::get_single_value(mul_const, v);
            if (is_single_value) {
                new_const = std::make_shared<opset5::Constant>(mul_const->get_element_type(), Shape{1}, v);
                const_shape = Shape{1};
            }
        }

        if (!is_single_value) {
            // disallow constant shapes other than (N, 1, 1, ..., 1) or (1, C, 1, ..., 1)
            if (!(const_shape[0] > 1 && const_shape[0] == const_shape_size) &&
                !(const_shape.size() > 1 && const_shape[1] == const_shape_size)) {
                return false;
            }
            const auto& rank = fq->get_input_partial_shape(0).rank();
            if (rank.is_dynamic())
                return false;
            auto fq_users = fq->get_users();
            // Concat LPT transformation supports per tensor quantization only
            bool fq_user_is_concat = std::any_of(fq_users.begin(), fq_users.end(),
                                                 [] (const Output<Node>& node) -> bool {
                                                     auto node_ptr = node.get_node();
                                                     return is_type<opset5::Concat>(node_ptr);
                                                 });
            if (fq_user_is_concat)
                return false;
            auto diff = rank.get_length() - static_cast<Dimension::value_type>(const_shape.size());
            // Reshape constants like (C, 1, 1) to (1, C, 1, 1)
            const_shape.insert(const_shape.begin(), diff, 1);
            new_const = std::make_shared<opset5::Reshape>(new_const,
                    op::Constant::create(element::u64, Shape{const_shape.size()}, const_shape), false);
        }

        bool all_negative = std::all_of(mul_const_value.begin(), mul_const_value.end(), [] (float f) -> bool { return f < 0.0f; });
        bool any_negative = std::any_of(mul_const_value.begin(), mul_const_value.end(), [] (float f) -> bool { return f < 0.0f; });
        if (any_negative && !all_negative) {
            auto fq_outputs = fq->get_users();
            // Convolution and GroupConvolution LP transformations require output low/high to have the same values
            bool fq_output_is_conv = std::any_of(fq_outputs.begin(), fq_outputs.end(),
                                                 [] (const std::shared_ptr<Node>& node) -> bool {
                                                     return is_type<opset5::Convolution>(node) ||
                                                            is_type<opset5::GroupConvolution>(node);
                                                 });
            if (fq_output_is_conv) {
                return false;
            }
        }

        auto input_low_div = std::make_shared<opset5::Divide>(fq->input_value(1), new_const);
        std::shared_ptr<Node> new_input_low = get_constant_from_source(input_low_div);
        if (!new_input_low)
            new_input_low = input_low_div;
        auto input_high_div = std::make_shared<opset5::Divide>(fq->input_value(2), new_const);
        std::shared_ptr<Node> new_input_high = get_constant_from_source(input_high_div);
        if (!new_input_high)
            new_input_high = input_high_div;

        auto mul = pattern_value_map.at(mul_pattern).get_node_shared_ptr();
        const auto& mul_data = pattern_value_map.at(input_pattern);

        std::shared_ptr<Node> new_fq;
        if (all_negative) {
            new_fq = register_new_node<opset5::FakeQuantize>(mul_data, new_input_low, new_input_high,
                    fq->input_value(4), fq->input_value(3), fq->get_levels());
            copy_runtime_info({mul, fq}, {new_const, new_input_low, new_input_high, new_fq});
        } else if (any_negative) {
            const auto& output_low = fq->input_value(3);
            const auto& output_high = fq->input_value(4);
            // get the mask of the values from mul_const that are less than zero
            std::vector<float> less_than_zero;
            const_shape_size = shape_size(const_shape);
            less_than_zero.reserve(const_shape_size);
            // and greater or equal to zero
            std::vector<float> greater_eq_zero;
            greater_eq_zero.reserve(const_shape_size);
            for (size_t i = 0; i < const_shape_size; i++) {
                less_than_zero.push_back(mul_const_value[i] < 0);
                greater_eq_zero.push_back(mul_const_value[i] >= 0);
            }
            auto less_const = op::Constant::create(output_low.get_element_type(), const_shape, less_than_zero);
            auto greater_eq_const = op::Constant::create(output_low.get_element_type(), const_shape, greater_eq_zero);
            // new_output_low is defined as follows:
            //   output_low[i],  when mul_const[i] >= 0
            //   output_high[i], when mul_const[i] < 0
            auto output_low_tmp = std::make_shared<opset5::Add>(
                    std::make_shared<opset5::Multiply>(greater_eq_const, output_low),
                    std::make_shared<opset5::Multiply>(less_const, output_high));
            std::shared_ptr<Node> new_output_low = get_constant_from_source(output_low_tmp);
            if (!new_output_low)
                new_output_low = output_low_tmp;

            // new_output_high is defined as follows:
            //   output_high[i], when mul_const[i] >= 0
            //   output_low[i],  when mul_const[i] < 0
            auto output_high_tmp = std::make_shared<opset5::Add>(
                    std::make_shared<opset5::Multiply>(greater_eq_const, output_high),
                    std::make_shared<opset5::Multiply>(less_const, output_low));
            std::shared_ptr<Node> new_output_high = get_constant_from_source(output_high_tmp);
            if (!new_output_high)
                new_output_high = output_high_tmp;
            new_fq = register_new_node<opset5::FakeQuantize>(mul_data, new_input_low,
                    new_input_high, new_output_low, new_output_high, fq->get_levels());
        } else {
            new_fq = register_new_node<opset5::FakeQuantize>(mul_data, new_input_low, new_input_high,
                    fq->input_value(3), fq->input_value(4), fq->get_levels());
        }

        copy_runtime_info({mul, fq}, {new_const, new_input_low, new_input_high, new_fq});
        new_fq->set_friendly_name(fq->get_friendly_name());
        replace_node(fq, new_fq);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fq_pattern, matcher_name);
    this->register_matcher(m, callback);
}
