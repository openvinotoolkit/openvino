// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mul_fake_quantize_fusion.hpp"

#include <memory>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/validation_util.hpp>
#include <vector>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

ngraph::pass::MulFakeQuantizeFusion::MulFakeQuantizeFusion() {
    MATCHER_SCOPE(MulFakeQuantizeFusion);
    auto input_pattern = ngraph::pattern::any_input();
    auto const_pattern = ngraph::pattern::wrap_type<opset5::Constant>();
    auto mul_pattern =
        ngraph::pattern::wrap_type<opset5::Multiply>({input_pattern, const_pattern}, pattern::consumers_count(1));
    auto fq_pattern = ngraph::pattern::wrap_type<opset5::FakeQuantize>({mul_pattern,
                                                                        ngraph::pattern::any_input(),
                                                                        ngraph::pattern::any_input(),
                                                                        ngraph::pattern::any_input(),
                                                                        ngraph::pattern::any_input()});
    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_value_map = m.get_pattern_value_map();
        const auto& input = pattern_value_map.at(input_pattern);
        const auto& type = input.get_element_type();
        if (type.bitwidth() < element::f32.bitwidth())
            return false;
        auto fq =
            std::dynamic_pointer_cast<opset5::FakeQuantize>(pattern_value_map.at(fq_pattern).get_node_shared_ptr());
        if (!fq)
            return false;
        auto mul_const =
            std::dynamic_pointer_cast<opset5::Constant>(pattern_value_map.at(const_pattern).get_node_shared_ptr());
        if (!mul_const)
            return false;

        auto const_shape = mul_const->get_shape();
        if (ngraph::op::util::check_for_broadcast(input.get_partial_shape(), const_shape)) {
            // We can't eliminate Multiply if Constant input broadcasts another input shape because
            // when we reconnect input from Multiply to FQ won't broadcast given input, so it will result
            // in shape collision.
            return false;
        }

        auto mul_const_value = mul_const->cast_vector<float>();
        if (std::any_of(mul_const_value.begin(), mul_const_value.end(), [](float f) -> bool {
                return f <= 0.0f;
            }))
            return false;

        std::shared_ptr<Node> new_const = mul_const;
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
            const auto& fq_input_shape = fq->get_input_partial_shape(0);
            if (fq_input_shape.rank().is_dynamic())
                return false;

            const auto diff = fq_input_shape.size() - const_shape.size();
            if (diff > 0) {
                // Reshape constants like (C, 1, 1) to (1, C, 1, 1)
                const_shape.insert(const_shape.begin(), diff, 1);
                new_const = std::make_shared<opset5::Reshape>(
                    new_const,
                    op::Constant::create(element::u64, Shape{const_shape.size()}, const_shape),
                    false);
            }

            // disallow constant shapes other than (N, 1, 1, ..., 1) or (1, C, 1, ..., 1)
            if (!(const_shape[0] > 1 && const_shape[0] == const_shape_size) &&
                !(const_shape.size() > 1 && const_shape[1] == const_shape_size)) {
                return false;
            }

            auto fq_users = fq->get_users();
            // Concat LPT transformation supports per tensor quantization only
            bool fq_user_is_concat =
                std::any_of(fq_users.begin(), fq_users.end(), [](const std::shared_ptr<Node> node_ptr) -> bool {
                    return is_type<opset5::Concat>(node_ptr);
                });
            if (fq_user_is_concat)
                return false;
        }

        auto input_low_div = std::make_shared<opset5::Divide>(fq->input_value(1), new_const);
        std::shared_ptr<Node> new_input_low = get_constant_from_source(input_low_div);
        if (!new_input_low)
            new_input_low = input_low_div;
        auto input_high_div = std::make_shared<opset5::Divide>(fq->input_value(2), new_const);
        std::shared_ptr<Node> new_input_high = get_constant_from_source(input_high_div);
        if (!new_input_high)
            new_input_high = input_high_div;

        auto new_fq =
            fq->clone_with_new_inputs({input, new_input_low, new_input_high, fq->input_value(3), fq->input_value(4)});
        if (transformation_callback(new_fq))
            return false;
        register_new_node(new_fq);
        copy_runtime_info({pattern_value_map.at(mul_pattern).get_node_shared_ptr(), fq},
                          {new_const, new_input_low, new_input_high, new_fq});
        new_fq->set_friendly_name(fq->get_friendly_name());
        replace_node(fq, new_fq);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fq_pattern, matcher_name);
    this->register_matcher(m, callback);
}
