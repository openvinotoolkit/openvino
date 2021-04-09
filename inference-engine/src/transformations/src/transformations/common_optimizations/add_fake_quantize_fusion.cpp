// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/add_fake_quantize_fusion.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/validation_util.hpp>
#include "itt.hpp"


NGRAPH_RTTI_DEFINITION(ngraph::pass::AddFakeQuantizeFusion, "AddFakeQuantizeFusion", 0);

ngraph::pass::AddFakeQuantizeFusion::AddFakeQuantizeFusion() {
    MATCHER_SCOPE(AddFakeQuantizeFusion);
    auto input_pattern = ngraph::pattern::any_input();
    auto const_pattern = ngraph::pattern::wrap_type<opset5::Constant>();
    auto add_pattern = ngraph::pattern::wrap_type<opset5::Add>({input_pattern, const_pattern},
                                                               pattern::consumers_count(1));
    auto fq_pattern = ngraph::pattern::wrap_type<opset5::FakeQuantize>({add_pattern,
                                                                        ngraph::pattern::any_input(),
                                                                        ngraph::pattern::any_input(),
                                                                        ngraph::pattern::any_input(),
                                                                        ngraph::pattern::any_input()});
    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_value_map = m.get_pattern_value_map();
        auto fq = std::dynamic_pointer_cast<opset5::FakeQuantize>(pattern_value_map.at(fq_pattern).get_node_shared_ptr());
        if (!fq)
            return false;
        const auto& add_node = pattern_value_map.at(add_pattern).get_node_shared_ptr();
        auto add_const = std::dynamic_pointer_cast<opset5::Constant>(pattern_value_map.at(const_pattern).get_node_shared_ptr());
        if (!add_const)
            return false;
        std::shared_ptr<Node> new_const = add_const;
        auto const_shape = add_const->get_shape();
        size_t const_shape_size = shape_size(const_shape);
        bool is_const_scalar = const_shape_size == 1;

        if (!is_const_scalar) {
            // disallow constant shapes other than (N, 1, 1, ..., 1) or (1, C, 1, ..., 1)
            if (!(const_shape[0] > 1 && const_shape[0] == const_shape_size) &&
                !(const_shape.size() > 1 && const_shape[1] == const_shape_size)) {
                return false;
            }
            float v;
            bool is_single_value = op::util::get_single_value(add_const, v);
            if (is_single_value) {
                new_const = op::Constant::create(element::f32, Shape{1}, {v});
                if (add_const->get_element_type() != element::f32) {
                    new_const = std::make_shared<opset5::Convert>(new_const, add_const->get_element_type());
                }
            } else {
                const auto& add_inputs = add_node->input_values();
                const auto& node_type_info = add_inputs[0].get_node()->get_type_info();
                if (node_type_info == opset5::Convolution::type_info ||
                    node_type_info == opset5::GroupConvolution::type_info ||
                    node_type_info == opset5::ConvolutionBackpropData::type_info ||
                    node_type_info == opset5::GroupConvolutionBackpropData::type_info ||
                    node_type_info == opset5::MatMul::type_info) {
                    return false;
                } else {
                    auto diff = fq->get_input_partial_shape(0).rank().get_length() - static_cast<Dimension::value_type>(const_shape.size());
                    if (diff > 0) {
                        // Reshape constants like (C, 1, 1) to (1, C, 1, 1)
                        const_shape.insert(const_shape.begin(), diff, 1);
                        new_const = std::make_shared<opset5::Reshape>(new_const,
                                op::Constant::create(element::u64, Shape{const_shape.size()}, const_shape), false);
                    }
                }
            }
        }

        auto new_input_low = get_constant_from_source(std::make_shared<opset5::Subtract>(fq->input_value(1), new_const));
        auto new_input_high = get_constant_from_source(std::make_shared<opset5::Subtract>(fq->input_value(2), new_const));
        auto new_fq = register_new_node<opset5::FakeQuantize>(pattern_value_map.at(input_pattern),
                                                              new_input_low,
                                                              new_input_high,
                                                              fq->input_value(3),
                                                              fq->input_value(4),
                                                              fq->get_levels());
        new_fq->set_friendly_name(fq->get_friendly_name());
        copy_runtime_info({add_node, fq}, {new_input_low, new_input_high, new_fq});
        replace_node(fq, new_fq);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fq_pattern, matcher_name);
    this->register_matcher(m, callback);
}
