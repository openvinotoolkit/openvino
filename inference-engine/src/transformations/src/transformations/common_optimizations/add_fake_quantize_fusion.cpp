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
        std::shared_ptr<Node> add_const = std::dynamic_pointer_cast<opset5::Constant>(pattern_value_map.at(const_pattern).get_node_shared_ptr());
        if (!add_const)
            return false;
        auto const_shape = add_const->get_shape();
        size_t const_shape_size = shape_size(const_shape);
        if (const_shape_size > 1) {
            // disallow constant shapes other than (N, 1, 1, ..., 1) or (1, C, 1, ..., 1)
            if (!(const_shape[0] > 1 && const_shape[0] == const_shape_size) &&
                !(const_shape.size() > 1 && const_shape[1] == const_shape_size)) {
                return false;
            }
        }

        if (const_shape_size > 1 &&
            static_cast<Dimension::value_type>(const_shape.size()) < fq->get_input_partial_shape(0).rank().get_length()) {
            // Reshape constants like (C, 1, 1) to (1, C, 1, 1)
            const_shape.insert(const_shape.begin(), fq->get_input_partial_shape(0).rank().get_length() - const_shape.size(), 1);
            add_const = std::make_shared<opset5::Reshape>(add_const, op::Constant::create(element::u64, Shape{const_shape.size()}, const_shape), false);
        }
        auto new_input_low = std::make_shared<opset5::Subtract>(fq->input_value(1), add_const);
        auto new_input_high = std::make_shared<opset5::Subtract>(fq->input_value(2), add_const);
        auto new_fq = register_new_node<opset5::FakeQuantize>(pattern_value_map.at(input_pattern),
                                                              new_input_low,
                                                              new_input_high,
                                                              fq->input_value(3),
                                                              fq->input_value(4),
                                                              fq->get_levels());
        new_fq->set_friendly_name(fq->get_friendly_name());
        copy_runtime_info({pattern_value_map.at(add_pattern).get_node_shared_ptr(), fq}, {new_input_low, new_input_high, new_fq});
        replace_node(fq, new_fq);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fq_pattern, matcher_name);
    this->register_matcher(m, callback);
}
