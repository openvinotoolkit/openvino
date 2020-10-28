// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/hsigmoid_decomposition.hpp"

#include <memory>

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::HSigmoidDecomposition, "HSigmoidDecomposition", 0);

ngraph::pass::HSigmoidDecomposition::HSigmoidDecomposition() {
    // Decomposes HSigmoid(x) op into sub-graph (min(Relu(x + 3), 6) * const(1/6)
    auto hsigmoid = ngraph::pattern::wrap_type<opset5::HSigmoid>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        auto &pattern_to_output = m.get_pattern_value_map();
        auto hsigmoid_node = pattern_to_output.at(hsigmoid).get_node_shared_ptr();

        if (m_transformation_callback(hsigmoid_node)) {
            return false;
        }

        auto input_type = hsigmoid_node->input_value(0).get_element_type();
        auto add_constant = ngraph::opset5::Constant::create(input_type, ngraph::Shape{}, {3.0});
        auto add = std::make_shared<ngraph::opset5::Add>(hsigmoid_node->input_value(0), add_constant);
        auto relu = std::make_shared<ngraph::opset5::Relu>(add);
        auto min_constant = ngraph::opset5::Constant::create(input_type, ngraph::Shape{}, {6.0});
        auto min = register_new_node<ngraph::opset5::Minimum>(relu, min_constant);
        auto mul = std::make_shared<ngraph::opset5::Multiply>(hsigmoid_node->input_value(0), min);

        mul->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info(hsigmoid_node,
                                  {add_constant, add, relu, min_constant, min, mul});
        ngraph::replace_node(m.get_match_root(), mul);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(hsigmoid, "HSigmoidDecomposition");
    register_matcher(m, callback);
}
