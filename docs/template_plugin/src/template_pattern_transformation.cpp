// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "template_pattern_transformation.hpp"

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/rt_info.hpp>

using namespace ngraph;

// ! [graph_rewrite:template_transformation_cpp]
// template_pattern_transformation.cpp
void ngraph::pass::MyPatternBasedTransformation::transform() {
    // Pattern example
    auto input0 = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1, 1});
    auto input1 = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1, 1});
    auto div = std::make_shared<ngraph::opset3::Divide>(input0, input1);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto div = std::dynamic_pointer_cast<ngraph::opset3::Divide> (m.get_match_root());
        // We can not apply this transformation in case with integer input data type
        if (!div || div->input(0).get_element_type().is_integral()) {
            return false;
        }

        // Decompose Divide into Multiply with Power operations
        auto pow = std::make_shared<ngraph::opset3::Power>(div->input_value(1),
                                                           opset3::Constant::create(div->get_input_element_type(1), Shape{1}, {-1}));

        auto mul = std::make_shared<ngraph::opset3::Multiply>(div->input_value(0), pow);

        // Save original name to last operation in replacement sub-graph
        mul->set_friendly_name(div->get_friendly_name());

        // Copy runtime info attributes to newly created operation
        ngraph::copy_runtime_info(div, {pow, mul});

        // Replace Divide operation with Multiply
        ngraph::replace_node(div, mul);

        // Return true as the root node was changed
        return true;
    };

    // Register pattern with divide operaiton as a pattern root node
    auto m = std::make_shared<ngraph::pattern::Matcher>(div, "ConvertDivide");
    // Register Matcher
    this->add_matcher(m, callback, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
}
// ! [graph_rewrite:template_transformation_cpp]