// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "template_pattern_transformation.hpp"

#include <ngraph/opsets/opset3.hpp>

// ! [graph_rewrite:template_transformation_cpp]
// template_pattern_transformation.cpp
void MyPatternBasedTransformation::transform() {
    // Pattern example
    auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{1});
    auto shapeof = std::make_shared<ngraph::opset3::ShapeOf>(input);

    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher& m) {
        // Transformation code
        return false;
    };

    // Register pattern with shapeof as a pattern root node
    auto m = std::make_shared<ngraph::pattern::Matcher>(shapeof, "MyPatternBasedTransformation");
    // Register Matcher
    this->add_matcher(m, callback, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
}
// ! [graph_rewrite:template_transformation_cpp]