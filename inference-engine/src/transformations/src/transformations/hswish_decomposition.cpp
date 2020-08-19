// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/hswish_decomposition.hpp"

#include <memory>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ngraph::pass::HSwishDecomposition::HSwishDecomposition() {
    // Decomposes HSwish(x) op into sub-graph x * (min(Relu(x + 3), 6) * const(1/6)
    auto hswish = ngraph::pattern::wrap_type<opset4::HSwish>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        auto &pattern_to_output = m.get_pattern_value_map();
        auto hswish_node = pattern_to_output.at(hswish).get_node_shared_ptr();

        if (m_transformation_callback(hswish_node)) {
            return false;
        }

        auto input_type = hswish_node->input_value(0).get_element_type();
        auto add_constant = ngraph::opset4::Constant::create(input_type, ngraph::Shape{}, {3.0});
        auto add = std::make_shared<ngraph::opset4::Add>(hswish_node->input_value(0), add_constant);
        auto relu = std::make_shared<ngraph::opset4::Relu>(add);
        auto min_constant = ngraph::opset4::Constant::create(input_type, ngraph::Shape{}, {6.0});
        auto min = std::make_shared<ngraph::opset4::Minimum>(relu, min_constant);
        auto mul_first = std::make_shared<ngraph::opset4::Multiply>(hswish_node->input_value(0), min);
        auto mul_constant = ngraph::opset4::Constant::create(input_type, ngraph::Shape{}, {(1.0/6.0)});  // const(1/6)
        auto mul_second = std::make_shared<ngraph::opset4::Multiply>(mul_first, mul_constant);

        mul_second->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info(hswish_node,
                                  {add_constant, add, relu, min_constant, min, mul_first, mul_constant, mul_second});
        ngraph::replace_node(m.get_match_root(), mul_second);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(hswish, "HSwishDecomposition");
    register_matcher(m, callback);
}
