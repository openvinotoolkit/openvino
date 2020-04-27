// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_negative.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

void ngraph::pass::ConvertNegative::convert_negative() {
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
    auto neg = std::make_shared<ngraph::opset1::Negative>(input);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto neg = std::dynamic_pointer_cast<ngraph::opset1::Negative> (m.get_match_root());
        if (!neg) {
            return false;
        }

        auto mul = std::make_shared<ngraph::opset1::Multiply>(neg->input(0).get_source_output(),
                                                              opset1::Constant::create(neg->get_element_type(), Shape{1}, {-1}));
        mul->set_friendly_name(neg->get_friendly_name());
        ngraph::copy_runtime_info(neg, mul);
        ngraph::replace_node(neg, mul);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(neg, "ConvertNegative");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}