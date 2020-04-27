// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_divide.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

void ngraph::pass::ConvertDivide::convert_divide() {
    auto input0 = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1, 1});
    auto input1 = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1, 1});
    auto div = std::make_shared<ngraph::opset1::Divide>(input0, input1);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto div = std::dynamic_pointer_cast<ngraph::opset1::Divide> (m.get_match_root());
        if (!div) {
            return false;
        }

        auto pow = std::make_shared<ngraph::opset1::Power>(div->input(1).get_source_output(),
                                                           op::Constant::create(div->get_input_element_type(1), Shape{1}, {-1}));

        auto mul = std::make_shared<ngraph::opset1::Multiply>(div->input(0).get_source_output(), pow);

        mul->set_friendly_name(div->get_friendly_name());
        ngraph::copy_runtime_info(div, {pow, mul});
        ngraph::replace_node(div, mul);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(div, "ConvertDivide");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}