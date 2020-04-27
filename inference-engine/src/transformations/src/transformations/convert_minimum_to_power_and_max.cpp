// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_minimum_to_power_and_max.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

void ngraph::pass::ConvertMinimum::convert_minimum() {
    auto input0 = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1, 1});
    auto input1 = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1, 1});
    auto minimum = std::make_shared<ngraph::opset1::Minimum>(input0, input1);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto minimum = std::dynamic_pointer_cast<ngraph::opset1::Minimum> (m.get_match_root());
        if (!minimum) {
            return false;
        }

        /*
         * Decompose Minimum operation to Mul(-1)---->Maximum-->Mul(-1)
         *                                Mul(-1)--'
         */

        auto neg_0 = std::make_shared<ngraph::opset1::Multiply>(minimum->input(0).get_source_output(),
                                                                opset1::Constant::create(minimum->get_input_element_type(0), Shape{1}, {-1}));

        auto neg_1 = std::make_shared<ngraph::opset1::Multiply>(minimum->input(1).get_source_output(),
                                                                opset1::Constant::create(minimum->get_input_element_type(1), Shape{1}, {-1}));

        auto max = std::make_shared<ngraph::opset1::Maximum>(neg_0, neg_1);

        auto neg_2 = std::make_shared<ngraph::opset1::Multiply>(max, opset1::Constant::create(max->get_element_type(), Shape{1}, {-1}));

        neg_2->set_friendly_name(minimum->get_friendly_name());
        ngraph::copy_runtime_info(minimum, {neg_0, neg_1, max, neg_2});
        ngraph::replace_node(minimum, neg_2);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(minimum, "ConvertMinimum");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}