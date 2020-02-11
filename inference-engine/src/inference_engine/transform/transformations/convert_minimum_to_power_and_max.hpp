// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/pass/graph_rewrite.hpp>

#include "ngraph/op/minimum.hpp"

namespace ngraph {
namespace pass {

class ConvertMinimum;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertMinimum: public ngraph::pass::GraphRewrite {
public:
    ConvertMinimum() : GraphRewrite() {
        convert_minimum();
    }

private:
    void convert_minimum() {
        auto input0 = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1, 1});
        auto input1 = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1, 1});
        auto minimum = std::make_shared<ngraph::op::v1::Minimum>(input0, input1);

        ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
            auto minimum = std::dynamic_pointer_cast<ngraph::op::v1::Minimum> (m.get_match_root());
            if (!minimum) {
                return false;
            }

            /*
             * Decompose Minimum operation to Mul(-1)---->Maximum-->Mul(-1)
             *                                Mul(-1)--'
             */

            auto neg_0 = std::make_shared<ngraph::op::v1::Multiply>(minimum->input(0).get_source_output(),
                                                                    op::Constant::create(minimum->get_input_element_type(0), Shape{1}, {-1}));

            auto neg_1 = std::make_shared<ngraph::op::v1::Multiply>(minimum->input(1).get_source_output(),
                                                                    op::Constant::create(minimum->get_input_element_type(1), Shape{1}, {-1}));

            auto max = std::make_shared<ngraph::op::v1::Maximum>(neg_0, neg_1);

            auto neg_2 = std::make_shared<ngraph::op::v1::Multiply>(max, op::Constant::create(max->get_element_type(), Shape{1}, {-1}));

            neg_2->set_friendly_name(minimum->get_friendly_name());

            ngraph::replace_node(minimum, neg_2);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(minimum, "ConvertMinimum");
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
};
