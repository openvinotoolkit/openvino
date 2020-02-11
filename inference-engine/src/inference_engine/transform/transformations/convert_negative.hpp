// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/pass/graph_rewrite.hpp>

#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"

namespace ngraph {
namespace pass {

class ConvertNegative;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertNegative: public ngraph::pass::GraphRewrite {
public:
    ConvertNegative() : GraphRewrite() {
        convert_negative();
    }

private:
    void convert_negative() {
        auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
        auto neg = std::make_shared<ngraph::op::Negative>(input);

        ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
            auto neg = std::dynamic_pointer_cast<ngraph::op::Negative> (m.get_match_root());
            if (!neg) {
                return false;
            }

            auto mul = std::make_shared<ngraph::op::v1::Multiply>(neg->input(0).get_source_output(),
                                                                  op::Constant::create(neg->get_element_type(), Shape{1}, {-1}));
            mul->set_friendly_name(neg->get_friendly_name());

            ngraph::replace_node(neg, mul);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(neg, "ConvertNegative");
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
};
