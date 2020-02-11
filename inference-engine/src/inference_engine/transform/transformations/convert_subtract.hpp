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

class ConvertSubtract;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertSubtract: public ngraph::pass::GraphRewrite {
public:
    ConvertSubtract() : GraphRewrite() {
        convert_subtract();
    }

private:
    void convert_subtract() {
        auto input0 = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1, 1});
        auto input1 = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1, 1});
        auto sub = std::make_shared<ngraph::op::v1::Subtract>(input0, input1);

        ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
            auto sub = std::dynamic_pointer_cast<ngraph::op::v1::Subtract> (m.get_match_root());
            if (!sub) {
                return false;
            }

            auto neg = std::make_shared<ngraph::op::v1::Multiply>(sub->input(1).get_source_output(),
                                                                  op::Constant::create(sub->get_input_element_type(1), Shape{1}, {-1}));

            auto add = std::make_shared<ngraph::op::v1::Add>(sub->input(0).get_source_output(), neg);

            add->set_friendly_name(sub->get_friendly_name());

            ngraph::replace_node(sub, add);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(sub, "ConvertSubtract");
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
};
