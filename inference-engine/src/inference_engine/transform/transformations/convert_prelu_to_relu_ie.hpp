// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph_ops/relu_ie.hpp>
#include <transform/transformations/utils/utils.hpp>

#include "ngraph/op/fused/prelu.hpp"
#include "ngraph/op/constant.hpp"

namespace ngraph {
namespace pass {

class ConvertPReLUToReLUIE;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertPReLUToReLUIE: public ngraph::pass::GraphRewrite {
public:
    ConvertPReLUToReLUIE() : GraphRewrite() {
        convert_prelu();
    }

private:
    void convert_prelu();
};

void ngraph::pass::ConvertPReLUToReLUIE::convert_prelu() {
    auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
    auto input_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
    auto prelu = std::make_shared<ngraph::op::PRelu>(input_0, input_1);


    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto prelu = std::dynamic_pointer_cast<ngraph::op::PRelu> (m.get_match_root());
        if (!prelu) {
            return false;
        }
        auto node = prelu->input(1).get_source_output().get_node_shared_ptr();
        if (auto const_node = std::dynamic_pointer_cast<ngraph::op::Constant>(node)) {
            float value(0);
            if (!ngraph::op::util::get_single_value(const_node, value)) {
                return false;
            }

            auto relu_ie = std::make_shared<ngraph::op::ReLUIE>(prelu->input(0).get_source_output(), value);
            relu_ie->set_friendly_name(prelu->get_friendly_name());
            ngraph::replace_node(m.get_match_root(), relu_ie);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(prelu, "ConvertPReLUToReLUIE");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
