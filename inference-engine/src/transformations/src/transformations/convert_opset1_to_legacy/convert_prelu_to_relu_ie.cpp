// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_prelu_to_relu_ie.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>

#include <ngraph_ops/relu_ie.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/rt_info.hpp>

ngraph::pass::ConvertPReLUToReLUIEMatcher::ConvertPReLUToReLUIEMatcher() {
    auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
    auto input_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
    auto prelu = std::make_shared<ngraph::opset1::PRelu>(input_0, input_1);


    ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto prelu = std::dynamic_pointer_cast<ngraph::opset1::PRelu> (m.get_match_root());
        if (!prelu) {
            return false;
        }
        auto node = prelu->input(1).get_source_output().get_node_shared_ptr();
        if (auto const_node = std::dynamic_pointer_cast<ngraph::opset1::Constant>(node)) {
            float value(0);
            if (!ngraph::op::util::get_single_value(const_node, value)) {
                return false;
            }

            auto relu_ie = std::make_shared<ngraph::op::ReLUIE>(prelu->input(0).get_source_output(), value);
            relu_ie->set_friendly_name(prelu->get_friendly_name());
            ngraph::copy_runtime_info(prelu, relu_ie);
            ngraph::replace_node(prelu, relu_ie);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(prelu, "ConvertPReLUToReLUIE");
    this->register_matcher(m, callback);
}