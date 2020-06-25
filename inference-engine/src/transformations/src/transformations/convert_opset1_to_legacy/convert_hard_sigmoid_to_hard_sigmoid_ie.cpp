// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_hard_sigmoid_to_hard_sigmoid_ie.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

#include <transformations/utils/utils.hpp>
#include <ngraph_ops/hard_sigmoid_ie.hpp>


void ngraph::pass::ConvertHardSigmoidToLegacyMatcher::convert_hard_sigmoid() {
    auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto input_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto input_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto node = std::make_shared<ngraph::opset1::HardSigmoid>(input_0, input_1, input_2);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto hard_sigmoid = std::dynamic_pointer_cast<ngraph::opset1::HardSigmoid> (m.get_match_root());
        if (!hard_sigmoid) {
            return false;
        }

        auto alpha = std::dynamic_pointer_cast<ngraph::opset1::Constant> (hard_sigmoid->input(1).get_source_output().get_node_shared_ptr());
        if (!alpha) {
            return false;
        }

        auto beta = std::dynamic_pointer_cast<ngraph::opset1::Constant> (hard_sigmoid->input(2).get_source_output().get_node_shared_ptr());
        if (!beta) {
            return false;
        }

        float alpha_value;
        float beta_value;
        if (!ngraph::op::util::get_single_value(alpha, alpha_value) || !ngraph::op::util::get_single_value(beta, beta_value))
            return false;

        auto hard_sigmoid_ie = std::make_shared<ngraph::op::HardSigmoid_IE> (hard_sigmoid->input(0).get_source_output(),
                                                                             alpha_value,
                                                                             beta_value);

        hard_sigmoid_ie->set_friendly_name(hard_sigmoid->get_friendly_name());
        ngraph::copy_runtime_info(hard_sigmoid, hard_sigmoid_ie);
        ngraph::replace_node(hard_sigmoid, hard_sigmoid_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(node, "ConvertHardSigmoidToLegacy");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}