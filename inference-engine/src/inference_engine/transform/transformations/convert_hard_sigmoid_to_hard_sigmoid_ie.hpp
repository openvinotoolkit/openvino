// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph_ops/hard_sigmoid_ie.hpp>

#include <transform/transformations/utils/utils.hpp>

#include "ngraph/op/fused/hard_sigmoid.hpp"
#include "ngraph/op/constant.hpp"

namespace ngraph {
namespace pass {

class ConvertHardSigmoidToHardSigmoidIE;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertHardSigmoidToHardSigmoidIE : public ngraph::pass::GraphRewrite {
public:
    ConvertHardSigmoidToHardSigmoidIE() : GraphRewrite() {
        convert_hard_sigmoid();
    }

private:
    void convert_hard_sigmoid();
};

void ngraph::pass::ConvertHardSigmoidToHardSigmoidIE::convert_hard_sigmoid() {
    auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto input_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto input_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto node = std::make_shared<ngraph::op::HardSigmoid>(input_0, input_1, input_2);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto hard_sigmoid = std::dynamic_pointer_cast<ngraph::op::HardSigmoid> (m.get_match_root());
        if (!hard_sigmoid) {
            return false;
        }

        auto alpha = std::dynamic_pointer_cast<ngraph::op::Constant> (hard_sigmoid->input(1).get_source_output().get_node_shared_ptr());
        if (!alpha) {
            return false;
        }

        auto beta = std::dynamic_pointer_cast<ngraph::op::Constant> (hard_sigmoid->input(2).get_source_output().get_node_shared_ptr());
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
        ngraph::replace_node(m.get_match_root(), hard_sigmoid_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(node, "ConvertHardSigmoidToHardSigmoidIE");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
