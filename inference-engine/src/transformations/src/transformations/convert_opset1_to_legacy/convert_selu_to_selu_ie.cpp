// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_selu_to_selu_ie.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>

#include <ngraph_ops/selu_ie.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/rt_info.hpp>

ngraph::pass::ConvertSeluToSeluIEMatcher::ConvertSeluToSeluIEMatcher() {
    auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
    auto input_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
    auto input_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
    auto selu = std::make_shared<ngraph::opset1::Selu>(input_0, input_1, input_2);

    ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto selu = std::dynamic_pointer_cast<ngraph::opset1::Selu> (m.get_match_root());
        if (!selu) {
            return false;
        }
        auto alpha_node = selu->input(1).get_source_output().get_node_shared_ptr();
        auto gamma_node = selu->input(2).get_source_output().get_node_shared_ptr();

        auto alpha_const = std::dynamic_pointer_cast<ngraph::opset1::Constant>(alpha_node);
        auto gamma_const = std::dynamic_pointer_cast<ngraph::opset1::Constant>(gamma_node);

        if (!alpha_const || !gamma_const) {
            return false;
        }

        float alpha, gamma;
        if (!ngraph::op::util::get_single_value(alpha_const, alpha) ||
            !ngraph::op::util::get_single_value(gamma_const, gamma)) {
            return false;
        }

        auto selu_ie = std::make_shared<ngraph::op::SeluIE>(selu->input(0).get_source_output(), alpha, gamma);
        selu_ie->set_friendly_name(selu->get_friendly_name());
        ngraph::copy_runtime_info(selu, selu_ie);
        ngraph::replace_node(selu, selu_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(selu, "ConvertSeluToSeluIE");
    this->register_matcher(m, callback);
}