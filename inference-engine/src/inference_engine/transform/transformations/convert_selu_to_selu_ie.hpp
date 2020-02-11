// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph_ops/selu_ie.hpp>
#include <transform/transformations/utils/utils.hpp>

#include <ngraph/op/fused/selu.hpp>
#include <ngraph/op/constant.hpp>

namespace ngraph {
namespace pass {

class ConvertSeluToSeluIE;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertSeluToSeluIE: public ngraph::pass::GraphRewrite {
public:
    ConvertSeluToSeluIE() : GraphRewrite() {
        convert_selu();
    }

private:
    void convert_selu();
};

void ngraph::pass::ConvertSeluToSeluIE::convert_selu() {
    auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
    auto input_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
    auto input_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
    auto selu = std::make_shared<ngraph::op::v0::Selu>(input_0, input_1, input_2);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto selu = std::dynamic_pointer_cast<ngraph::op::v0::Selu> (m.get_match_root());
        if (!selu) {
            return false;
        }
        auto alpha_node = selu->input(1).get_source_output().get_node_shared_ptr();
        auto gamma_node = selu->input(2).get_source_output().get_node_shared_ptr();

        auto alpha_const = std::dynamic_pointer_cast<ngraph::op::Constant>(alpha_node);
        auto gamma_const = std::dynamic_pointer_cast<ngraph::op::Constant>(gamma_node);

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
        ngraph::replace_node(selu, selu_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(selu, "ConvertSeluToSeluIE");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
