// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/mish_fusion.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>

void ngraph::pass::MishFusion::mish_fusion() {
    auto input0 = std::make_shared<pattern::op::Label>(element::f64, Shape{1, 1, 1, 1});
    auto exp = std::make_shared<ngraph::opset4::Exp>(input0);
    auto input_const = op::Constant::create(element::f64, Shape{1}, {-1});
    auto add = std::make_shared<ngraph::opset4::Add>(exp, input_const);
    auto log = std::make_shared<ngraph::opset4::Log>(add);
    auto tanh = std::make_shared<ngraph::opset4::Tanh>(log);
    auto mul = std::make_shared<ngraph::opset4::Multiply>(input0, tanh);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto mul = std::dynamic_pointer_cast<ngraph::opset4::Multiply> (m.get_match_root());
        if (!mul) {
            return false;
        }

        auto tanh = std::dynamic_pointer_cast<ngraph::opset4::Tanh> (mul->input_value(1).get_node_shared_ptr());
        if (!tanh) {
            return false;
        }

        auto log = std::dynamic_pointer_cast<ngraph::opset4::Log> (tanh->input_value(0).get_node_shared_ptr());
        if (!log) {
            return false;
        }

        auto add = std::dynamic_pointer_cast<ngraph::opset4::Add> (log->input_value(0).get_node_shared_ptr());
        if (!add) {
            return false;
        }

        auto exp = std::dynamic_pointer_cast<ngraph::opset4::Add> (add->input_value(0).get_node_shared_ptr());
        if (!exp) {
            return false;
        }

        auto mish = std::make_shared<ngraph::opset4::Mish>(exp->input(0).get_source_output());

        mish->set_friendly_name(exp->get_friendly_name());
        ngraph::copy_runtime_info({mul, tanh, log, add, exp}, mish);
        ngraph::replace_node(exp, mish);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul, "MishFusion");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
