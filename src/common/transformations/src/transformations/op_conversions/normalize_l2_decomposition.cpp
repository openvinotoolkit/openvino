// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/normalize_l2_decomposition.hpp"

#include <memory>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"

ngraph::pass::NormalizeL2Decomposition::NormalizeL2Decomposition() {
    MATCHER_SCOPE(NormalizeL2Decomposition);
    auto normalize_l2_pattern = ngraph::pattern::wrap_type<opset8::NormalizeL2>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto normalize_l2 = std::dynamic_pointer_cast<opset8::NormalizeL2>(m.get_match_root());

        if (!normalize_l2 || transformation_callback(normalize_l2)) {
            return false;
        }

        auto power = std::make_shared<opset8::Power>(
            normalize_l2->input_value(0),
            opset8::Constant::create(normalize_l2->get_input_element_type(0), Shape{}, {2.0}));
        auto reduce_sum = std::make_shared<opset8::ReduceSum>(power, normalize_l2->input_value(1), true);

        std::shared_ptr<Node> eps_node;
        auto eps_const_node =
            opset8::Constant::create(normalize_l2->get_input_element_type(0), Shape{}, {normalize_l2->get_eps()});
        switch (normalize_l2->get_eps_mode()) {
        case op::EpsMode::ADD:
            eps_node = std::make_shared<opset8::Add>(reduce_sum, eps_const_node);
            break;
        case op::EpsMode::MAX:
            eps_node = std::make_shared<opset8::Maximum>(reduce_sum, eps_const_node);
            break;
        default:
            return false;
        }

        auto sqrt = std::make_shared<opset8::Sqrt>(eps_node);
        auto div = std::make_shared<opset8::Divide>(normalize_l2->input_value(0), sqrt);

        div->set_friendly_name(normalize_l2->get_friendly_name());
        ngraph::copy_runtime_info(normalize_l2, {power, reduce_sum, eps_node, sqrt, div});
        ngraph::replace_node(normalize_l2, div);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(normalize_l2_pattern, matcher_name);
    register_matcher(m, callback);
}
