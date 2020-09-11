// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/reshape_with_hc_output.hpp"

#include <memory>
#include <vector>

#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset4.hpp>

void relax_hc_reshape_followed_by_matmul(std::shared_ptr<ngraph::opset4::MatMul> matmul, std::shared_ptr<ngraph::Node> reshape,
                                         ngraph::Output<ngraph::Node> shape_source, std::shared_ptr<ngraph::opset4::Constant> reshape_pattern,
                                         const bool & reshape_is_A_input) {
    const auto &raw_idx = reshape_is_A_input ? (matmul->get_transpose_b() ? -1 : -2) : (matmul->get_transpose_a() ? -2 : -1);
    const auto &idx = ngraph::normalize_axis(matmul->description(), raw_idx, reshape->get_output_partial_shape(0).rank());

    const auto &shape_of = std::make_shared<ngraph::opset4::ShapeOf>(shape_source);
    const auto &C = std::make_shared<ngraph::opset4::Gather>(shape_of,
                                                     ngraph::opset4::Constant::create(ngraph::element::i64, {1}, {idx}),
                                                     ngraph::opset4::Constant::create(ngraph::element::i64, {}, {0}));
    const auto &N = ngraph::opset4::Constant::create(ngraph::element::i64, {1}, {-1});
    const auto &pattern_vector = reshape_is_A_input ? (matmul->get_transpose_a() ? ngraph::OutputVector({C, N}) :
                                                       ngraph::OutputVector({N, C})) : (matmul->get_transpose_b() ? ngraph::OutputVector({N, C})
                                                                                                                  : ngraph::OutputVector({C, N}));
    const auto new_reshape_pattern = std::make_shared<ngraph::opset4::Concat>(pattern_vector, 0);

    new_reshape_pattern->set_friendly_name(reshape_pattern->get_friendly_name());
    copy_runtime_info(reshape_pattern, new_reshape_pattern);
    replace_node(reshape_pattern, new_reshape_pattern);
}

ngraph::pass::ReshapeAMatMul::ReshapeAMatMul() {
    auto other_input_label = pattern::any_input();
    auto reshape_input_label = pattern::any_input();
    auto reshape_pattern_label = ngraph::pattern::wrap_type<opset4::Constant>();
    auto reshape_label = ngraph::pattern::wrap_type<opset4::Reshape>({reshape_input_label, reshape_pattern_label});

    auto matmul_A_label = ngraph::pattern::wrap_type<opset4::MatMul>({reshape_label, other_input_label});
    matcher_pass_callback callback_A_input = [=](pattern::Matcher &m) -> bool {
        const auto &pattern_to_output = m.get_pattern_value_map();

        const auto & reshape_pattern = std::dynamic_pointer_cast<opset4::Constant>(pattern_to_output.at(reshape_pattern_label).get_node_shared_ptr());
        const auto & matmul = std::dynamic_pointer_cast<op::MatMul>(pattern_to_output.at(matmul_A_label).get_node_shared_ptr());
        if (!reshape_pattern || !matmul || reshape_pattern->get_shape() != Shape{2})
            return false;
        const auto &shape_source = pattern_to_output.at(other_input_label);
        const auto &reshape = pattern_to_output.at(reshape_label).get_node_shared_ptr();

        relax_hc_reshape_followed_by_matmul(matmul, reshape, shape_source, reshape_pattern, true);
        return true;
    };
    auto m_A = std::make_shared<ngraph::pattern::Matcher>(matmul_A_label, "ReshapeMatMul_A");
    register_matcher(m_A, callback_A_input);
}

ngraph::pass::ReshapeBMatMul::ReshapeBMatMul() {
    auto other_input_label = pattern::any_input();
    auto reshape_input_label = pattern::any_input();
    auto reshape_pattern_label = ngraph::pattern::wrap_type<opset4::Constant>();
    auto reshape_label = ngraph::pattern::wrap_type<opset4::Reshape>({reshape_input_label, reshape_pattern_label});

    auto matmul_B_label = ngraph::pattern::wrap_type<opset4::MatMul>({other_input_label, reshape_label});
    matcher_pass_callback callback_B_input = [=](pattern::Matcher &m) -> bool {
        const auto &pattern_to_output = m.get_pattern_value_map();

        const auto & reshape_pattern = std::dynamic_pointer_cast<opset4::Constant>(pattern_to_output.at(reshape_pattern_label).get_node_shared_ptr());
        const auto & matmul = std::dynamic_pointer_cast<op::MatMul>(pattern_to_output.at(matmul_B_label).get_node_shared_ptr());
        if (!reshape_pattern || !matmul || reshape_pattern->get_shape() != Shape{2})
            return false;
        const auto &shape_source = pattern_to_output.at(other_input_label);
        const auto &reshape = pattern_to_output.at(reshape_label).get_node_shared_ptr();

        relax_hc_reshape_followed_by_matmul(matmul, reshape, shape_source, reshape_pattern, false);
        return true;
    };
    auto m_B = std::make_shared<ngraph::pattern::Matcher>(matmul_B_label, "ReshapeMatMul_B");
    register_matcher(m_B, callback_B_input);
}