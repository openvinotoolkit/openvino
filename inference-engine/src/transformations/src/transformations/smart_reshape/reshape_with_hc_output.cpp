// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/reshape_with_hc_output.hpp"
#include "transformations/smart_reshape/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset4.hpp>

bool relax_hc_reshape_followed_by_matmul(const ngraph::pattern::PatternValueMap & pattern_to_output,
                                         const std::shared_ptr<ngraph::Node> & matmul_label,
                                         const std::shared_ptr<ngraph::Node> & reshape_label,
                                         const std::shared_ptr<ngraph::Node> & other_input_label,
                                         const std::shared_ptr<ngraph::Node> & reshape_pattern_label,
                                         bool reshape_is_A_input) {
    auto reshape_pattern = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(reshape_pattern_label).get_node_shared_ptr());
    const auto & matmul = std::dynamic_pointer_cast<ngraph::opset4::MatMul>(pattern_to_output.at(matmul_label).get_node_shared_ptr());
    if (!reshape_pattern || !matmul || reshape_pattern->get_shape() != ngraph::Shape{2})
        return false;
    const auto &shape_source = pattern_to_output.at(other_input_label);
    if (ngraph::is_type<ngraph::opset4::Transpose>(shape_source.get_node_shared_ptr()) ||
            ngraph::is_type<ngraph::opset4::Reshape>(shape_source.get_node_shared_ptr()))
        // avoiding loop creation
        return false;
    const auto & reshape = pattern_to_output.at(reshape_label).get_node_shared_ptr();

    const auto & raw_idx = reshape_is_A_input ? (matmul->get_transpose_b() ? -1 : -2) : (matmul->get_transpose_a() ? -2 : -1);
    const auto & idx = ngraph::normalize_axes(matmul->description(), {raw_idx}, reshape->get_output_partial_shape(0).rank());
    const auto & C = ngraph::op::util::node_to_get_shape_value_of_indices_from_shape_source(shape_source, idx);
    const auto & N = ngraph::opset4::Constant::create(ngraph::element::i64, {1}, {-1});
    const auto & pattern_vector = reshape_is_A_input ?
            (matmul->get_transpose_a() ? ngraph::OutputVector({C, N}) : ngraph::OutputVector({N, C})) :
            (matmul->get_transpose_b() ? ngraph::OutputVector({N, C}) : ngraph::OutputVector({C, N}));
    const auto & new_reshape_pattern = std::make_shared<ngraph::opset4::Concat>(pattern_vector, 0);

    new_reshape_pattern->set_friendly_name(reshape_pattern->get_friendly_name());
    copy_runtime_info(reshape_pattern, new_reshape_pattern);
    replace_node(reshape_pattern, new_reshape_pattern);
    return true;
}

ngraph::pass::ReshapeAMatMul::ReshapeAMatMul() {
    auto other_input_label = pattern::any_input();
    auto reshape_input_label = pattern::any_input();
    auto reshape_pattern_label = ngraph::pattern::wrap_type<opset4::Constant>();
    auto reshape_label = ngraph::pattern::wrap_type<opset4::Reshape>({reshape_input_label, reshape_pattern_label});
    auto matmul_label = ngraph::pattern::wrap_type<opset4::MatMul>({reshape_label, other_input_label});

    matcher_pass_callback callback = [=](pattern::Matcher &m) -> bool {
        const auto & pattern_to_output = m.get_pattern_value_map();
        return relax_hc_reshape_followed_by_matmul(
                pattern_to_output, matmul_label, reshape_label, other_input_label, reshape_pattern_label, true);
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(matmul_label, "ReshapeMatMul_A");
    register_matcher(m, callback);
}

ngraph::pass::ReshapeBMatMul::ReshapeBMatMul() {
    auto other_input_label = pattern::any_input();
    auto reshape_input_label = pattern::any_input();
    auto reshape_pattern_label = ngraph::pattern::wrap_type<opset4::Constant>();
    auto reshape_label = ngraph::pattern::wrap_type<opset4::Reshape>({reshape_input_label, reshape_pattern_label});
    auto matmul_label = ngraph::pattern::wrap_type<opset4::MatMul>({other_input_label, reshape_label});

    matcher_pass_callback callback = [=](pattern::Matcher &m) -> bool {
        const auto & pattern_to_output = m.get_pattern_value_map();
        return relax_hc_reshape_followed_by_matmul(
                pattern_to_output, matmul_label, reshape_label, other_input_label, reshape_pattern_label, false);
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(matmul_label, "ReshapeMatMul_B");
    register_matcher(m, callback);
}