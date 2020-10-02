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
    const auto & matmul = std::dynamic_pointer_cast<ngraph::opset4::MatMul>(pattern_to_output.at(matmul_label).get_node_shared_ptr());
    if (!matmul)
        return false;
    const auto &shape_source = pattern_to_output.at(other_input_label);
    if (ngraph::is_type<ngraph::opset4::Transpose>(shape_source.get_node_shared_ptr()) ||
            ngraph::is_type<ngraph::opset4::Reshape>(shape_source.get_node_shared_ptr()))
        // avoiding loop creation
        return false;
    const auto & reshape = pattern_to_output.at(reshape_label).get_node_shared_ptr();
    auto reshape_pattern = pattern_to_output.at(reshape_pattern_label).get_node_shared_ptr();

    if (!reshape->get_output_partial_shape(0).rank().is_static())
        return false;

    if (reshape_pattern->get_shape() == ngraph::Shape{0} || reshape_pattern->get_shape() == ngraph::Shape{1})
        return false;

    const auto &raw_idx = reshape_is_A_input ? (matmul->get_transpose_b() ? -1 : -2) : (matmul->get_transpose_a()
                                                                                        ? -2 : -1);
    const auto &idx = ngraph::normalize_axes(matmul->description(), {raw_idx},
                                             reshape->get_output_partial_shape(0).rank());
    const auto &C = ngraph::op::util::node_to_get_shape_value_of_indices_from_shape_source(shape_source, idx);
    const auto &N = ngraph::opset4::Constant::create(ngraph::element::i64, {1}, {-1});

    if (reshape_pattern->get_shape() == ngraph::Shape{2}) {
        const auto & pattern_vector = reshape_is_A_input ?
                              (matmul->get_transpose_a() ? ngraph::OutputVector({C, N}) : ngraph::OutputVector({N, C}))
                                                 :
                              (matmul->get_transpose_b() ? ngraph::OutputVector({N, C}) : ngraph::OutputVector({C, N}));
        const auto & new_reshape_pattern = std::make_shared<ngraph::opset4::Concat>(pattern_vector, 0);

        new_reshape_pattern->set_friendly_name(reshape_pattern->get_friendly_name());
        copy_runtime_info(reshape_pattern, new_reshape_pattern);
        reshape->input_value(1).replace(new_reshape_pattern->output(0));
        return true;
    } else if (ngraph::is_type<ngraph::opset4::Constant>(reshape_pattern)) {
        auto const_reshape_pattern = std::dynamic_pointer_cast<ngraph::opset4::Constant>(reshape_pattern);
        const auto values = const_reshape_pattern->cast_vector<int64_t>();
        for (auto i : values)
            if (i <= 0)
                return false;
        const auto &reshape_raw_idx = reshape_is_A_input ? (matmul->get_transpose_a() ? -1 : -2) :
                                      (matmul->get_transpose_b() ? -2 : -1);
        const auto &reshape_idx = ngraph::normalize_axes(matmul->description(), {reshape_raw_idx},
                                                         reshape->get_output_partial_shape(0).rank());
        const auto &D = ngraph::op::util::node_to_get_shape_value_of_indices_from_shape_node(const_reshape_pattern,
                                                                                               reshape_idx);
        auto pattern_vector = reshape_is_A_input ?
                (matmul->get_transpose_a() ? ngraph::OutputVector({N, C, D}) : ngraph::OutputVector({N, D, C})) :
                (matmul->get_transpose_b() ? ngraph::OutputVector({N, D, C}) : ngraph::OutputVector({N, C, D}));
        if (reshape->get_output_partial_shape(0).rank().get_length() > 3) {
            auto shape = reshape_pattern->get_shape();
            shape.erase(shape.begin());
            shape.erase(shape.end(), shape.end() - 1);
            auto old_raw = -3;
            std::vector<int64_t > axes;
            for (auto i : shape) {
                axes.push_back(old_raw);
                old_raw--;
            }
            const auto &reshape_indices = ngraph::normalize_axes(matmul->description(), axes,
                                                                  reshape->get_output_partial_shape(0).rank());
            const auto &reshape_indices_value = ngraph::op::util::node_to_get_shape_value_of_indices_from_shape_node(const_reshape_pattern,
                                                                                                                     reshape_indices);
            pattern_vector.insert(pattern_vector.begin() + 1, reshape_indices_value);
        }
        const auto & new_reshape_pattern = std::make_shared<ngraph::opset4::Concat>(pattern_vector, 0);
        auto new_reshape = reshape->copy_with_new_inputs({reshape->input_value(0), new_reshape_pattern});
        new_reshape->set_friendly_name(reshape->get_friendly_name());
        copy_runtime_info(reshape, new_reshape);
        reshape->output(0).replace(new_reshape->output(0));
        return true;
    }

    return false;
}

ngraph::pass::ReshapeAMatMul::ReshapeAMatMul() {
    auto other_input_label = pattern::any_input();
    auto reshape_input_label = pattern::any_input();
    auto reshape_pattern_label = pattern::any_input();
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
    auto reshape_pattern_label = pattern::any_input();
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