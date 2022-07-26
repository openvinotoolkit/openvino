// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/pull_transpose_through_fq.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/validation_util.hpp>
#include <transformations/utils/utils.hpp>
#include <vector>

#include "itt.hpp"

ngraph::pass::PullTransposeThroughFQUp::PullTransposeThroughFQUp() {
    MATCHER_SCOPE(PullTransposeThroughFQUp);
    const auto weights = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    auto m_fq = pattern::wrap_type<opset1::FakeQuantize>({weights,
                                                          pattern::any_input(pattern::has_static_shape()),
                                                          pattern::any_input(pattern::has_static_shape()),
                                                          pattern::any_input(pattern::has_static_shape()),
                                                          pattern::any_input(pattern::has_static_shape())},
                                                         pattern::consumers_count(1));
    auto m_transpose_perm = pattern::wrap_type<opset1::Constant>();
    auto m_transpose = pattern::wrap_type<opset1::Transpose>({m_fq, m_transpose_perm});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto& pattern_map = m.get_pattern_value_map();
        auto transpose = pattern_map[m_transpose].get_node_shared_ptr();
        auto fq = pattern_map[m_fq].get_node_shared_ptr();

        auto are_inputs_scalars =
            shape_size(fq->input_value(1).get_shape()) == 1 && shape_size(fq->input_value(2).get_shape()) == 1 &&
            shape_size(fq->input_value(3).get_shape()) == 1 && shape_size(fq->input_value(4).get_shape()) == 1;
        if (!are_inputs_scalars) {
            auto perm =
                std::dynamic_pointer_cast<opset1::Constant>(pattern_map[m_transpose_perm].get_node_shared_ptr());
            if (!perm)
                return false;
            auto perm_val = perm->cast_vector<int64_t>();
            if (!(perm_val[0] == 0 && perm_val[1] == 1))
                return false;
        }

        auto input_rank = fq->input(0).get_partial_shape().rank().get_length();

        ngraph::NodeVector new_ops;
        ngraph::OutputVector fq_inputs;
        for (size_t i = 0; i < fq->inputs().size(); ++i) {
            auto fq_input = fq->input_value(i);
            auto fq_input_rank = fq_input.get_partial_shape().rank().get_length();
            std::vector<int64_t> unsqueeze_axes;
            for (int64_t j = 0; j < input_rank - fq_input_rank; ++j) {
                unsqueeze_axes.push_back(j);
            }
            if (!unsqueeze_axes.empty()) {
                fq_input = std::make_shared<ngraph::opset1::Unsqueeze>(
                    fq_input,
                    opset1::Constant::create(element::i64, Shape{unsqueeze_axes.size()}, unsqueeze_axes));
                new_ops.push_back(fq_input.get_node_shared_ptr());
            }
            fq_input = std::make_shared<opset1::Transpose>(fq_input, transpose->input_value(1));
            if (auto constant = get_constant_from_source(fq_input)) {
                fq_input = constant;
            }
            ngraph::copy_runtime_info(transpose, fq_input.get_node_shared_ptr());
            fq_inputs.push_back(fq_input);
        }

        auto new_fq = fq->clone_with_new_inputs(fq_inputs);
        new_ops.push_back(new_fq);
        new_fq->set_friendly_name(transpose->get_friendly_name());
        ngraph::copy_runtime_info({fq, transpose}, new_ops);
        ngraph::replace_node(transpose, new_fq);
        MATCHER_SCOPE_ENABLE(PullTransposeThroughFQUp);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(m_transpose, matcher_name);
    this->register_matcher(m, callback);
}
