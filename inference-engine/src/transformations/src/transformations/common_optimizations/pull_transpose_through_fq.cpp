// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/common_optimizations/pull_transpose_through_fq.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::PullTransposeThroughFQUp, "PullTransposeThroughFQUp", 0);

ngraph::pass::PullTransposeThroughFQUp::PullTransposeThroughFQUp() {
    MATCHER_SCOPE(PullTransposeThroughFQUp);
    auto m_fq = pattern::wrap_type<opset1::FakeQuantize>({pattern::any_input(pattern::has_static_rank()),
                                                          pattern::any_input(pattern::has_static_rank()),
                                                          pattern::any_input(pattern::has_static_rank()),
                                                          pattern::any_input(pattern::has_static_rank()),
                                                          pattern::any_input(pattern::has_static_rank())},
                                                          pattern::consumers_count(1));
    auto m_transpose = pattern::wrap_type<opset1::Transpose>({m_fq, pattern::wrap_type<opset1::Constant>()});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto & pattern_map = m.get_pattern_value_map();
        auto transpose = pattern_map[m_transpose].get_node_shared_ptr();
        auto fq = pattern_map[m_fq].get_node_shared_ptr();

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
                fq_input = std::make_shared<ngraph::opset1::Unsqueeze>(fq_input,
                                                                       opset1::Constant::create(element::i64, Shape{unsqueeze_axes.size()}, unsqueeze_axes));
                new_ops.push_back(fq_input.get_node_shared_ptr());
            }
            fq_input = transpose->copy_with_new_inputs({fq_input, transpose->input_value(1)});
            ngraph::copy_runtime_info(transpose, fq_input.get_node_shared_ptr());
            fq_inputs.push_back(fq_input);
        }

        auto new_fq = fq->copy_with_new_inputs(fq_inputs);
        new_ops.push_back(new_fq);
        new_fq->set_friendly_name(fq->get_friendly_name());
        ngraph::copy_runtime_info({fq, transpose}, new_ops);
        ngraph::replace_node(transpose, new_fq);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(m_transpose, matcher_name);
    this->register_matcher(m, callback);
}
