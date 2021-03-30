// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/common_optimizations/transpose_sinking_fq_reduce.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset6.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <numeric>

NGRAPH_RTTI_DEFINITION(ngraph::pass::TransposeSinkingFQReduce, "TransposeSinkingFQReduce", 0);

std::shared_ptr<ngraph::opset6::Constant> get_reversed_order_constant(const std::shared_ptr<ngraph::opset6::Constant>& order_const) {
    const auto& order = order_const->cast_vector<size_t>();
    const auto& rank = order.size();
    const auto& default_order = ngraph::get_default_order(rank);
    std::vector<size_t> reverse_order(rank);
    for (size_t i = 0; i < rank; ++i)
        reverse_order[order[i]] = default_order[i];

    return std::make_shared<ngraph::opset6::Constant>(
            ngraph::element::i64, ngraph::Shape{reverse_order.size()}, reverse_order);
}

std::shared_ptr<ngraph::opset6::Constant> get_reduced_order_constant(const std::shared_ptr<ngraph::opset6::Constant>& axes_const,
                                                                     const std::shared_ptr<ngraph::opset6::Constant>& order_const) {
    auto order = order_const->cast_vector<int64_t>();
    auto axes = axes_const->cast_vector<int64_t>();
    for (const auto& i : axes)
        order.erase(std::remove(order.begin(), order.end(), i), order.end());

    const auto& updated_order_size = static_cast<int64_t>(order.size());

    auto order_sorted = order;
    sort(order_sorted.begin(), order_sorted.end());
    for (int64_t i = 0; i < updated_order_size; ++i) {
        auto lowest_greater_eq_i = std::lower_bound(order_sorted.begin(), order_sorted.end(), i);
        std::replace(order.begin(), order.end(), *lowest_greater_eq_i, i);
        std::replace(order_sorted.begin(), order_sorted.end(), *lowest_greater_eq_i, i);
    }
    return std::make_shared<ngraph::opset6::Constant>(
            ngraph::element::i64, ngraph::Shape{order.size()}, order);
}


ngraph::pass::TransposeSinkingFQReduce::TransposeSinkingFQReduce() {
    MATCHER_SCOPE(TransposeSinkingFQReduce);

    auto transpose_label = pattern::wrap_type<opset6::Transpose>({pattern::any_input(), pattern::wrap_type<opset6::Constant>()});
    auto fq_label = pattern::wrap_type<opset6::FakeQuantize>(
            {transpose_label, pattern::any_input(pattern::has_static_rank()), pattern::any_input(pattern::has_static_rank()),
                              pattern::any_input(pattern::has_static_rank()), pattern::any_input(pattern::has_static_rank())});
//    auto reduce_label = pattern::wrap_type<op::util::ArithmeticReductionKeepDims>({fq_label, pattern::wrap_type<opset6::Constant>()});
    auto reduce_label = pattern::wrap_type<opset6::ReduceMean>({fq_label, pattern::wrap_type<opset6::Constant>()});

    ngraph::matcher_pass_callback matcher_pass_callback = [=](ngraph::pattern::Matcher &m) {
        auto &pattern_to_output = m.get_pattern_value_map();

        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto transpose_order = std::dynamic_pointer_cast<opset6::Constant>(transpose->get_input_node_shared_ptr(1));
        auto fq = pattern_to_output.at(fq_label).get_node_shared_ptr();
        auto reduce = std::dynamic_pointer_cast<op::util::ArithmeticReductionKeepDims>(
                pattern_to_output.at(reduce_label).get_node_shared_ptr());
        auto reduce_axes = std::dynamic_pointer_cast<opset6::Constant>(reduce->get_input_node_shared_ptr(1));
        if (!transpose || !transpose_order || !fq || !reduce || !reduce_axes)
            return false;

        ngraph::NodeVector new_ops;

        // Transpose -> FQ  == FQ -> Transpose
        const auto& reverse_order_constant = get_reversed_order_constant(transpose_order);
        new_ops.push_back(reverse_order_constant);

        const auto& input_rank = fq->get_input_partial_shape(0).rank().get_length();
        ngraph::OutputVector fq_inputs = {transpose->input_value(0)};
        for (size_t i = 1; i < fq->inputs().size(); ++i) {
            auto input = fq->input_value(i);
            const auto& ranks_diff = input_rank - input.get_partial_shape().rank().get_length();
            NGRAPH_CHECK(ranks_diff >= 0);
            if (ranks_diff > 0) {
                std::vector<int64_t> axes(ranks_diff);
                std::iota(axes.begin(), axes.end(), 0);
                const auto& axes_const = opset6::Constant::create(element::i64, Shape{axes.size()}, axes);
                new_ops.push_back(axes_const);
                const auto& unsqueezed_input = op::util::make_try_fold<opset6::Unsqueeze>(input, axes_const);
                new_ops.push_back(unsqueezed_input);
                input = unsqueezed_input->output(0);
            }
            const auto& transposed_input = op::util::make_try_fold<opset6::Transpose>(input, reverse_order_constant);
            new_ops.push_back(transposed_input);
            fq_inputs.push_back(transposed_input);
        }
        auto new_fq = fq->copy_with_new_inputs(fq_inputs);
        new_ops.push_back(new_fq);

        auto new_transpose = std::make_shared<ngraph::opset6::Transpose>(new_fq, transpose_order);
        new_ops.push_back(new_transpose);
        new_transpose->set_friendly_name(fq->get_friendly_name());

        ngraph::copy_runtime_info({fq, transpose}, new_ops);
        ngraph::replace_node(fq, new_transpose);

        new_ops.clear();

        // Transpose -> Reduce == Reduce -> Transpose
        // new_transpose and reduce

        auto new_axes = op::util::make_try_fold<opset6::Gather>(
                transpose_order, reduce_axes, opset6::Constant::create(element::i64, Shape{}, {0}));
        new_ops.push_back(new_axes);
        auto new_reduce = reduce->copy_with_new_inputs({new_transpose->input_value(0), new_axes});
        new_ops.push_back(new_reduce);

        auto updated_order = transpose_order;
        if (!reduce->get_keep_dims()) {
            updated_order = get_reduced_order_constant(reduce_axes, transpose_order);
            new_ops.push_back(updated_order);
        }
        auto final_transpose = new_transpose->copy_with_new_inputs({new_reduce, updated_order});
        new_ops.push_back(final_transpose);

        final_transpose->set_friendly_name(reduce->get_friendly_name());

        ngraph::copy_runtime_info({reduce, new_transpose}, new_ops);
        ngraph::replace_node(reduce, final_transpose);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reduce_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
