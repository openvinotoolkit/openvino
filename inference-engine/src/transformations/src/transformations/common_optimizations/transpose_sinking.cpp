// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/common_optimizations/transpose_sinking.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset6.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <numeric>

NGRAPH_RTTI_DEFINITION(ngraph::pass::TransposeSinking, "TransposeSinking", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::TransposeReduction, "TransposeReduction", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::TransposeFQReduction, "TransposeFQReduction", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::TransposeFuse, "TransposeFuse", 0);

using namespace ngraph;

std::shared_ptr<ngraph::opset6::Constant> get_reduced_order_constant(const std::shared_ptr<ngraph::opset6::Constant>& axes_const,
                                                                     const std::shared_ptr<ngraph::opset6::Constant>& order_const) {
    auto order = order_const->cast_vector<int64_t>();

    auto axes = axes_const->cast_vector<int64_t>();
    std::sort(axes.rbegin(), axes.rend());
    for (const auto& i : axes)
        order.erase(order.begin() + i);

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

ngraph::pass::TransposeReduction::TransposeReduction() {
    MATCHER_SCOPE(TransposeReduction);

    auto transpose_label = pattern::wrap_type<opset6::Transpose>({pattern::any_input(), pattern::wrap_type<opset6::Constant>()});
    auto reduce_or_squeeze_label = pattern::wrap_type<op::util::ArithmeticReductionKeepDims, op::util::LogicalReductionKeepDims, opset6::Squeeze>(
            {transpose_label, pattern::wrap_type<opset6::Constant>()});

    ngraph::matcher_pass_callback matcher_pass_callback = [=](ngraph::pattern::Matcher &m) {
        const auto &pattern_to_output = m.get_pattern_value_map();

        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto reduction = pattern_to_output.at(reduce_or_squeeze_label).get_node_shared_ptr();
        auto arithmetic_reduce = std::dynamic_pointer_cast<op::util::ArithmeticReductionKeepDims>(reduction);
        auto logical_reduce = std::dynamic_pointer_cast<op::util::LogicalReductionKeepDims>(reduction);
        auto squeeze = std::dynamic_pointer_cast<opset6::Squeeze>(reduction);
        if (!transpose || !(arithmetic_reduce || logical_reduce || squeeze))
            return false;

        bool keep_dims = false; // squeeze always reduces number of output dimensions
        if (logical_reduce)
            keep_dims = logical_reduce->get_keep_dims();
        else if (arithmetic_reduce)
            keep_dims = arithmetic_reduce->get_keep_dims();

        auto transpose_order = std::dynamic_pointer_cast<ngraph::opset6::Constant>(transpose->get_input_node_shared_ptr(1));
        auto reduction_axes = std::dynamic_pointer_cast<ngraph::opset6::Constant>(reduction->get_input_node_shared_ptr(1));
        if (!transpose_order || !reduction_axes)
            return false;

        const auto& non_negative_axes = ngraph::normalize_axes(
                reduction->get_friendly_name(), reduction_axes->cast_vector<int64_t>(), reduction->get_input_partial_shape(0).rank());
        reduction_axes = ngraph::opset6::Constant::create(ngraph::element::i64, {non_negative_axes.size()}, non_negative_axes);

        ngraph::NodeVector new_ops;
        auto new_axes = ngraph::op::util::make_try_fold<ngraph::opset6::Gather>(
                transpose_order, reduction_axes, ngraph::opset6::Constant::create(ngraph::element::i64, {}, {0}));
        new_ops.push_back(new_axes);
        auto new_reduce = reduction->copy_with_new_inputs({transpose->input_value(0), new_axes});
        new_ops.push_back(new_reduce);

        auto updated_order = transpose_order;
        if (!keep_dims) {
            updated_order = get_reduced_order_constant(reduction_axes, transpose_order);
            new_ops.push_back(updated_order);
        }
        auto new_transpose = register_new_node<opset6::Transpose>(new_reduce, updated_order);
        new_ops.push_back(new_transpose);
        new_transpose->set_friendly_name(reduction->get_friendly_name());

        ngraph::copy_runtime_info({reduction, transpose}, new_ops);
        ngraph::replace_node(reduction, new_transpose);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reduce_or_squeeze_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ngraph::pass::TransposeFQReduction::TransposeFQReduction() {
    MATCHER_SCOPE(TransposeFQReduction);

    auto transpose_label = pattern::wrap_type<opset6::Transpose>({pattern::any_input(), pattern::wrap_type<opset6::Constant>()});
    auto fq_label = pattern::wrap_type<opset6::FakeQuantize>(
            {transpose_label, pattern::any_input(pattern::has_static_rank()), pattern::any_input(pattern::has_static_rank()),
                              pattern::any_input(pattern::has_static_rank()), pattern::any_input(pattern::has_static_rank())});
    auto reduce_or_squeeze_label = pattern::wrap_type<op::util::ArithmeticReductionKeepDims, op::util::LogicalReductionKeepDims, opset6::Squeeze>(
            {fq_label, pattern::wrap_type<opset6::Constant>()});

    ngraph::matcher_pass_callback matcher_pass_callback = [=](ngraph::pattern::Matcher &m) {
        auto &pattern_to_output = m.get_pattern_value_map();

        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto transpose_order = std::dynamic_pointer_cast<opset6::Constant>(transpose->get_input_node_shared_ptr(1));
        auto fq = pattern_to_output.at(fq_label).get_node_shared_ptr();
        if (!transpose || !transpose_order || !fq)
            return false;

        ngraph::NodeVector new_ops;

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
        // The root node (reduction) left unchanged during current matcher pass.
        // We return false here for further MatcherPasses to be applicable for this node as a root node
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reduce_or_squeeze_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ngraph::pass::TransposeFuse::TransposeFuse() {
    MATCHER_SCOPE(TransposeFuse);

    auto transpose_1 = pattern::wrap_type<opset7::Transpose>({ pattern::any_input(), pattern::wrap_type<opset7::Constant>() }, pattern::consumers_count(1));
    auto transpose_2 = pattern::wrap_type<opset7::Transpose>({ transpose_1, pattern::wrap_type<opset7::Constant>() });

    ngraph::matcher_pass_callback matcher_pass_callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto transpose1 = pattern_to_output.at(transpose_1).get_node_shared_ptr();
        auto transpose2 = pattern_to_output.at(transpose_2).get_node_shared_ptr();
        auto input = transpose1->input_value(0);

        auto transpose1_order = std::dynamic_pointer_cast<ngraph::opset7::Constant>(transpose1->get_input_node_shared_ptr(1));
        auto transpose2_order = std::dynamic_pointer_cast<ngraph::opset7::Constant>(transpose2->get_input_node_shared_ptr(1));
        if (!transpose1_order || !transpose2_order)
            return false;

        auto order1 = transpose1_order->cast_vector<int64_t>();
        auto order2 = transpose2_order->cast_vector<int64_t>();
        if (order1.size() != order2.size())
            return false;

        bool is_ordered = true;
        for (size_t i = 0; i < order1.size(); i++) {
            order2[i] = order1[order2[i]];
            if (order2[i] != (int64_t)i)
                is_ordered = false;
        }

        if (is_ordered) {
            return ngraph::replace_output_update_name(transpose2->output(0), input);
        } else {
            auto new_order = ngraph::opset7::Constant::create(element::i64, {order2.size()}, order2);
            auto new_transpose = register_new_node<ngraph::opset7::Transpose>(input, new_order);

            ngraph::copy_runtime_info({ transpose1, transpose2 }, new_transpose);
            ngraph::replace_node(transpose2, new_transpose);
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose_2, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
