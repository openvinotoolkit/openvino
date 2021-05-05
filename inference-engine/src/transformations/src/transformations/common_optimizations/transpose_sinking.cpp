// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/common_optimizations/transpose_sinking.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset6.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <numeric>

NGRAPH_RTTI_DEFINITION(ngraph::pass::TransposeSinking, "TransposeSinking", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::TransposeOptimization, "TransposeOptimization", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::TransposeReduction, "TransposeReduction", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::TransposeFQReduction, "TransposeFQReduction", 0);

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


bool replace_transpose_with_reshape(const std::shared_ptr<Node>& transpose) {
    auto data = transpose->input_value(0);
    const auto input_shape = transpose->input(0).get_partial_shape();

    const size_t input_shape_rank = input_shape.rank().get_length();

    auto order = as_type_ptr<opset6::Constant>(transpose->input_value(1).get_node_shared_ptr());
    if (!order || !ngraph::shape_size(order->get_shape())) {
        return false;
    }

    const auto order_value = order->cast_vector<int64_t>();

    // Check that transpose order without 1 dims has an ascending order
    int64_t last_dim(-1);
    for (size_t i = 0; i < input_shape_rank; ++i) {
        if (input_shape[order_value[i]].is_dynamic() || input_shape[order_value[i]] != 1) {
            if (order_value[i] < last_dim) {
                return false;
            }
            last_dim = order_value[i];
        }
    }

    // Transpose operation can be removed if original transpose order is sorted
    // or dimension that changes their places equal to 1
    using DimensionToPosition = struct {
        Dimension dim;
        size_t pos;
    };
    std::vector<DimensionToPosition> dims;
    for (size_t i = 0; i < input_shape_rank; ++i) {
        if (order_value[i] != static_cast<int64_t>(i)) {
            dims.push_back({input_shape[order_value[i]], i});
        }
    }

    // If number of dimensions != 1 to move equal to 0 we can remove this Transpose
    if (count_if(dims.begin(), dims.end(), [](const DimensionToPosition& item) {
        return !(item.dim.is_static() && item.dim.get_length() == 1);
    }) == 0) {
        return replace_output_update_name(transpose->output(0), transpose->input_value(0));
    }

    // Transpose can be replaced with Reshape in two ways:
    // 1. Reshape with dims as Constant
    // 2. Reshape with dims as input (ShapeOf->Gather)
    //
    // The first case is possible only if one or less dynamic dimensions changes their position
    // For example: input_shape {?, 3, 1, ?} and order {0, 1, 3, 2} can be replaced with Reshape
    // with Constant {0, 3, -1, 1} but if input_shape {?, 1, 1, ?} and order {1, 0, 3, 2} transpose
    // cannot be replaced int the same way and in this case its only possible to use Gather(ShapeOf,
    // order)

    Output<Node> reshape_dim;
    NodeVector new_ops;

    if (count_if(dims.begin(), dims.end(), [](const DimensionToPosition& item) {
        return item.dim.is_dynamic();
    }) < 2) {
        std::vector<int64_t> reshape_value(input_shape_rank, 0);
        for (const auto& item : dims) {
            reshape_value[item.pos] = item.dim.is_dynamic() ? -1 : item.dim.get_length();
        }
        reshape_dim =
                opset3::Constant::create(element::i64, Shape{reshape_value.size()}, reshape_value);
    } else {
        auto shape_of = std::make_shared<opset3::ShapeOf>(data);
        new_ops.push_back(shape_of);
        reshape_dim = std::make_shared<opset3::Gather>(
                shape_of, order, opset3::Constant::create(element::i64, Shape{1}, {0}));
        new_ops.push_back(reshape_dim.get_node_shared_ptr());
    }

    auto reshape_op = std::make_shared<opset3::Reshape>(data, reshape_dim, true);
    new_ops.push_back(reshape_op);

    reshape_op->set_friendly_name(transpose->get_friendly_name());
    copy_runtime_info(transpose, new_ops);
    replace_node(transpose, reshape_op);
    return true;
}

ngraph::pass::TransposeOptimization::TransposeOptimization() {
    MATCHER_SCOPE(TransposeOptimization);

    auto transpose_label = pattern::wrap_type<opset6::Transpose>(
            {pattern::any_input(pattern::has_static_rank()), pattern::wrap_type<opset6::Constant>()});
    ngraph::matcher_pass_callback matcher_pass_callback = [=](ngraph::pattern::Matcher &m) {
        return replace_transpose_with_reshape(m.get_match_root());
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
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
