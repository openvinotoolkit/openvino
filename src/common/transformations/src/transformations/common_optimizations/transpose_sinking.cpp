// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/transpose_sinking.hpp"

#include <memory>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <numeric>
#include <openvino/core/validation_util.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset7.hpp>
#include <vector>

#include "itt.hpp"
#include "transformations/common_optimizations/transpose_sinking_utils.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;

namespace {

std::shared_ptr<opset6::Constant> get_reversed_order_constant(const std::shared_ptr<opset6::Constant>& order_const) {
    const auto& order = order_const->cast_vector<size_t>();
    const auto& rank = order.size();
    const auto& default_order = ngraph::get_default_order(rank);
    std::vector<size_t> reverse_order(rank);
    for (size_t i = 0; i < rank; ++i)
        reverse_order[order[i]] = default_order[i];

    return std::make_shared<opset6::Constant>(ngraph::element::i64, ngraph::Shape{reverse_order.size()}, reverse_order);
}

}  // namespace

ov::pass::TransposeEltwise::TransposeEltwise() {
    MATCHER_SCOPE(TransposeEltwise);

    auto eltwise_data_input_p = pattern::any_input();
    auto eltwise_const_input_p = pattern::wrap_type<opset6::Constant>();
    auto eltwise_p = pattern::wrap_type<op::util::BinaryElementwiseArithmetic>(
        {eltwise_data_input_p, eltwise_const_input_p},
        [](const Output<Node>& output) {
            return ov::is_preprocesing_node(output.get_node_shared_ptr());
        });
    auto transpose_p = pattern::wrap_type<opset6::Transpose>({eltwise_p, pattern::wrap_type<opset6::Constant>()},
                                                             pattern::consumers_count(1));

    auto callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto eltwise = pattern_to_output.at(eltwise_p).get_node_shared_ptr();
        auto eltwise_const_input = pattern_to_output.at(eltwise_const_input_p);
        auto eltwise_data_input = pattern_to_output.at(eltwise_data_input_p);
        auto transpose = pattern_to_output.at(transpose_p).get_node_shared_ptr();

        const auto& order_size = transpose->get_input_shape(1).at(0);
        const auto& shape = eltwise_const_input.get_shape();
        if (shape.size() != order_size && ov::shape_size(shape) != 1) {
            // TODO: temporary restrictions
            return false;
        }

        if (ov::shape_size(shape) != 1) {
            eltwise_const_input = std::make_shared<opset6::Transpose>(eltwise_const_input, transpose->input_value(1));
            if (auto const_node = ov::get_constant_from_source(eltwise_const_input)) {
                eltwise_const_input = const_node;
            }
        }

        auto new_transpose = transpose->clone_with_new_inputs({eltwise_data_input, transpose->input_value(1)});
        auto new_eltwise = eltwise->clone_with_new_inputs({new_transpose, eltwise_const_input});
        register_new_node(new_transpose);

        new_transpose->set_friendly_name(eltwise->get_friendly_name());
        copy_runtime_info({eltwise, transpose}, {new_transpose, new_eltwise});
        replace_node(transpose, new_eltwise);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose_p, matcher_name);
    register_matcher(m, callback);
}

ov::pass::TransposeConvert::TransposeConvert() {
    MATCHER_SCOPE(TransposeConvert);

    auto transpose_label =
        pattern::wrap_type<opset6::Transpose>({pattern::any_input(), pattern::wrap_type<opset6::Constant>()},
                                              pattern::consumers_count(1));
    auto convert_label = pattern::wrap_type<opset6::Convert>({transpose_label});

    matcher_pass_callback matcher_pass_callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto convert = pattern_to_output.at(convert_label).get_node_shared_ptr();

        auto new_convert = convert->clone_with_new_inputs({transpose->input_value(0)});
        auto new_transpose = transpose->clone_with_new_inputs({new_convert, transpose->input_value(1)});
        register_new_node(new_transpose);

        new_transpose->set_friendly_name(convert->get_friendly_name());
        copy_runtime_info({transpose, convert}, {new_convert, new_transpose});
        replace_node(convert, new_transpose);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(convert_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::TransposeFQReduction::TransposeFQReduction() {
    MATCHER_SCOPE(TransposeFQReduction);

    auto transpose_label =
        pattern::wrap_type<opset6::Transpose>({pattern::any_input(), pattern::wrap_type<opset6::Constant>()});
    auto fq_label = pattern::wrap_type<opset6::FakeQuantize>({transpose_label,
                                                              pattern::any_input(pattern::has_static_rank()),
                                                              pattern::any_input(pattern::has_static_rank()),
                                                              pattern::any_input(pattern::has_static_rank()),
                                                              pattern::any_input(pattern::has_static_rank())});
    auto reduce_or_squeeze_label =
        pattern::wrap_type<op::util::ArithmeticReductionKeepDims, op::util::LogicalReductionKeepDims, opset6::Squeeze>(
            {fq_label, pattern::wrap_type<opset6::Constant>()});

    ov::matcher_pass_callback matcher_pass_callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();

        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        if (!transpose)
            return false;

        auto transpose_order = std::dynamic_pointer_cast<opset6::Constant>(transpose->get_input_node_shared_ptr(1));
        auto fq = pattern_to_output.at(fq_label).get_node_shared_ptr();
        if (!transpose_order || !fq)
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
        auto new_fq = fq->clone_with_new_inputs(fq_inputs);
        new_ops.push_back(new_fq);

        auto new_transpose = register_new_node<opset6::Transpose>(new_fq, transpose_order);
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
