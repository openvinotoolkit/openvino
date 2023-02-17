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

std::vector<size_t> get_updated_order(std::vector<size_t>& axes_values,
                                       std::vector<size_t>& order_values,
                                       bool is_forward) {
    std::sort(axes_values.begin(), axes_values.end());
    size_t buffer_size = is_forward ? order_values.size() - axes_values.size() : order_values.size() + axes_values.size();

    std::vector<size_t> aligned_order(buffer_size);
    for (size_t i = 0, j = 0; i < std::max(aligned_order.size(), order_values.size()); ++i) {
        if (std::find(axes_values.begin(), axes_values.end(), i) != axes_values.end()) {
            if (is_forward) {
                continue;
            } else {
                aligned_order[i] = i;
                continue;
            }
        }

        if (is_forward) {
            auto ub = std::upper_bound(axes_values.begin(), axes_values.end(), order_values[i]);
            aligned_order[j] = order_values[i] - (ub - axes_values.begin());
        } else {
            auto ub = std::upper_bound(axes_values.begin(), axes_values.end(), order_values[j]);
            aligned_order[i] = order_values[j] + (ub - axes_values.begin());
        }
        ++j;
    }
    std::cout << "new order" << std::endl;
    for (const auto& it : aligned_order) {
        std::cout << it << " ";
    }
    std::cout << std::endl;
    return aligned_order;
}

bool get_keep_dims(const std::shared_ptr<Node>& reduction) {
    auto arithmetic_reduce = std::dynamic_pointer_cast<op::util::ArithmeticReductionKeepDims>(reduction);
    auto logical_reduce = std::dynamic_pointer_cast<op::util::LogicalReductionKeepDims>(reduction);
    auto squeeze = std::dynamic_pointer_cast<opset6::Squeeze>(reduction);

    bool keep_dims = false;  // squeeze always reduces number of output dimensions
    if (logical_reduce)
        keep_dims = logical_reduce->get_keep_dims();
    else if (arithmetic_reduce)
        keep_dims = arithmetic_reduce->get_keep_dims();
    return keep_dims;
}

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

ov::pass::TransposeReductionBackward::TransposeReductionBackward() {
    MATCHER_SCOPE(TransposeReductionBackward);

    auto reduce_or_squeeze_label =
        pattern::wrap_type<op::util::ArithmeticReductionKeepDims, op::util::LogicalReductionKeepDims>(
            {pattern::any_input(), pattern::wrap_type<opset6::Constant>()}, transpose_sinking::HasSameOutputTransposeNodes);
    auto transpose_label =
            pattern::wrap_type<opset6::Transpose>({reduce_or_squeeze_label, pattern::wrap_type<opset6::Constant>()});
    ov::matcher_pass_callback matcher_pass_callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto reduction = pattern_to_output.at(reduce_or_squeeze_label).get_node_shared_ptr();
        auto keep_dims = get_keep_dims(reduction);
        auto transpose_order = std::dynamic_pointer_cast<opset6::Constant>(transpose->get_input_node_shared_ptr(1));
        auto reduction_axes = std::dynamic_pointer_cast<opset6::Constant>(reduction->get_input_node_shared_ptr(1));
        if (!transpose_order || !reduction_axes)
            return false;

        auto non_negative_axes = normalize_axes(reduction->get_friendly_name(),
                                                reduction_axes->cast_vector<int64_t>(),
                                                reduction->get_input_partial_shape(0).rank());

        for (const auto& it : reduction->output(0).get_target_inputs()) {
            it.get_node()->output(0).replace(reduction);
        }
        auto transpose_order_values = transpose_order->cast_vector<size_t>();
        if (!keep_dims) {
            transpose_order_values = get_updated_order(non_negative_axes, transpose_order_values, false);
        }

        auto reversed_order_values = transpose_sinking::ReverseTransposeOrder(transpose_order_values);
        std::vector<size_t> new_values;
        new_values.reserve(non_negative_axes.size());
        for (const auto& axis : non_negative_axes) {
            new_values.push_back(reversed_order_values[axis]);
        }

        auto new_transpose_order = std::make_shared<opset6::Constant>(transpose_order->get_element_type(),
                                                                      Shape{transpose_order_values.size()},
                                                                      transpose_order_values);
        auto new_const = std::make_shared<opset6::Constant>(reduction_axes->get_element_type(),
                                                            reduction_axes->get_shape(),
                                                            new_values);
        transpose->input(0).replace_source_output(reduction->input_value(0));
        transpose->input(1).replace_source_output(new_transpose_order);
        reduction->input(1).replace_source_output(new_const);
        reduction->input(0).replace_source_output(transpose);
        register_new_node(transpose);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::TransposeReduction::TransposeReduction() {
    MATCHER_SCOPE(TransposeReduction);

    auto transpose_label =
        pattern::wrap_type<opset6::Transpose>({pattern::any_input(), pattern::wrap_type<opset6::Constant>()},
                                              pattern::consumers_count(1));
    auto reduce_or_squeeze_label =
        pattern::wrap_type<op::util::ArithmeticReductionKeepDims, op::util::LogicalReductionKeepDims>(
            {transpose_label, pattern::wrap_type<opset6::Constant>()});

    ov::matcher_pass_callback matcher_pass_callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto reduction = pattern_to_output.at(reduce_or_squeeze_label).get_node_shared_ptr();
        auto keep_dims = get_keep_dims(reduction);

        auto transpose_order = std::dynamic_pointer_cast<opset6::Constant>(transpose->get_input_node_shared_ptr(1));
        auto reduction_axes = std::dynamic_pointer_cast<opset6::Constant>(reduction->get_input_node_shared_ptr(1));
        if (!transpose_order || !reduction_axes)
            return false;

        auto non_negative_axes = normalize_axes(reduction->get_friendly_name(),
                                                       reduction_axes->cast_vector<int64_t>(),
                                                       reduction->get_input_partial_shape(0).rank());

        reduction->output(0).replace(transpose);
        auto transpose_order_values = transpose_order->cast_vector<size_t>();
        std::vector<size_t> new_values;
        new_values.reserve(non_negative_axes.size());
        for (const auto& axis : non_negative_axes) {
            new_values.push_back(transpose_order_values[axis]);
        }

        if (!keep_dims) {
            transpose_order_values = get_updated_order(non_negative_axes, transpose_order_values, true);
        }
        std::cout << "XXXXXX TransposeReductionForward" << std::endl;

        auto new_transpose_order = std::make_shared<opset6::Constant>(transpose_order->get_element_type(),
                                                                      Shape{transpose_order_values.size()},
                                                                      transpose_order_values);
        auto new_const = std::make_shared<opset6::Constant>(reduction_axes->get_element_type(),
                                                            reduction_axes->get_shape(),
                                                            new_values);
        reduction->input(0).replace_source_output(transpose->input_value(0));
        reduction->input(1).replace_source_output(new_const);
        transpose->input(1).replace_source_output(new_transpose_order);
        transpose->input(0).replace_source_output(reduction);
        register_new_node(transpose);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reduce_or_squeeze_label, matcher_name);
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

ov::pass::TransposeFuse::TransposeFuse() {
    MATCHER_SCOPE(TransposeFuse);
    auto transpose_label =
        pattern::wrap_type<opset7::Transpose>({pattern::any_input(), pattern::wrap_type<opset7::Constant>()});
    ov::matcher_pass_callback matcher_pass_callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();
        auto transpose_1 = pattern_to_output.at(transpose_label);
        auto order_const_1 =
            std::dynamic_pointer_cast<opset7::Constant>(transpose_1->input_value(1).get_node_shared_ptr());
        auto consumers = transpose_1->get_output_target_inputs(0);

        std::vector<int64_t> saved_order_values;
        auto saved_type = order_const_1->get_element_type();
        for (const auto& it : consumers) {
            auto out_transpose = dynamic_cast<opset7::Transpose*>(it.get_node());
            if (!out_transpose) {
                return false;
            }

            auto order = out_transpose->input_value(1).get_node_shared_ptr();
            auto order_const = std::dynamic_pointer_cast<opset7::Constant>(order);
            if (!order_const) {
                return false;
            }

            auto order_values = order_const->cast_vector<int64_t>();
            if (order_values.empty()) {
                return false;
            }

            if (saved_order_values.empty()) {
                saved_order_values = order_values;
            } else {
                if (saved_order_values != order_values) {
                    return false;
                }
            }

            if (order_const->get_element_type() != saved_type) {
                saved_type = element::i64;
            }
        }

        auto order1 = order_const_1->cast_vector<int64_t>();
        if (order1.size() != saved_order_values.size()) {
            return false;
        }

        bool is_ordered = true;
        for (size_t i = 0; i < order1.size(); i++) {
            saved_order_values[i] = order1[saved_order_values[i]];
            if (saved_order_values[i] != (int64_t)i)
                is_ordered = false;
        }

        if (is_ordered) {
            for (const auto& it : consumers) {
                it.get_node()->output(0).replace(transpose_1->input_value(0));
            }
        } else {
            auto new_order = opset7::Constant::create(saved_type, {saved_order_values.size()}, saved_order_values);
            auto new_transpose = register_new_node<opset7::Transpose>(transpose_1->input_value(0), new_order);
            for (const auto& it : consumers) {
                new_transpose->set_friendly_name(it.get_node()->get_friendly_name());
                it.get_node()->output(0).replace(new_transpose);
                copy_runtime_info(transpose_1, new_transpose);
            }
            transpose_sinking::UpdateForwardSinkingAbility(new_transpose);
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
