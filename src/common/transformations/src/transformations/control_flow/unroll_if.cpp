// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/control_flow/unroll_if.hpp"

#include <memory>
#include <utility>

#include "itt.hpp"
#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/binary_elementwise_comparison.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace {
// Evaluates self-comparison (x op x) where both inputs are identical.
// Returns {is_self_comparison, result}.
// Self-comparison semantics (for non-NaN values):
//   x == x  -> true    x != x  -> false
//   x <= x  -> true    x <  x  -> false
//   x >= x  -> true    x >  x  -> false
//
// NOTE: For floating-point types, IEEE 754 specifies that NaN != NaN is TRUE.
// To ensure IEEE 754 compliance, this optimization is only applied to integer types
// where NaN values are impossible. For float types, we skip the optimization to
// preserve correct NaN semantics if the runtime values happen to contain NaN.
std::pair<bool, bool> evaluate_self_comparison(const std::shared_ptr<ov::Node>& cond_node) {
    auto comparison = ov::as_type_ptr<ov::op::util::BinaryElementwiseComparison>(cond_node);
    if (!comparison) {
        return {false, false};
    }

    const auto& input0 = comparison->input_value(0);
    const auto& input1 = comparison->input_value(1);

    // Check if both inputs come from the same source (same node and same output index)
    if (input0 != input1) {
        return {false, false};
    }

    // For floating-point types, skip optimization to preserve IEEE 754 NaN semantics.
    // NaN != NaN is TRUE per IEEE 754, so we cannot assume x != x is always false for floats.
    // Integer types cannot contain NaN, so the optimization is safe for them.
    auto input_type = comparison->get_input_element_type(0);
    if (input_type.is_real()) {
        return {false, false};
    }

    // For integer types: reflexive comparisons (==, <=, >=) return true;
    // irreflexive (!=, <, >) return false
    if (ov::is_type<ov::op::v1::Equal>(cond_node) || ov::is_type<ov::op::v1::LessEqual>(cond_node) ||
        ov::is_type<ov::op::v1::GreaterEqual>(cond_node)) {
        return {true, true};
    }

    if (ov::is_type<ov::op::v1::NotEqual>(cond_node) || ov::is_type<ov::op::v1::Less>(cond_node) ||
        ov::is_type<ov::op::v1::Greater>(cond_node)) {
        return {true, false};
    }

    return {false, false};
}
}  // namespace

bool ov::pass::UnrollIf::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(UnrollIf);
    bool is_applicable = false;
    for (const auto& op : f->get_ordered_ops()) {
        auto multisubgraph_op = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(op);
        if (multisubgraph_op) {
            for (size_t i = 0; i < multisubgraph_op->get_internal_subgraphs_size(); ++i) {
                run_on_model(multisubgraph_op->get_function(static_cast<int>(i)));
            }
        }
        auto if_node = ov::as_type_ptr<ov::op::v8::If>(op);
        if (!if_node || transformation_callback(if_node)) {
            continue;
        }
        Output<Node> cond = if_node->input_value(0);

        // First, try to get condition as a constant (existing logic)
        const auto cond_is_const = ov::util::get_constant_from_source(cond);

        bool cond_value_bool = false;
        if (!cond_is_const) {
            // If not a constant, check for self-comparison pattern (x op x)
            auto [is_self_cmp, result] = evaluate_self_comparison(cond.get_node_shared_ptr());
            if (!is_self_cmp) {
                continue;
            }
            cond_value_bool = result;
        } else {
            cond_value_bool = cond_is_const->cast_vector<bool>()[0];
        }

        auto body = cond_value_bool ? if_node->get_then_body() : if_node->get_else_body();
        auto input_descriptions = if_node->get_input_descriptions(static_cast<int>(!cond_value_bool));
        auto output_descriptions = if_node->get_output_descriptions(static_cast<int>(!cond_value_bool));
        // copy rt info before reconnection
        for (auto& op : body->get_ops())
            copy_runtime_info({op, if_node}, op);

        // connect inputs instead of body parameters
        for (const auto& input_descr : input_descriptions) {
            auto in_data = if_node->input_value(input_descr->m_input_index);
            auto& param = body->get_parameters()[input_descr->m_body_parameter_index];
            for (const auto& input : param->output(0).get_target_inputs()) {
                input.replace_source_output(in_data);
            }
        }
        for (const auto& output_desc : output_descriptions) {
            std::shared_ptr<ov::op::v0::Result> result = body->get_results()[output_desc->m_body_value_index];

            for (const auto& input : if_node->output(output_desc->m_output_index).get_target_inputs()) {
                input.replace_source_output(result->get_input_source_output(0));
            }
        }
        is_applicable = true;
        f->add_sinks(body->get_sinks());
    }
    return is_applicable;
}
