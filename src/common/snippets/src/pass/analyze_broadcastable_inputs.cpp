// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/analyze_broadcastable_inputs.hpp"

#include "snippets/lowered/pass/insert_broadcastmove.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace pass {

AnalyzeBroadcastableInputs::AnalyzeBroadcastableInputs(BroadcastableInputsMap& map) : m_broadcastable_inputs(map) {}

bool pass::AnalyzeBroadcastableInputs::run_on_model(const std::shared_ptr<ov::Model>& body) {
    RUN_ON_MODEL_SCOPE(AnalyzeBroadcastableInputs);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::AnalyzeBroadcastableInputs")
    // Snippets supports tokenization of the following operations:
    // - Unary, Binary and Ternary (Select) Elementwise ops
    // - Softmax, MatMul, Transpose, GroupNorm
    // Binary Elementwise ops (+ Select) requires explicit Broadcast op
    // on inputs if broadcasting of latest dimensions is needed.
    // These ops will be start points of DFS - need to go to Parameters and update `broadcastable_inputs_map`.
    // We iterates through all ops by execution order. So if we already analyzied some op in the input branch - skip this branch.
    // However, there some ops which can change `processing_dim_idx`:
    // - Transpose has order which changes `processing_dim_idx`. But Transpose can be only after Parameters and before Results.
    // - MatMul's first input doesn't affect output latest dimension - skip this branch.
    //   Also MatMul has `transposed_b` which changes `processing_dim_idx`
    m_broadcastable_inputs.clear();
    // Currently Broadcasting can be changed only if there are several Parameters in body
    if (body->get_parameters().size() < 2)
        return false;

    const auto& ops = body->get_ordered_ops();
    std::set<std::shared_ptr<ov::Node>> visited_ops = {};
    for (const auto& op : ops) {
        if (!ov::snippets::lowered::pass::InsertBroadcastMove::is_broadcasting_supported(op))
            continue;

        size_t processing_dim_idx = 0;

        // We need to propagate `processing_dim_idx` from input of the current node to the parameter.
        // To do it we use DFS
        std::stack<std::shared_ptr<ov::Node>> nodes_to_calculate;
        nodes_to_calculate.push(op);
        while (!nodes_to_calculate.empty()) {
            auto current_node = nodes_to_calculate.top();
            nodes_to_calculate.pop();

            if (const auto& param = ov::as_type_ptr<ov::op::v0::Parameter>(current_node)) {
                const auto consumers = param->get_output_target_inputs(0);
                if (std::any_of(consumers.cbegin(), consumers.cend(),
                                [](const ov::Input<ov::Node>& in) { return ov::is_type<ov::op::v1::Transpose>(in.get_node()); })) {
                    OPENVINO_ASSERT(consumers.size() == 1, "Incorrect count of outputs of Parameter!");
                    const auto transpose = consumers.begin()->get_node();
                    std::vector<size_t> order;
                    const auto& constant = ov::as_type_ptr<const opset1::Constant>(transpose->get_input_node_shared_ptr(1));
                    OPENVINO_ASSERT(constant, "Unsupported order node of Transpose");
                    order = constant->cast_vector<size_t>();
                    if (order.empty()) {
                        order.resize(transpose->get_output_partial_shape(0).size());
                        std::iota(order.rbegin(), order.rend(), 0);
                    }
                    // `processing_dim_idx` starts from the end
                    processing_dim_idx = order.size() - 1 - ov::snippets::utils::get_input_dim_idx(order, processing_dim_idx);
                }
                const auto param_idx = body->get_parameter_index(param);
                if (m_broadcastable_inputs.count(param_idx) == 0) {
                    m_broadcastable_inputs[param_idx] = processing_dim_idx;
                } else {
                    OPENVINO_ASSERT(m_broadcastable_inputs.at(param_idx) == processing_dim_idx,
                                    "Parameter has been already analyzed and has another processing dim index!");
                }
                processing_dim_idx = 0;
                continue;
            } else if (ov::is_type<ov::op::v0::Constant>(current_node)) {
                visited_ops.insert(op);
                continue;
            }

            ov::OutputVector inputs = current_node->input_values();
            if (const auto mm = ov::as_type_ptr<ov::op::v0::MatMul>(current_node)) {
                inputs = { current_node->input_value(1) };
                processing_dim_idx = static_cast<size_t>(mm->get_transpose_b());
            }

            // not a leaf - continue to search
            for (const auto& input_value : inputs) {
                const auto& input_node = input_value.get_node()->shared_from_this();
                if (visited_ops.count(input_node) == 0) {
                    nodes_to_calculate.push(input_node);
                }
            }
        }

        visited_ops.insert(op);
    }

    return true;
}

} // namespace pass
} // namespace snippets
} // namespace ov