// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/linear_ir_builder.hpp"

#include "snippets/lowered/loop_manager.hpp"

namespace ov {
namespace snippets {
namespace lowered {

namespace {
using NodeMap = std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>>;

std::vector<std::shared_ptr<ov::Node>> clone_nodes(const std::vector<std::shared_ptr<ov::Node>>& nodes,
                                                   NodeMap& node_map) {
    // for each node in topological order
    auto sorted_nodes = topological_sort(nodes);
    for (const auto& node : sorted_nodes) {
        if (node_map.count(node.get()) == 0) {
            // get (already) cloned arguments and clone the node
            OutputVector cloned_args;
            for (auto input : node->inputs()) {
                ov::Output<Node> output = input.get_source_output();
                cloned_args.push_back(output.for_node(node_map.at(output.get_node())));
            }
            std::vector<std::shared_ptr<Node>> cloned_dependencies;
            for (auto& dependency : node->get_control_dependencies()) {
                std::shared_ptr<Node>& dependent = node_map.at(dependency.get());
                if (find(cloned_dependencies.begin(), cloned_dependencies.end(), dependent) ==
                    cloned_dependencies.end()) {
                    cloned_dependencies.push_back(dependent);
                }
            }
            auto cloned_node = node->copy_with_new_inputs(cloned_args, cloned_dependencies);
            // There is a friendly name for this node so copy it
            cloned_node->set_friendly_name(node->get_friendly_name());
            auto rt_info = node->get_rt_info();
            cloned_node->get_rt_info() = rt_info;

            for (auto output : node->outputs()) {
                const auto& output_rt_info = output.get_rt_info();
                auto new_output = output.for_node(cloned_node);
                new_output.get_rt_info() = output_rt_info;
            }

            for (auto input : node->inputs()) {
                const auto& output_rt_info = input.get_rt_info();
                auto new_input = cloned_node->input(input.get_index());
                new_input.get_rt_info() = output_rt_info;
            }

            node_map[node.get()] = cloned_node;
        }
    }

    // create and return vector of cloned nodes
    // order matches input vector (not necessarily topological)
    std::vector<std::shared_ptr<ov::Node>> cloned_nodes;
    for (const auto& node : nodes) {
        cloned_nodes.push_back(node_map.at(node.get()));
    }
    return cloned_nodes;
}
}  // namespace

void LinearIRBuilder::clone(const LinearIR* src, LinearIR* dst,  ExpressionMap& expression_map) const {
    OPENVINO_ASSERT(src && dst, "Invalid pointers were provided for LinearIRBuilder::clone");
    dst->m_config = src->m_config;

    dst->m_expressions = clone_range(src->m_expressions.cbegin(), src->m_expressions.cend(), expression_map);
    for (const auto& expr : dst->m_expressions) {
        dst->register_expression(expr, true, expr->get_exec_num());
    }

    dst->m_loop_manager = src->m_loop_manager->clone_with_new_expr(expression_map);
    // It's Ok to share shapeInfer factory ptr, since the factory doesn't depend on LIR in any way
    dst->m_shape_infer_factory = src->m_shape_infer_factory;
    dst->m_shape_infer = std::make_shared<LinearIR::LIRShapeInfer>(dst->m_expressions, dst->m_parameter_expressions,
                                                                   dst->m_result_expressions);
    dst->m_is_dynamic = src->m_is_dynamic;
}

LinearIR::container LinearIRBuilder::clone_range(LinearIR::container::const_iterator begin, LinearIR::container::const_iterator end,
                                                 ExpressionMap& expression_map) const {
    OPENVINO_ASSERT(expression_map.empty(), " LinearIRBuilder::clone_range expects empty expression_map as an input");
    LinearIR::container result;
    NodeVector original_nodes;
    for (auto it = begin; it != end; it++) {
        original_nodes.push_back((*it)->get_node());
    }

    // node_map and expr_map map original node pointer (expression) to a new pointer (expression)
    NodeMap node_map;
    clone_nodes(original_nodes, node_map);

    for (auto it = begin; it != end; it++) {
        const auto& expr = *it;
        const auto& new_expr = expr->clone_with_new_inputs(expression_map, node_map[expr->get_node().get()]);
        result.push_back(new_expr);
        expression_map[expr.get()] = new_expr;
    }

    for (auto result_it = result.cbegin(), original_it = begin; result_it != result.cend(); ++result_it, ++original_it) {
        const auto& result_expr = *result_it;
        const auto& original_expr = *original_it;
        // Checking that the cloning was successful: the cloned part of LinearIR is identical to the original one
        OPENVINO_ASSERT(result_expr->get_node()->get_type_info() == original_expr->get_node()->get_type_info() &&
                        result_expr->get_input_count() == original_expr->get_input_count() &&
                        result_expr->get_output_count() == original_expr->get_output_count(),
                        "Expressions after copying aren't matched!");
        // Copy tensor shapes as shared pointer if needed
        if (!m_config.deep_copy_of_shapes) {
            for (size_t i = 0; i < original_expr->get_input_count(); ++i)
                result_expr->get_input_port_descriptor(i)->m_tensor_shape = original_expr->get_input_port_descriptor(i)->m_tensor_shape;
            for (size_t i = 0; i < original_expr->get_output_count(); ++i)
                result_expr->get_output_port_descriptor(i)->m_tensor_shape = original_expr->get_output_port_descriptor(i)->m_tensor_shape;
        }
    }

    return result;
}


}// namespace lowered
}// namespace snippets
}// namespace ov
