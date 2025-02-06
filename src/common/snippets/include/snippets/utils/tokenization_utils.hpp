// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A file contains tokenization related utilities.
 * @file tokenization_utils.hpp
 */
#pragma once

#include "snippets/op/subgraph.hpp"
#include "snippets/pass/tokenization.hpp"

namespace ov {
namespace snippets {
namespace utils {

void fill_empty_output_names(const Output<Node>& target_output_node, const Output<Node>& replacement_output_node);

// todo: add description
std::shared_ptr<op::Subgraph> wrap_nodes_as_subgraph(const NodeVector& ordered_ops);

// Non-scalar Constants are tokenized as Parameters inside Subgraph body but some operations with constant inputs
// should have explicit Constants even if they're non-scalar (Reshape, Transpose, Broadcast)
// This check returns True if Constant op which is input of this op should be inside Subgraph body
bool constant_input_should_be_inside_body(const std::shared_ptr<ov::Node>& node);

/**
 * @brief Tokenizes a node into Subgraph. 2 options are possible (depending on config's values and internal logic)L
 *        1. The node is wrapped in a trivial Subgraph which contains only this node
 *        2. The node is fused in parent's Subgraphs
 * @param node node which should be tokenized
 * @param config tokenization config which regulates 
 * @return whether the node was tokenized or not
 */
bool tokenize_node(const std::shared_ptr<ov::Node>& node, const ov::snippets::pass::SnippetsTokenization::Config& config);

static inline auto create_body(const std::string& name, const ov::ResultVector& results, const ov::ParameterVector& parameters) ->
std::shared_ptr<ov::Model> {
    auto body = std::make_shared<ov::Model>(results, parameters, name);
    return body;
}

static inline auto build_subgraph(const std::shared_ptr<ov::Node>& node, const ov::OutputVector& inputs,
                                  const std::shared_ptr<ov::Model>& body, const std::string& name = "")
-> std::shared_ptr<op::Subgraph>{
    auto subgraph = std::make_shared<op::Subgraph>(inputs, body);
    copy_runtime_info(node, subgraph);
    subgraph->set_friendly_name(name.empty() ? node->get_friendly_name() : name);
    return subgraph;
}

// Need to update tensor name manually, since intel_cpu::Graph::Replicate() looks at input.get_shape().get_name();
// If subgraph->get_output_size() == 1, then the name will be restored correctly from the node name
auto inline update_out_tensor_name(const std::shared_ptr<ov::snippets::op::Subgraph>& subgraph) -> void {
    bool not_set = true;
    for (unsigned int i = 0; i < subgraph->get_output_size() && not_set; i++) {
        for (const auto& in : subgraph->get_output_target_inputs(i)) {
            if (ov::is_type<ov::op::v0::Result>(in.get_node())) {
                const auto& body_result = subgraph->body_ptr()->get_output_op(i);
                const auto& body_result_input = body_result->get_input_source_output(0);
                utils::fill_empty_output_names(subgraph->output(i), body_result_input);
                not_set = false;
                break;
            }
        }
    }
}

} // namespace utils
} // namespace snippets
} // namespace ov