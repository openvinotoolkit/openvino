// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A file contains tokenization related utilities.
 * @file tokenization_utils.hpp
 */
#pragma once

#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/tokenization.hpp"

namespace ov::snippets::utils {
/**
 * @brief Tokenizes a node into Subgraph. 2 options are possible (depending on config's values and internal logic)L
 *        1. The node is wrapped in a trivial Subgraph which contains only this node
 *        2. The node is fused in parent's Subgraphs
 * @param node node which should be tokenized
 * @param config tokenization config which regulates
 * @return whether the node was tokenized or not
 */
bool tokenize_node(const std::shared_ptr<ov::Node>& node, const ov::snippets::pass::TokenizationConfig& config);
/**
 * @brief Tokenizes a list of nodes into Subgraph with the following rules:
 *        1. The user is responsible for valid count of parameters, results and hidden virtual ports (constants)
 *        2. The list of nodes cannot contain Subgraph ops
 * @param ordered_ops node list which should be tokenized
 * @param are_shared_internal_params_allowed if true, allows sharing internal parameters.
 * Note: Snippets support only internal parameters which are used by all the consumers as is.
 * This means that e.g. if the shared parameter is used by 2 MatMuls on B input,
 * both matmuls must have equal transpose_b parameter.
 * This is a user responsibility to ensure that the shared internal parameters can be used.
 * @return tokenized subgraph
 */
std::shared_ptr<ov::snippets::op::Subgraph> tokenize_ordered_nodes(const ov::NodeVector& ordered_ops,
                                                                   bool are_shared_internal_params_allowed = false);

/**
 * @brief Calculates the potential number of body parameters that would be required for a given operation.
 * Body parameters are created for Snippets node in 2 cases:
 *   1. The input is not a Constant node
 *   2. The input is a Constant node but it is not scalar
 * @note This function assumes that 0'th input of the operation is already counted, so it is ignored here
 * @param op The operation node to analyze
 * @return The estimated number of body parameters needed for this operation
 */
size_t get_potential_body_params(const std::shared_ptr<ov::Node>& op);
}  // namespace ov::snippets::utils
