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
bool tokenize_node(const std::shared_ptr<ov::Node>& node,
                   const ov::snippets::pass::SnippetsTokenization::Config& config);
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
}  // namespace ov::snippets::utils
