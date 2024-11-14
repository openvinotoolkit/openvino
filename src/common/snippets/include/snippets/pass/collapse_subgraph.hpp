// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "snippets/pass/tokenization.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface TokenizeSnippets
 * @brief Splits model to subgraphs if possible using rules above
 * This pass tokenizes topology graph into subgraphs.
 * Those subgraphs consists of unary or binary layout-oblivious (LO) opetations found in subset 1.
 * Non-layout-oblivious (NLO) operations operations (called also support in this context) are ignored and become a fullstop in tokenization routine
 * 1. if a considered LO operation doesn't have any unput subgraphs
 *    -> a new single-op subgraph is introduced
 * 1. if a considered LO operation is a binary or an unary operation with at least one subgraph as an input
 *    -> 1. all inputs from the conput subgraphs are collected together
 *       1. non-subgraph inputs are wrapped into parameters
 *       1. all input bodies are merged and
 *       1. this new operation is added to a body of input subgraph
 *       1. outputs are collected subgraph (outputs consumed by some other node & subgraph outputs consumed by the node to be merged)
 *       1. finally current node is replaced with the new subgraph. We cannot use replace_node because multiple nodes are replaced so
 *       make the replacement manually by redirecting ports
 * New subgraph is introduced, if there is a loop introduced
 * New subgraph is introduced, if number of inputs and outputs exceeds 7 due to scheduling limitation
 * New subgraph is introduced, if multiple outputs of merged nodes are not broadcastable to each other (equality of all outputs is too much on the other hand)
 * Scalar constants are placed as is into subgraph due to optimization purpose
 * @ingroup snippets
 */
class TokenizeSnippets: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TokenizeSnippets", "0");
    explicit TokenizeSnippets(const SnippetsTokenization::Config& config);

    static bool AppropriateForSubgraph(const std::shared_ptr<const Node>&);

    static const std::set<ov::element::Type>& get_supported_element_types();
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
