// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>


namespace ngraph {
namespace snippets {
namespace pass {
/*
 NotSet - default value returned by GetSnippetsNodeType(...) if the node wasn't marked
 SkippedByPlugin - indicate that snippets can't include this node in subgraph. Can be set by Plugin via SetSnippetsNodeType(...).
 */
enum class SnippetsNodeType : int64_t {NotSet, SkippedByPlugin};
void SetSnippetsNodeType(const std::shared_ptr<Node>&, SnippetsNodeType);
SnippetsNodeType GetSnippetsNodeType(const std::shared_ptr<const Node>&);
void SetTopologicalOrder(const std::shared_ptr<Node>&, int64_t);
int64_t GetTopologicalOrder(const std::shared_ptr<const Node>&);
bool AppropriateForSubgraph(const std::shared_ptr<const Node>&);

/**
 * @interface EnumerateNodes
 * @brief  Snippets rely on topological order to avoid creating cyclic dependencies. This transformation sets the order.
 * @ingroup snippets
 */
class EnumerateNodes : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("EnumerateNodes", "0");
    EnumerateNodes() : ModelPass() {}
    bool run_on_model(const std::shared_ptr<ov::Model>&) override;
};

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
class TokenizeSnippets: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("TokenizeSnippets", "0");
    explicit TokenizeSnippets();
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
