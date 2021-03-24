// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>


namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface StartSubgraph
 * @brief Matches multiple output loyout-oblivious operations to start a new subgraph
 * @ingroup snippets
 */
class TRANSFORMATIONS_API StartSubgraph: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    explicit StartSubgraph(bool tokenize_by_node = false);
};

/**
 * @interface AttachToSubgraph
 * @brief Matches loyout-oblivious operations with subgraph operation as an input to attech this node into it
 * @ingroup snippets
 */
class TRANSFORMATIONS_API AttachToSubgraph: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    explicit AttachToSubgraph(bool tokenize_by_node = false);
};

/**
 * @interface TokenizeSnippets
 * @brief Splits function to subgraphs if possible using rules above
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
 * Input subgraph is prefented from visiting twice if more than one output of it consumed by currently considered node
 * New subgraph is introduced, if there is a loop introduced
 * New subgraph is introduced, if number of inputs and outputs exceeds 7 due to scheduling limitation
 * New subgraph is introduced, if multiple outputs of merged nodes are not broadcastable to each other (equality of all outputs is too much on the other hand)
 * Scalar constants are placed as is into subgraph due to optimization purpose
 * @ingroup snippets
 */
class TRANSFORMATIONS_API TokenizeSnippets: public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    TokenizeSnippets(bool tokenize_by_node = false) {
        add_matcher<ngraph::snippets::pass::StartSubgraph>(tokenize_by_node);
        add_matcher<ngraph::snippets::pass::AttachToSubgraph>(tokenize_by_node);
    }
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
