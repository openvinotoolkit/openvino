// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

#include "snippets/pass/mha_tokenization.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "snippets/op/subgraph.hpp"

namespace ngraph {
namespace snippets {
namespace pass {

/*
 NotSet - default value returned by GetSnippetsNodeType(...) if the node wasn't marked
 SkippedByPlugin - indicate that snippets can't include this node in subgraph. Can be set by Plugin via SetSnippetsNodeType(...).
 */
enum class SnippetsNodeType : int64_t {NotSet, SkippedByPlugin};
/*
 NotSet - default value returned if the subgraph wasn't marked and snippets can include nodes in this subgraph
 Completed - indicate that snippets can't include any nodes in this subgraph.
             It's used in separate tokenization pass, for example, tokenization by matcher (MHA Tokenization).
 */
enum class SnippetsSubgraphType : int64_t {NotSet, Completed};
void SetSnippetsNodeType(const std::shared_ptr<Node>&, SnippetsNodeType);
void SetSnippetsSubgraphType(const std::shared_ptr<op::Subgraph>&, SnippetsSubgraphType);
SnippetsNodeType GetSnippetsNodeType(const std::shared_ptr<const Node>&);
SnippetsSubgraphType GetSnippetsSubgraphType(const std::shared_ptr<const op::Subgraph>&);
void SetTopologicalOrder(const std::shared_ptr<Node>&, int64_t);
int64_t GetTopologicalOrder(const std::shared_ptr<const Node>&);

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
 * @interface SnippetsTokenization
 * @brief  Splits model to supported subgraphs
 *         1. Enumerate nodes by topological order
 *         2. MHA tokenization
 *         3. Common tokenization
 *         4. Some common transformations for Subgraphs. For example, FakeQuantize decomposition
 * @ingroup snippets
 */
class SnippetsTokenization : public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("SnippetsTokenization", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};


}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
