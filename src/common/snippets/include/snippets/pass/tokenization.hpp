// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"

#include "snippets/op/subgraph.hpp"

namespace ov {
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
class SnippetsTokenization : public ov::pass::ModelPass {
public:
    /**
     * @interface Config
     * @brief Allow to adjust tokenization passes
     * @ingroup snippets
     */
    struct Config {
        Config(size_t concurrency = 1, bool split_m_dimension = true, bool enable_transpose_on_output = true)
            : concurrency(concurrency), split_m_dimension(split_m_dimension),
              mha_token_enable_transpose_on_output(enable_transpose_on_output) {}

        size_t concurrency = 1;
        // True if "SplitDimensionM" optimization is enabled. Otherwise, it's disabled.
        bool split_m_dimension = true;
        // False if Transpose on output isn't tokenized in MHA Tokenization.
        // Otherwise, it may be fused into Subgraph if possible
        // TODO [111813]: Remove please when the ticket 111813 is implemented
        bool mha_token_enable_transpose_on_output = true;
    };

    OPENVINO_RTTI("SnippetsTokenization", "0");
    SnippetsTokenization(const Config& config) : m_config(config) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    Config m_config{};
};


}  // namespace pass
}  // namespace snippets
}  // namespace ov
