// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/pass/pass.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/common_optimizations.hpp"
#include "snippets/pass/mha_tokenization.hpp"
#include "snippets/pass/mlp_seq_tokenization.hpp"
#include "snippets/pass/tokenization_config.hpp"
#include "snippets/snippets_visibility.hpp"

namespace ov::snippets::pass {

/*
 NotSet - default value returned by GetSnippetsNodeType(...) if the node wasn't marked
 SkippedByPlugin - indicate that snippets can't include this node in subgraph. Can be set by Plugin via
 SetSnippetsNodeType(...).
 */
enum class SnippetsNodeType : uint8_t { NotSet, SkippedByPlugin };
/*
 NotSet - default value returned if the subgraph wasn't marked and snippets can include nodes in this subgraph
 Completed - indicate that snippets can't include any nodes in this subgraph.
             It's used in separate tokenization pass, for example, tokenization by matcher (MHA Tokenization).
 */
enum class SnippetsSubgraphType : uint8_t { NotSet, Completed };
SNIPPETS_API void SetSnippetsNodeType(const std::shared_ptr<Node>& node, SnippetsNodeType nodeType);
SNIPPETS_API void SetSnippetsSubgraphType(const std::shared_ptr<op::Subgraph>& node, SnippetsSubgraphType nodeType);
SNIPPETS_API SnippetsNodeType GetSnippetsNodeType(const std::shared_ptr<const Node>& node);
SNIPPETS_API SnippetsSubgraphType GetSnippetsSubgraphType(const std::shared_ptr<const op::Subgraph>& node);
SNIPPETS_API void SetTopologicalOrder(const std::shared_ptr<Node>& node, int64_t order);
SNIPPETS_API int64_t GetTopologicalOrder(const std::shared_ptr<const Node>& node);

/**
 * @interface EnumerateNodes
 * @brief  Snippets rely on topological order to avoid creating cyclic dependencies. This transformation sets the order.
 * @ingroup snippets
 */
class SNIPPETS_API EnumerateNodes : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("snippets::pass::EnumerateNodes");
    EnumerateNodes() : ModelPass() {}
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

/**
 * @interface SnippetsTokenization
 * @brief  Splits model to supported subgraphs
 *         1. Enumerate nodes by topological order
 *         2. MHA tokenization
 *         3. Common tokenization
 *         4. Some common transformations for Subgraphs. For example, FakeQuantize decomposition
 *         Naming policy:
 *           - During tokenization new Subgraph op takes the name of the last tokenized op.
 *             It's needed to save output names of model in cases when tokenized op was before model Result.
 *           - If some transformation (for example, SplitDimensionM) insert new op after Subgraph,
 *             the op should be called as this Subgraph to save output name. The Subgraph name is updated using suffix
 *             "_original".
 * @ingroup snippets
 */
class SNIPPETS_API SnippetsTokenization : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("snippets::pass::SnippetsTokenization");

    explicit SnippetsTokenization(TokenizationConfig config,
                                  CommonOptimizations::Config common_config,
                                  TokenizeMHASnippets::Config mha_config,
                                  TokenizeMLPSeqSnippets::Config mlp_seq_config)
        : m_tokenization_config(config),
          m_common_optimizations_config(common_config),
          m_mha_config(std::move(mha_config)),
          m_mlp_seq_config(std::move(mlp_seq_config)) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    TokenizationConfig m_tokenization_config;
    CommonOptimizations::Config m_common_optimizations_config;
    TokenizeMHASnippets::Config m_mha_config;
    TokenizeMLPSeqSnippets::Config m_mlp_seq_config;
};

}  // namespace ov::snippets::pass
