// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

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
 *         Naming policy:
 *           - During tokenization new Subgraph op takes the name of the last tokenized op.
 *             It's needed to save output names of model in cases when tokenized op was before model Result.
 *           - If some transformation (for example, SplitDimensionM) insert new op after Subgraph,
 *             the op should be called as this Subgraph to save output name. The Subgraph name is updated using suffix "_original".
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
        Config(size_t concurrency, size_t data_ptr_gpr_count, bool split_m_dimension, bool enable_transpose_on_output,
               bool dyn_mha_token, std::set<size_t> mha_transpose_ranks)
            : m_concurrency(concurrency), m_data_ptr_gpr_count(data_ptr_gpr_count), m_split_m_dimension(split_m_dimension),
              m_mha_token_enable_transpose_on_output(enable_transpose_on_output), m_is_dynamic_mha_token_enabled(dyn_mha_token),
              m_mha_supported_transpose_ranks(std::move(mha_transpose_ranks)) {
            OPENVINO_ASSERT(concurrency > 0, "Concurrency should be greater than 0");
            OPENVINO_ASSERT(data_ptr_gpr_count > 0, "data_ptr_gpr_count should be greater than 0");
        }

        void set_concurrency(size_t concur) {
            m_concurrency = concur;
        }

        size_t get_concurrency() const {
            return m_concurrency;
        }

        size_t get_data_ptr_gpr_count() const {
            return m_data_ptr_gpr_count;
        }

        bool get_split_m_dimension() const {
            return m_split_m_dimension;
        }

        bool get_mha_token_enable_transpose_on_output() const {
            return m_mha_token_enable_transpose_on_output;
        }

        bool is_dynamic_mha_token_enabled() const {
            return m_is_dynamic_mha_token_enabled;
        }

        std::set<size_t> get_mha_supported_transpose_ranks() const {
            return m_mha_supported_transpose_ranks;
        }

    private:
        size_t m_concurrency = 0;
        // The number of gpr that can be used as data pointers for data nodes (Parameter (and non-Scalar Constants),
        // Result, Buffers with the same ID)
        size_t m_data_ptr_gpr_count = 0;
        // True if "SplitDimensionM" optimization is enabled. Otherwise, it's disabled.
        bool m_split_m_dimension = true;
        // False if Transpose on output isn't tokenized in MHA Tokenization.
        // Otherwise, it may be fused into Subgraph if possible
        // TODO [111813]: Remove please when the ticket 111813 is implemented
        bool m_mha_token_enable_transpose_on_output = true;
        // If True, MHA pattern with dynamic nodes will be tokenized
        // Otherwise dynamic MHA won't be tokenized
        // Currently, the flag can be set to `True` only for testing purposes.
        bool m_is_dynamic_mha_token_enabled = true;
        // Set of supported Transpose shape ranks for tokenization in MHATokenization pass.
        // Note that in general Snippets support Transpose of any ranks.
        // But at the moment Transpose is used only in MHA pattern where 3D and 4D tensors are supported.
        std::set<size_t> m_mha_supported_transpose_ranks = { 3, 4 };
    };

    OPENVINO_RTTI("SnippetsTokenization", "0");
    SnippetsTokenization(const Config& config) : m_config(config) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    Config m_config;
};


}  // namespace pass
}  // namespace snippets
}  // namespace ov
