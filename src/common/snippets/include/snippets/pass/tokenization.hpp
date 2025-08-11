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
void SetSnippetsNodeType(const std::shared_ptr<Node>& node, SnippetsNodeType nodeType);
void SetSnippetsSubgraphType(const std::shared_ptr<op::Subgraph>& node, SnippetsSubgraphType nodeType);
SnippetsNodeType GetSnippetsNodeType(const std::shared_ptr<const Node>& node);
SnippetsSubgraphType GetSnippetsSubgraphType(const std::shared_ptr<const op::Subgraph>& node);
void SetTopologicalOrder(const std::shared_ptr<Node>& node, int64_t order);
int64_t GetTopologicalOrder(const std::shared_ptr<const Node>& node);

/**
 * @interface EnumerateNodes
 * @brief  Snippets rely on topological order to avoid creating cyclic dependencies. This transformation sets the order.
 * @ingroup snippets
 */
class EnumerateNodes : public ov::pass::ModelPass {
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
class SnippetsTokenization : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("snippets::pass::SnippetsTokenization");

    /**
     * @interface Config
     * @brief Allow to adjust tokenization passes
     * @ingroup snippets
     */
    struct Config {
        using CanBeFusedAsPostOpPred = std::function<bool(const std::shared_ptr<const ov::op::v0::MatMul>&,
                                                          const std::shared_ptr<const ov::Node>&)>;

        Config(size_t concurrency,
               size_t data_ptr_gpr_count,
               bool split_m_dimension,
               bool enable_transpose_on_output,
               bool dyn_mha_token,
               std::set<size_t> mha_transpose_ranks,
               CanBeFusedAsPostOpPred can_be_fused_as_postop = nullptr)
            : m_concurrency(concurrency),
              m_data_ptr_gpr_count(data_ptr_gpr_count),
              m_split_m_dimension(split_m_dimension),
              m_mha_token_enable_transpose_on_output(enable_transpose_on_output),
              m_is_dynamic_mha_token_enabled(dyn_mha_token),
              m_mha_supported_transpose_ranks(std::move(mha_transpose_ranks)),
              m_can_be_fused_as_postop(std::move(can_be_fused_as_postop)) {
            OPENVINO_ASSERT(concurrency > 0, "Concurrency should be greater than 0");
            OPENVINO_ASSERT(data_ptr_gpr_count > 0, "data_ptr_gpr_count should be greater than 0");
        }

        void set_concurrency(size_t concur) {
            m_concurrency = concur;
        }

        [[nodiscard]] size_t get_concurrency() const {
            return m_concurrency;
        }

        [[nodiscard]] size_t get_data_ptr_gpr_count() const {
            return m_data_ptr_gpr_count;
        }

        [[nodiscard]] bool get_split_m_dimension() const {
            return m_split_m_dimension;
        }

        [[nodiscard]] bool get_mha_token_enable_transpose_on_output() const {
            return m_mha_token_enable_transpose_on_output;
        }

        [[nodiscard]] bool is_dynamic_mha_token_enabled() const {
            return m_is_dynamic_mha_token_enabled;
        }

        [[nodiscard]] std::set<size_t> get_mha_supported_transpose_ranks() const {
            return m_mha_supported_transpose_ranks;
        }

        [[nodiscard]] const CanBeFusedAsPostOpPred& get_can_be_fused_as_postop() const {
            return m_can_be_fused_as_postop;
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
        std::set<size_t> m_mha_supported_transpose_ranks = {3, 4};
        // Predicate that checks if the node can be fused as MatMul post-op.
        // It is currently used only in TokenizeMLPSeqSnippets
        CanBeFusedAsPostOpPred m_can_be_fused_as_postop = nullptr;
    };

    explicit SnippetsTokenization(Config config) : m_config(std::move(config)) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    Config m_config;
};

}  // namespace ov::snippets::pass
