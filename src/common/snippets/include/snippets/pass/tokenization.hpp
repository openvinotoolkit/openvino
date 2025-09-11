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

        Config(size_t available_gprs_count,
               bool enable_transpose_on_output,
               bool dyn_mha_token,
               std::set<size_t> mha_transpose_ranks,
               CanBeFusedAsPostOpPred can_be_fused_as_postop = nullptr)
            : m_available_gprs_count(available_gprs_count),
              m_mha_token_enable_transpose_on_output(enable_transpose_on_output),
              m_is_dynamic_mha_token_enabled(dyn_mha_token),
              m_mha_supported_transpose_ranks(std::move(mha_transpose_ranks)),
              m_can_be_fused_as_postop(std::move(can_be_fused_as_postop)) {
            OPENVINO_ASSERT(available_gprs_count > 0, "available_gprs_count should be greater than 0");
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

        /**
         * @brief Checks if the available GPRs count is sufficient for the given requirements.
         * @param io_count Number of input/output,
         *        each of which requires GPR allocated throughout the life of the kernel.
         * @param expected_bufer_reg_groups Number of unique buffer register groups,
         *        each of which requires GPR allocated throughout the life of the kernel.
         * @param expected_maximal_loop_depth Each loop uses GPR for work amount storage.
         *        For the expressions covered with all `expected_maximal_loop_depth` loops,
         *        `expected_maximal_loop_depth` GPRS must be alive
         * @param is_dynamic Indicates whether the subgraph is dynamic.
         *        It affects the number of available GPRs:
         *        in static case, abi_param2 is used to pass precomputed offsets to the kernel.
         * @return true if the available GPRs are sufficient; false otherwise.
         */
        [[nodiscard]] bool is_gprs_count_sufficient(const size_t io_count,
                                                    const size_t expected_bufer_reg_groups,
                                                    const size_t expected_maximal_loop_depth,
                                                    bool is_dynamic = false) const {
            const auto available_gprs_count = is_dynamic ? m_available_gprs_count : m_available_gprs_count - 1;
            return (io_count + expected_bufer_reg_groups + expected_maximal_loop_depth) <= available_gprs_count;
        }

    private:
        // The number of gpr that can be used inside snippets kernel
        // (data pointers for Parameters/Results/Buffers, as well as loop work amounts)
        size_t m_available_gprs_count = 0;
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

    explicit SnippetsTokenization(Config config, ov::snippets::pass::CommonOptimizations::Config common_config) 
        : m_config(std::move(config)), m_common_config(std::move(common_config)) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    Config m_config;
    ov::snippets::pass::CommonOptimizations::Config m_common_config;
};

}  // namespace ov::snippets::pass
