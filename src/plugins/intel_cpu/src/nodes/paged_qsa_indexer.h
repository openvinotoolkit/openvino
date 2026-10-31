// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"

namespace ov::intel_cpu::node {

namespace qsa {
class BrgemmBlockScorer;
}  // namespace qsa

/// @brief F32 reference implementation of ov::op::internal::PagedQSAIndexer (paged-cache
/// complete-block plan producer for Qwen Sparse Attention).
///
/// Same per-token math as the dense QSAIndexer node (query RMSNorm + partial RoPE, raw-K
/// append, completed-block summary pooling, per-selection-head scoring + top-k), but every
/// cache access (index_key_cache, index_block_rope_cache, block_index_keys_cache) is addressed
/// through the shared PagedAttention page table (block_indices/block_indices_begins/
/// subsequence_begins/past_lens) instead of a flat dense [B, ...] layout.
class PagedQSAIndexer : public Node {
public:
    PagedQSAIndexer(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    [[nodiscard]] bool created() const override {
        return getType() == Type::PagedQSAIndexer;
    }
    [[nodiscard]] bool needPrepareParams() const override {
        return false;
    }
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    void createPrimitive() override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    // `T`: the precision of the summary blocks, the one the block scoring on brgemm (see qsa_brgemm_scorer.hpp) works
    // in.
    template <typename T>
    void execute_typed();

    ov::op::internal::PagedQSAIndexer::Config m_config{};
    // The scorer of the block scoring on brgemm (see qsa_brgemm_scorer.hpp), taken from the params cache of the graph
    // context in createPrimitive(): the layers of the same configuration share it, and nothing is built while the model
    // runs. Null when this CPU or configuration has no brgemm scoring (also when the query and the summary blocks
    // differ in precision): the scalar scoring runs then.
    std::shared_ptr<const qsa::BrgemmBlockScorer> m_scorer;
};

}  // namespace ov::intel_cpu::node
