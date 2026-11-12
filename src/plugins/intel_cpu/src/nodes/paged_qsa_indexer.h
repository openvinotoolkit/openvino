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
#include "nodes/kernels/sparse_attn/qsa_executor.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/node.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"

namespace ov::intel_cpu::node {

/// @brief CPU implementation of ov::op::internal::PagedQSAIndexer (paged-cache
/// complete-block plan producer for Qwen Sparse Attention).
///
/// Same per-token math as the dense QSAIndexer node (query RMSNorm + partial RoPE, raw-K
/// append, completed-block summary pooling, per-selection-head scoring + top-k), but every
/// cache access (index_key_cache, index_block_rope_cache, block_index_keys_cache) is addressed
/// through the shared PagedAttention page table (block_indices/block_indices_begins/
/// subsequence_begins/past_lens) instead of a flat dense [B, ...] layout. The page table is validated on every call
/// (check_paged_page_table); the caches are written in place and keep the inference precision.
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
    // The computation is the cross-compiled executor (AVX-512 / AVX2 / plain C++ by the CPU); the scoring of the blocks
    // runs on the brgemm scorer instead where the CPU, the precisions and the head counts allow it.
    ov::op::internal::PagedQSAIndexer::Config m_config;
    std::shared_ptr<ov::Extensions::Cpu::qsa::PagedExecutor> m_executor;
    std::shared_ptr<ov::Extensions::Cpu::qsa::PagedScorer> m_scorer;  // null: the executor scores the blocks itself
    impl_desc_type m_impl = impl_desc_type::ref;
};

}  // namespace ov::intel_cpu::node
