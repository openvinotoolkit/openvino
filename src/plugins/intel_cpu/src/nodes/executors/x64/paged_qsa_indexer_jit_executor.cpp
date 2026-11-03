// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_qsa_indexer_jit_executor.hpp"

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <utility>

#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/paged_qsa_indexer_config.hpp"
#include "nodes/kernels/sparse_attn/paged_qsa_indexer_kernel.hpp"
#include "nodes/kernels/sparse_attn/qsa_brgemm_scorer.hpp"
#include "nodes/kernels/sparse_attn/qsa_jit_kernels.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

namespace {

using In = ov::op::internal::PagedQSAIndexer::InputIdx;

// The static head counts: the query and key rows are [tokens, heads * dim], the token count is dynamic.
bool head_counts(const MemoryDescArgs& descs, size_t dim, size_t& q_heads, size_t& k_heads) {
    const auto q_width = descs.at(paged_qsa_src_arg(In::INDEX_QUERY))->getShape().getDims()[1];
    const auto k_width = descs.at(paged_qsa_src_arg(In::RAW_INDEX_KEYS))->getShape().getDims()[1];
    if (q_width == Shape::UNDEFINED_DIM || k_width == Shape::UNDEFINED_DIM || dim == 0) {
        return false;
    }
    q_heads = q_width / dim;
    k_heads = k_width / dim;
    return true;
}

}  // namespace

bool PagedQSAIndexerJitExecutor::supports(const PagedQSAIndexerConfig& config) {
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
        return false;
    }
    const auto precision = config.descs.at(paged_qsa_src_arg(In::BLOCK_INDEX_KEYS_CACHE))->getPrecision();
    return any_of(precision, ov::element::f32, ov::element::f16, ov::element::bf16);
}

PagedQSAIndexerJitExecutor::PagedQSAIndexerJitExecutor(const PagedQSAIndexerAttrs& attrs,
                                                       const MemoryArgs& memory,
                                                       ExecutorContext::CPtr context)
    : m_attrs(attrs),
      m_context(std::move(context)) {
    const auto& c = m_attrs.config;
    const auto cache = m_context->getRuntimeCache();
    m_rms_rope_kernel =
        node::qsa::JitRmsRope::get(cache, node::qsa::RmsRopeKey{c.indexer_head_dim, c.indexer_rotary_dim, c.epsilon});
    OPENVINO_ASSERT(m_rms_rope_kernel, "PagedQSAIndexer: no JIT RMSNorm / RoPE kernel for this CPU");

    MemoryDescArgs descs;
    for (const auto& [arg, mem] : memory) {
        descs[arg] = mem->getDescPtr();
    }
    const auto summary_precision = descs.at(paged_qsa_src_arg(In::BLOCK_INDEX_KEYS_CACHE))->getPrecision();
    const auto query_precision = descs.at(paged_qsa_src_arg(In::INDEX_QUERY))->getPrecision();
    size_t q_heads = 0;
    size_t k_heads = 0;
    if (head_counts(descs, c.indexer_head_dim, q_heads, k_heads) && k_heads != 0 && q_heads % k_heads == 0) {
        m_score_kernel = node::qsa::JitQsaScore::get(
            cache,
            node::qsa::QsaScoreKey{summary_precision, c.indexer_head_dim, q_heads / k_heads});
        // the query is rounded to the precision of the summary blocks for the brgemm products: only done when it is
        // that precision already
        if (query_precision == summary_precision) {
            const node::qsa::BrgemmBlockScorer::Config config{
                q_heads,
                k_heads,
                c.num_selection_heads,
                c.indexer_head_dim,
                c.indexer_rotary_dim,
                c.block_topk,
                static_cast<size_t>(m_context->getCpuParallel()->get_num_worker_threads()),
                c.epsilon,
                ov::op::internal::PagedQSAIndexer::effective_scale(c),
                summary_precision};
            m_brgemm = node::qsa::BrgemmBlockScorer::get(cache, config);
        }
    }
    if (m_brgemm) {
        m_impl = node::qsa::score_impl_type(summary_precision);
    } else {
        m_impl = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ? impl_desc_type::jit_avx512
                                                                                  : impl_desc_type::jit_avx2;
    }
}

void PagedQSAIndexerJitExecutor::execute(const MemoryArgs& memory) {
    node::qsa::paged::run_indexer(*this, memory, m_attrs.config, m_context);
}

}  // namespace ov::intel_cpu
