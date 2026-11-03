// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qsa_indexer_jit_executor.hpp"

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <utility>

#include "cpu_parallel.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/qsa_indexer_config.hpp"
#include "nodes/kernels/sparse_attn/qsa_brgemm_scorer.hpp"
#include "nodes/kernels/sparse_attn/qsa_indexer_kernel.hpp"
#include "nodes/kernels/sparse_attn/qsa_jit_kernels.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "transformations/cpu_opset/x64/op/qsa_indexer.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

namespace {

using In = QSAIndexerNode::InputIdx;

// The static head counts: the batch and the window are dynamic, the heads are not.
bool head_counts(const MemoryDescArgs& descs, size_t& q_heads, size_t& k_heads) {
    q_heads = descs.at(qsa_src_arg(In::Q_IDX))->getShape().getDims()[2];
    k_heads = descs.at(qsa_src_arg(In::K_IDX))->getShape().getDims()[2];
    return q_heads != Shape::UNDEFINED_DIM && k_heads != Shape::UNDEFINED_DIM;
}

}  // namespace

// Phase 2 of the JIT executor: brgemm where the scorer exists, the token scorer on the JIT scoring kernel otherwise.
template <typename T>
void run_scorer(const QSAIndexerJitExecutor& backend,
                const node::qsa::Ports& p,
                const node::qsa::Config& c,
                const node::qsa::IndexerScratch<T>& scratch,
                const CpuParallel& parallel) {
    if (node::qsa::run_brgemm_scorer<T>(p, c, scratch, parallel)) {
        return;
    }
    node::qsa::run_token_scorer<T>(backend, p, c, scratch, parallel);
}

bool QSAIndexerJitExecutor::supports(const QSAIndexerConfig& config) {
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
        return false;
    }
    const auto precision = config.descs.at(qsa_src_arg(In::Q_IDX))->getPrecision();
    return any_of(precision, ov::element::f32, ov::element::f16, ov::element::bf16);
}

QSAIndexerJitExecutor::QSAIndexerJitExecutor(const QSAIndexerAttrs& attrs,
                                             const MemoryArgs& memory,
                                             ExecutorContext::CPtr context)
    : m_attrs(attrs),
      m_context(std::move(context)) {
    const auto& c = m_attrs.config;
    const auto cache = m_context->getRuntimeCache();
    m_rms_rope_kernel =
        node::qsa::JitRmsRope::get(cache, node::qsa::RmsRopeKey{c.indexer_head_dim, c.indexer_rotary_dim, c.eps});
    OPENVINO_ASSERT(m_rms_rope_kernel, "QSAIndexer: no JIT RMSNorm / RoPE kernel for this CPU");

    MemoryDescArgs descs;
    for (const auto& [arg, mem] : memory) {
        descs[arg] = mem->getDescPtr();
    }
    size_t q_heads = 0;
    size_t k_heads = 0;
    const auto precision = descs.at(qsa_src_arg(In::Q_IDX))->getPrecision();
    if (head_counts(descs, q_heads, k_heads)) {
        if (k_heads != 0 && q_heads % k_heads == 0) {
            m_score_kernel =
                node::qsa::JitQsaScore::get(cache,
                                            node::qsa::QsaScoreKey{precision, c.indexer_head_dim, q_heads / k_heads});
        }
        const node::qsa::BrgemmBlockScorer::Config config{
            q_heads,
            k_heads,
            c.h_sel,
            c.indexer_head_dim,
            c.indexer_rotary_dim,
            c.block_topk,
            static_cast<size_t>(m_context->getCpuParallel()->get_num_worker_threads()),
            c.eps,
            QSAIndexerNode::effective_scale(c),
            precision,
            c.order_by_score};
        m_brgemm = node::qsa::BrgemmBlockScorer::get(cache, config);
    }
    if (m_brgemm) {
        m_impl = node::qsa::score_impl_type(precision);
    } else {
        m_impl = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ? impl_desc_type::jit_avx512
                                                                                  : impl_desc_type::jit_avx2;
    }
}

void QSAIndexerJitExecutor::execute(const MemoryArgs& memory) {
    node::qsa::run_indexer(*this, memory, m_attrs.config, m_context);
}

}  // namespace ov::intel_cpu
