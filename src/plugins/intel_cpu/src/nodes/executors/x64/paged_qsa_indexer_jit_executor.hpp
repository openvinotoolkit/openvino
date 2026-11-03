// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "nodes/executors/executor.hpp"
#include "nodes/executors/paged_qsa_indexer_config.hpp"
#include "nodes/kernels/sparse_attn/qsa_indexer_common.hpp"
#include "nodes/kernels/sparse_attn/qsa_jit_kernels.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {

// The x64 implementation : the RMSNorm + RoPE and, without brgemm, the scoring on JIT kernels; the
// scoring on brgemm (the AMX / AVX-512 matrix products of qsa_brgemm_scorer.hpp) where the CPU, the precisions and the
// head counts allow it.
class PagedQSAIndexerJitExecutor : public Executor {
public:
    static bool supports(const PagedQSAIndexerConfig& config);

    PagedQSAIndexerJitExecutor(const PagedQSAIndexerAttrs& attrs,
                               const MemoryArgs& memory,
                               ExecutorContext::CPtr context);

    bool update(const MemoryArgs& /*memory*/) override {
        return true;
    }
    void execute(const MemoryArgs& memory) override;
    [[nodiscard]] impl_desc_type implType() const override {
        return m_impl;
    }

    // What the kernels of qsa_indexer_kernel.hpp need of a backend (see ScalarBackend); the kernels come from the
    // params cache when the executor is created, never while the model runs.
    void rms_rope(const float* src,
                  float* dst,
                  const float* weight,
                  size_t /*dim*/,
                  size_t /*rotary_dim*/,
                  float /*eps*/,
                  const float* cos,
                  const float* sin) const {
        (*m_rms_rope_kernel)(src, dst, weight, cos, sin);
    }
    [[nodiscard]] const node::qsa::BrgemmBlockScorer* scorer() const {
        return m_brgemm.get();
    }
    template <typename T>
    void score_run(const float* q,
                   const T* rows,
                   size_t n,
                   size_t G,
                   size_t D,
                   float* scores,
                   float* stage,
                   float* row_buf) const {
        if (m_score_kernel) {
            (*m_score_kernel)(q, rows, n, scores, stage);
        } else {
            node::qsa::score_run_scalar<T>(q, rows, n, G, D, scores, row_buf);
        }
    }

private:
    PagedQSAIndexerAttrs m_attrs;
    ExecutorContext::CPtr m_context;
    std::shared_ptr<const node::qsa::JitRmsRope> m_rms_rope_kernel;
    std::shared_ptr<const node::qsa::JitQsaScore> m_score_kernel;  // null: the scoring runs in plain C++
    std::shared_ptr<const node::qsa::BrgemmBlockScorer> m_brgemm;  // null: no brgemm scoring
    impl_desc_type m_impl = impl_desc_type::ref;
};

}  // namespace ov::intel_cpu
