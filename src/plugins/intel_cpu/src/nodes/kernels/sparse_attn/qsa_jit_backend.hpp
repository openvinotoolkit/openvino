// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>

#include "nodes/kernels/sparse_attn/qsa_backends.hpp"
#include "nodes/kernels/sparse_attn/qsa_brgemm_scorer.hpp"
#include "nodes/kernels/sparse_attn/qsa_jit_kernels.hpp"

namespace ov::intel_cpu::node::qsa {

// The x64 backend of the node (the `jit` executor, see ScalarBackend for what a backend provides): the RMSNorm + RoPE
// on the JIT kernels of the RMSNorm and RoPE nodes, the scoring of the blocks on brgemm where the CPU, the precision
// and the head counts allow it and on a JIT kernel otherwise. Its kernels come from the params cache of the graph
// context when the executor is created, never while the model runs.
struct JitBackend {
    std::shared_ptr<const JitRmsRope> rms_rope_kernel;
    std::shared_ptr<const JitQsaScore> score_kernel;  // null: the scoring runs in plain C++
    std::shared_ptr<const BrgemmBlockScorer> brgemm;  // null: no brgemm scoring

    void rms_rope(const float* src,
                  float* dst,
                  const float* weight,
                  size_t /*dim*/,
                  size_t /*rotary_dim*/,
                  float /*eps*/,
                  const float* cos,
                  const float* sin) const {
        (*rms_rope_kernel)(src, dst, weight, cos, sin);
    }
    [[nodiscard]] const BrgemmBlockScorer* scorer() const {
        return brgemm.get();
    }
    // scores[i] += the scores of n summary rows of an index key head (see score_run_scalar).
    template <typename T>
    void score_run(const float* q,
                   const T* rows,
                   size_t n,
                   size_t G,
                   size_t D,
                   float* scores,
                   float* stage,
                   float* row_buf) const {
        if (score_kernel) {
            (*score_kernel)(q, rows, n, scores, stage);
        } else {
            score_run_scalar<T>(q, rows, n, G, D, scores, row_buf);
        }
    }
};

}  // namespace ov::intel_cpu::node::qsa
