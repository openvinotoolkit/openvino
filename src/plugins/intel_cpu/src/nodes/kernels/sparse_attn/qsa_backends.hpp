// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include "nodes/kernels/sparse_attn/qsa_indexer_common.hpp"

#ifdef OPENVINO_ARCH_X86_64
#    include "nodes/kernels/sparse_attn/qsa_brgemm_scorer.hpp"
#else
namespace ov::intel_cpu::node::qsa {
class BrgemmBlockScorer;
}  // namespace ov::intel_cpu::node::qsa
#endif

namespace ov::intel_cpu::node::qsa {

// scores[i] += sum over the G query heads h of max(0, q[h] . rows[i]), for the n summary rows ([n, D] of the type T) of
// one index key head and its G query heads q ([G, D] f32), the plain C++ way. A row is converted to f32 once, whatever
// the number of query heads that share it (`row_buf`, D floats).
template <typename T>
void score_run_scalar(const float* q, const T* rows, size_t n, size_t G, size_t D, float* scores, float* row_buf) {
    for (size_t i = 0; i < n; i++) {
        const T* row = rows + i * D;
        const float* row_f32 = row_buf;
        if constexpr (std::is_same_v<T, float>) {
            row_f32 = row;
        } else {
            row_to_f32<T>(row, row_buf, D);
        }
        float sum = 0.0F;
        for (size_t h = 0; h < G; h++) {
            float dot = 0.0F;
            for (size_t d = 0; d < D; d++) {
                dot += q[h * D + d] * row_f32[d];
            }
            sum += std::max(0.0F, dot);
        }
        scores[i] += sum;
    }
}

// What differs between the implementations of the nodes (QSAIndexer, PagedQSAIndexer). A backend provides
//   rms_rope(src, dst, weight, dim, rotary_dim, eps, cos, sin)   the RMSNorm + partial RoPE of an f32 row (`dst` may be
//                                                                  `src`)
//   scorer()                                                     the brgemm scorer of the call, null if it has none
//   score_run<T>(q, rows, n, G, D, scores, stage, row_buf)       scores[i] += the scores of n summary rows of an
//                                                                  index key head, see score_run_scalar
// ScalarBackend is the plain C++ one (the `ref` executors); the JIT one is JitBackend in qsa_jit_backend.hpp.
struct ScalarBackend {
    static void rms_rope(const float* src,
                         float* dst,
                         const float* weight,
                         size_t dim,
                         size_t rotary_dim,
                         float eps,
                         const float* cos,
                         const float* sin) {
        rmsnorm_partial_rope(src, dst, weight, dim, rotary_dim, eps, cos, sin);
    }
    [[nodiscard]] static const BrgemmBlockScorer* scorer() {
        return nullptr;
    }
    template <typename T>
    static void score_run(const float* q,
                          const T* rows,
                          size_t n,
                          size_t G,
                          size_t D,
                          float* scores,
                          float* /*stage*/,
                          float* row_buf) {
        score_run_scalar<T>(q, rows, n, G, D, scores, row_buf);
    }
};

}  // namespace ov::intel_cpu::node::qsa
