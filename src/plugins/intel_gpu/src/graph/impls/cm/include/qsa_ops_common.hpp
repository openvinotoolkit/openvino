// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Shared helpers for the split QSA Indexer/SparsePA/SparseSDPA kernels.
//
// Paged addressing (page-table gather) reuses qsa::kv_cache_offset / raw_k_offset /
// summary_offset from qsa_common.hpp verbatim (those helpers already take the physical
// page / page-local offset as plain parameters, so they carry no dependency on the old
// fused op's macros). Dense addressing is new: direct `(batch, logical_position)`
// strides, no page table, no identity/degenerate paging -- the runtime stride is an
// explicit kernel scalar, never a JIT-baked page-capacity constant, so a dense call
// whose length is not a multiple of `compress_ratio` compiles and runs unchanged.

#ifndef ATTR
#    define ATTR [[type("svmptr_t")]]
#endif

#ifndef QSA_OPS_COMMON_HPP
#    define QSA_OPS_COMMON_HPP

#include "qsa_common.hpp"

// 0 = paged (shared main-KV page table), 1 = dense (direct batch/stride addressing).
#ifndef QSA_ADDRESS_MODE
#    define QSA_ADDRESS_MODE 0
#endif

namespace qsa_ops {

/// Number of QSA blocks fully visible to a query at absolute position `abs_pos`.
CM_INLINE uint num_complete_blocks(uint abs_pos) {
    return (abs_pos + 1) / (uint)QSA_COMPRESS_RATIO;
}

/// Resolves a flattened paged token index to its sequence and absolute logical position.
CM_INLINE void resolve_token_paged(const int32_t* subsequence_begins ATTR, const int32_t* past_lens ATTR, uint num_seqs, uint token_idx, uint& seq,
                                   uint& abs_pos) {
    seq = 0;
    for (uint i = 0; i < num_seqs; i++) {
        if (token_idx >= (uint)subsequence_begins[i] && token_idx < (uint)subsequence_begins[i + 1]) {
            seq = i;
            break;
        }
    }
    abs_pos = (uint)past_lens[seq] + token_idx - (uint)subsequence_begins[seq];
}

/// Resolves a flattened dense token index `t = b * seq_len + l` to its batch index and
/// absolute logical position. No subsequence table is synthesized: the dense batch axis
/// IS the sequence axis, and every batch element shares the same query length `seq_len`.
CM_INLINE void resolve_token_dense(const int32_t* past_lens ATTR, uint seq_len, uint token_idx, uint& batch, uint& abs_pos) {
    batch = token_idx / seq_len;
    abs_pos = (uint)past_lens[batch] + (token_idx - batch * seq_len);
}

/// Dense element offset of logical position `pos` of batch `b` in a `[B, stride, Di]`
/// buffer (`stride` is the raw-key capacity `S_cap` or the summary capacity
/// `floor(S_cap / r)`, a runtime value -- never a compile-time page-size constant).
CM_INLINE uint offset_dense(uint b, uint pos, uint stride, uint di) {
    return (b * stride + pos) * di;
}

/// Dense element offset into a `[B, Hkv, S, head_size]` main K/V buffer.
CM_INLINE uint kv_offset_dense(uint b, uint kv_head, uint pos, uint hkv, uint s, uint head_size) {
    return (((b * hkv) + kv_head) * s + pos) * head_size;
}

// --------------------------------------------------------------------------
// SIMD64 helpers for the cooperative work-group top-k finalizer (qsa_indexer_topk.cm).
// Lifted verbatim from qsa_topk_fast.hpp, which cannot be included directly here: it
// pulls in qsa_score_common.hpp, whose QSA_SUMMARIES_PER_PAGE macro is unconditionally
// defined in terms of QSA_PA_BLOCK_SIZE -- a constant the split kernels never define
// (dense mode has no page/page-size concept at all; paged mode uses QSA_PAGE_SIZE).
// --------------------------------------------------------------------------

/// Loads 64 consecutive score entries as monotone-ordered uint keys, out-of-range lanes
/// (i + lane >= n) read as the all-zero-score key so they always sort last.
CM_INLINE vector<uint, 64> fast_keys(const float* scores ATTR, uint i, uint n) {
    vector<uint, 64> lane;
    cmtl::cm_vector_assign(lane.select_all(), 0, 1);
    vector<float, 64> f;
    if (i + 64 <= n)
        f = cm_ptr_load<float, 64>(scores, i * 4);
    else
        f = cm_ptr_load<float>(scores, (lane + i) * 4, lane + i < n);
    vector<uint, 64> bits = f.format<uint>();
    vector<uint, 64> mask = 0x80000000u;
    mask.merge(0xFFFFFFFFu, (bits & 0x80000000u) != 0);
    return bits ^ mask;
}

/// Inclusive prefix sum over 64 lanes (Hillis-Steele, log2(64) = 6 steps).
CM_INLINE vector<uint, 64> fast_scan(vector<uint, 64> x) {
    vector<uint, 64> y = x;
    y.select<63, 1>(1) = x.select<63, 1>(1) + x.select<63, 1>(0);
    x = y;
    y.select<62, 1>(2) = x.select<62, 1>(2) + x.select<62, 1>(0);
    x = y;
    y.select<60, 1>(4) = x.select<60, 1>(4) + x.select<60, 1>(0);
    x = y;
    y.select<56, 1>(8) = x.select<56, 1>(8) + x.select<56, 1>(0);
    x = y;
    y.select<48, 1>(16) = x.select<48, 1>(16) + x.select<48, 1>(0);
    x = y;
    y.select<32, 1>(32) = x.select<32, 1>(32) + x.select<32, 1>(0);
    return y;
}

}  // namespace qsa_ops

#endif  // QSA_OPS_COMMON_HPP
