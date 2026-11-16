// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Q2 top-k finalize, shared by the paged and dense indexers through
// QSA_ADDRESS_MODE (0 = paged, 1 = dense).
//
// A work-group of QSA_TOPK_WG subgroups jointly selects the top-k blocks of one token row: score descending, block
// id ascending as tie-break, written in ASCENDING block order. The k-th largest order-preserving uint key is found by
// an (NT + 1)-ary count search: each probe counts the keys >= NT evenly spaced thresholds in one pass over the row and
// one barrier (per-subgroup partials are double-buffered in SLM and reduced redundantly by every work-item), so 32 key
// bits take ceil(32 / log2(NT + 1)) probes. The keys of the row live in registers (QSA_TOPK_REG_KEYS per lane); a row
// that does not fit rescans global memory on every probe, with the same result. Subgroup s owns the contiguous block
// range [s*chunk, (s+1)*chunk) and lane l of it the contiguous run of chunk/SG keys starting at l * chunk/SG, so every
// lane writes its picks in ascending order and one subgroup scan of the per-lane counts places them. A row that selects every
// block never reads its scores.
//
// JIT: QSA_BLOCK_TOPK, QSA_COMPRESS_RATIO, QSA_ADDRESS_MODE, QSA_TOPK_WG (most subgroups), QSA_TOPK_REG_KEYS, QSA_TOPK_THRESHOLDS (<= 7),
//      SUBGROUP_SIZE
// Dispatch: gws = [rows * W * SG, 1, 1], lws = [W * SG, 1, 1] with W <= QSA_TOPK_WG subgroups (W = QSA_TOPK_WG if QSA_TOPK_FIXED_WG)
// block_scores holds only rows [row_begin, row_end) with row pitch `pitch`.

#include "qsa_common.cl"

#define QSA_NT QSA_TOPK_THRESHOLDS
#define QSA_KPL QSA_TOPK_REG_KEYS
#define QSA_GRP 8  // the register walk tests the run length once per QSA_GRP keys; QSA_KPL is a multiple of it

// Monotone float -> uint key.
inline uint FUNC(qsa_ordered_key)(float f) {
    const uint bits = as_uint(f);
    return bits ^ ((bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u);
}

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(qsa_indexer_topk)(const __global float* block_scores,            // [row_end - row_begin, pitch]
                         const __global int* past_lens,                 // paged: [num_seqs]; dense: [B]
#if QSA_ADDRESS_MODE == 0
                         const __global int* subsequence_begins,        // [num_seqs+1]
#endif
                         __global int* selected_blocks,                 // [tokens, block_topk]
                         __global int* selected_counts,                 // [tokens]
                         const uint num_tokens,
#if QSA_ADDRESS_MODE == 0
                         const uint num_seqs,
#else
                         const uint seq_len,
#endif
                         const uint pitch,
                         const uint row_begin,
                         const uint row_end) {
    const uint K = QSA_BLOCK_TOPK;
#ifdef QSA_TOPK_FIXED_WG
    // Compile-time work-group size (decode): the per-probe SLM reductions unroll.
    const uint WGN = QSA_TOPK_WG;
    const uint WGT = QSA_TOPK_WG * QSA_SG;
#else
    const uint WGN = get_num_sub_groups();
    const uint WGT = get_local_size(0);
#endif

    __local uint slm_cnt[2][QSA_TOPK_WG][8];
    __local uint slm_tie[QSA_TOPK_WG][2];

    const uint token = get_group_id(0) + row_begin;
    if (token >= row_end || token >= num_tokens) {
        return;
    }
    const uint sg = get_sub_group_id();
    const uint lane = get_sub_group_local_id();
    const uint lid = get_local_id(0);

    uint seq = 0, abs_pos = 0;
#if QSA_ADDRESS_MODE == 0
    if (!FUNC_CALL(qsa_resolve_token_paged)(subsequence_begins, past_lens, num_seqs, token, &seq, &abs_pos)) {
        return;
    }
#else
    FUNC_CALL(qsa_resolve_token_dense)(past_lens, seq_len, token, &seq, &abs_pos);
#endif

    const uint n = FUNC_CALL(qsa_num_complete_blocks)(abs_pos);
    const uint k = n < K ? n : K;
    __global int* out = selected_blocks + (size_t)token * K;
    if (lid == 0) {
        selected_counts[token] = (int)k;
    }
    for (uint i = k + lid; i < K; i += WGT) {
        out[i] = -1;
    }
    if (k == 0) {
        return;
    }
    if (k == n) {
        // Dense prefix: every visible block, ascending; no score is read.
        for (uint i = lid; i < k; i += WGT) {
            out[i] = (int)i;
        }
        return;
    }

    const __global float* scores = block_scores + (size_t)(token - row_begin) * pitch;

    // 64-aligned consecutive ownership, possibly empty subgroups; lane l of a subgroup owns the nj elements begin + l * nj + j.
    const uint chunk = ((n + WGN * 64 - 1) / (WGN * 64)) * 64;
    const uint begin = sg * chunk;
    const uint end = min(begin + chunk, n);
    const bool use_reg = chunk <= QSA_KPL * QSA_SG;
    const uint nj = chunk / QSA_SG;  // keys per lane

    uint rk[QSA_KPL];
    if (use_reg) {
        // Every slot below the run length rounded up to QSA_GRP is defined: 0 past the run / the row.
        unroll_for (uint j = 0; j < QSA_KPL; j += 4) {
            if (j >= nj + (QSA_GRP - 4)) {
                break;
            }
            const uint idx = begin + lane * nj + j;
            const uint lim = j < nj ? min(end - min(end, idx), 4u) : 0u;  // valid keys of this quad
            float4 v = (float4)(0.0f);
            if (lim == 4) {
                v = vload4(0, scores + idx);
            } else {
                v.s0 = lim > 0 ? scores[idx + 0] : 0.0f;
                v.s1 = lim > 1 ? scores[idx + 1] : 0.0f;
                v.s2 = lim > 2 ? scores[idx + 2] : 0.0f;
                v.s3 = lim > 3 ? scores[idx + 3] : 0.0f;
            }
            rk[j + 0] = lim > 0 ? FUNC_CALL(qsa_ordered_key)(v.s0) : 0u;
            rk[j + 1] = lim > 1 ? FUNC_CALL(qsa_ordered_key)(v.s1) : 0u;
            rk[j + 2] = lim > 2 ? FUNC_CALL(qsa_ordered_key)(v.s2) : 0u;
            rk[j + 3] = lim > 3 ? FUNC_CALL(qsa_ordered_key)(v.s3) : 0u;
        }
    }
// Visits this lane's keys in ascending block order: `key`, `idx` and `valid` (inside the row; `key` is 0 otherwise) are in
// scope in BODY. The register walk tests the run length once per QSA_GRP keys to keep branches out of the hot loops.
#define QSA_FOR_KEYS(...)                                                                                  \
    if (use_reg) {                                                                                         \
        unroll_for (uint jb = 0; jb < QSA_KPL; jb += QSA_GRP) {                                            \
            if (jb >= nj) {                                                                                \
                break;                                                                                     \
            }                                                                                              \
            unroll_for (uint jj = 0; jj < QSA_GRP; jj++) {                                                 \
                const uint j = jb + jj;                                                                    \
                const uint idx = begin + lane * nj + j;                                                    \
                const bool valid = j < nj && idx < end;                                                    \
                const uint key = rk[j];                                                                    \
                __VA_ARGS__                                                                                \
            }                                                                                              \
        }                                                                                                  \
    } else {                                                                                               \
        for (uint j = 0; j < nj; j++) {                                                                    \
            const uint idx = begin + lane * nj + j;                                                        \
            const bool valid = idx < end;                                                                  \
            const uint key = valid ? FUNC_CALL(qsa_ordered_key)(scores[idx]) : 0u;                         \
            __VA_ARGS__                                                                                    \
        }                                                                                                  \
    }

    // ---- (NT + 1)-ary count search for the k-th largest key: it stays inside [lo, hi] ----
    uint lo = 0, hi = 0xFFFFFFFFu;
    uint iter = 0;
    while (lo < hi) {
        const uint step = (hi - lo) / (QSA_NT + 1) + 1;
        uint th[QSA_NT];
        uint cnt[QSA_NT];
        unroll_for (uint i = 0; i < QSA_NT; i++) {
            const ulong m = (ulong)lo + (ulong)(i + 1) * step;
            th[i] = m > hi ? 0xFFFFFFFFu : (uint)m;
            cnt[i] = 0;
        }
        QSA_FOR_KEYS(
            unroll_for (uint i = 0; i < QSA_NT; i++) {
                cnt[i] += key >= th[i] ? 1u : 0u;
            })
        const uint slot = iter & 1;
        unroll_for (uint i = 0; i < QSA_NT; i++) {
            cnt[i] = sub_group_reduce_add(cnt[i]);
        }
        if (lane == 0) {
            unroll_for (uint i = 0; i < QSA_NT; i++) {
                slm_cnt[slot][sg][i] = cnt[i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        uint8 tot = (uint8)(0);
        for (uint t = 0; t < WGN; t++) {
            tot += vload8(0, &slm_cnt[slot][t][0]);
        }
        const uint tc[8] = {tot.s0, tot.s1, tot.s2, tot.s3, tot.s4, tot.s5, tot.s6, tot.s7};
        uint nlo = lo, nhi = hi;
        bool below = false;
        unroll_for (uint i = 0; i < QSA_NT; i++) {
            const ulong m = (ulong)lo + (ulong)(i + 1) * step;
            if (m <= hi && !below) {
                if (tc[i] >= k) {
                    nlo = (uint)m;
                } else {
                    nhi = (uint)m - 1;
                    below = true;
                }
            }
        }
        lo = nlo;
        hi = nhi;
        iter++;
    }

    // ---- entries strictly above / exactly at the threshold ----
    uint lane_greater = 0, lane_equal = 0;
    QSA_FOR_KEYS(
        lane_greater += (valid && key > lo) ? 1u : 0u;
        lane_equal += (valid && key == lo) ? 1u : 0u;)
    const uint greater = sub_group_reduce_add(lane_greater);
    const uint equal = sub_group_reduce_add(lane_equal);
    if (lane == 0) {
        slm_tie[sg][0] = greater;
        slm_tie[sg][1] = equal;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // Every subgroup derives its own output offset and its share of the threshold-level entries.
    uint total_greater = 0;
    for (uint t = 0; t < WGN; t++) {
        total_greater += slm_tie[t][0];
    }
    uint equals_left = k - total_greater, offset = 0, my_equal = 0;
    for (uint t = 0; t < WGN; t++) {
        const uint g = slm_tie[t][0];
        const uint e = min(equals_left, slm_tie[t][1]);
        if (t == sg) {
            my_equal = e;
            break;
        }
        offset += g + e;
        equals_left -= e;
    }

    // The first `my_equal` threshold-level entries of the subgroup (in block order) are taken: split them over lanes.
    const uint eq_before = sub_group_scan_exclusive_add(lane_equal);
    uint quota = my_equal > eq_before ? min(lane_equal, my_equal - eq_before) : 0u;
    uint written = offset + sub_group_scan_exclusive_add(lane_greater + quota);
    QSA_FOR_KEYS(
        if (valid && key > lo) {
            out[written++] = (int)idx;
        } else if (valid && key == lo && quota != 0) {
            quota--;
            out[written++] = (int)idx;
        })
}
