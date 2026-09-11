/*******************************************************************************
 * Copyright (c) 2018-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

// Shared address arithmetic, RoPE / RMSNorm helpers and the deterministic radix
// top-k selector used by the four QSA CM kernel families.
//
// Address contract (design sections 6.1 / 7.1.1 / 7.1.2): the main K/V cache, the
// indexer raw-K cache and the QSA block-summary cache all share ONE page table
// (`block_indices` / `block_indices_begins`). Each buffer converts a page index into
// a base address using its own per-page element count. Sharing the page table is what
// keeps selection and the actual KV in sync across seq_rm / seq_cp / beam reorder.

// ATTR must sit OUTSIDE the include guard. The codegen appends an `#undef ATTR` at the
// end of every kernel it inlines this header into, and the guard suppresses the second
// textual copy - so a guarded definition would leave ATTR undefined for every kernel
// after the first one in a batched program.
#ifndef ATTR
#define ATTR [[type("svmptr_t")]]
#endif

// Standalone stage 1: selection/fused scratch may bypass dense rows by default.
// Separate score kernels retain their score-output contract unless the caller
// explicitly pairs them with a bypass-enabled finalizer. Keep defaults outside
// the guard, like ATTR, for flattened multi-kernel sources.
#ifndef QSA_DENSE_BYPASS
#define QSA_DENSE_BYPASS 1
#endif
#ifndef QSA_DENSE_SCORE_BYPASS
#define QSA_DENSE_SCORE_BYPASS 0
#endif
#if QSA_DENSE_SCORE_BYPASS && !QSA_DENSE_BYPASS
#error "Dense score bypass requires QSA_DENSE_BYPASS in the paired finalizer"
#endif

// Classic include guard - see qsa_gr_common.hpp for why `#pragma once` is not usable here.
#ifndef QSA_COMMON_HPP
#define QSA_COMMON_HPP

#include <cm/cm.h>
#include <cm/cmtl.h>

#include "qsa_gr_common.hpp"

namespace qsa {

// --------------------------------------------------------------------------
// Page-table address helpers (element offsets, not bytes)
// --------------------------------------------------------------------------

/// Main f16 K/V cache: [num_blocks, KV_HEADS, PA_BLOCK_SIZE, HEAD_SIZE]
CM_INLINE uint kv_cache_offset(uint physical_page, uint kv_head, uint token_in_page, uint kv_heads, uint pa_block_size, uint head_size) {
    return ((physical_page * kv_heads + kv_head) * pa_block_size + token_in_page) * head_size;
}

/// Indexer raw-K cache: [num_blocks, 1, PA_BLOCK_SIZE, Di] - unnormalized and unrotated.
CM_INLINE uint raw_k_offset(uint physical_page, uint token_in_page, uint pa_block_size, uint idx_head_dim) {
    return (physical_page * pa_block_size + token_in_page) * idx_head_dim;
}

/// QSA block-summary cache: [num_blocks, 1, PA_BLOCK_SIZE / r, Di].
/// Valid only because PA_BLOCK_SIZE % r == 0 is enforced when the primitive is created,
/// so a QSA block never straddles a PA page.
CM_INLINE uint summary_offset(uint physical_page, uint slot_in_page, uint summaries_per_page, uint idx_head_dim) {
    return (physical_page * summaries_per_page + slot_in_page) * idx_head_dim;
}

// --------------------------------------------------------------------------
// Partial NEOX ("rotate_half") RoPE - the convention Qwen3.8 uses.
//
// The reference is transformers' `apply_rotary_pos_emb`:
//     x_rope = x[..., :rd];  x1 = x_rope[:rd/2];  x2 = x_rope[rd/2:]
//     out    = x_rope * cos + cat(-x2, x1) * sin
// where `cos`/`sin` are `cat(freqs, freqs)`, i.e. cos[j] == cos[j + rd/2].
// So frequency j pairs channel j with channel j + rd/2 - NOT the adjacent
// channels (2j, 2j+1) that the GPT-J / "interleaved" variant uses.
//
// `table` is [max_positions, rotary_dim], storing the rd/2 distinct frequencies as
// adjacent (cos_j, sin_j) pairs; the duplicated upper half of the transformers
// cos/sin vectors carries no extra information and is not materialized. Only the
// first `rotary_dim` channels rotate; the rest pass through unchanged, which is what
// "partial rotary" means for Qwen3.8 (Di = 128, rotary_dim = 64). mRoPE section
// handling is folded into the table contents by the host.
// --------------------------------------------------------------------------
template <int HEAD_DIM, int ROTARY_DIM>
CM_INLINE void apply_partial_rope(vector_ref<float, HEAD_DIM> v, const half* table ATTR, uint position) {
    static_assert(ROTARY_DIM % 2 == 0, "rotary dim must be even");
    static_assert(ROTARY_DIM <= HEAD_DIM, "rotary dim must not exceed head dim");
    if constexpr (ROTARY_DIM > 0) {
        constexpr int HALF = ROTARY_DIM / 2;
        const uint base = position * (uint)ROTARY_DIM;
#pragma unroll
        for (int j = 0; j < HALF; j++) {
            const float c = (float)table[base + 2 * j];
            const float s = (float)table[base + 2 * j + 1];
            const float x0 = v[j];
            const float x1 = v[j + HALF];
            v[j] = x0 * c - x1 * s;
            v[j + HALF] = x1 * c + x0 * s;
        }
    }
}

// --------------------------------------------------------------------------
// RMSNorm over one head vector, FP32 accumulation. The gain is applied exactly
// as stored: any `1 + w` folding must happen once, either in lowering or in the
// kernel, never twice (design section 8.5).
// --------------------------------------------------------------------------
template <int HEAD_DIM>
CM_INLINE void rms_norm(vector_ref<float, HEAD_DIM> v, const half* weight ATTR, float eps) {
    static_assert(HEAD_DIM % 16 == 0, "head dim must be a multiple of 16");
    float sumsq = 0.0f;
#pragma unroll
    for (int k = 0; k < HEAD_DIM; k += 16) {
        vector<float, 16> chunk = v.template select<16, 1>(k);
        sumsq += cm_sum<float>(chunk * chunk);
    }
    vector<float, 1> denom = sumsq / (float)HEAD_DIM + eps;
    const float inv = cm_rsqrt(denom)[0];
#pragma unroll
    for (int k = 0; k < HEAD_DIM; k += 16) {
        vector<float, 16> w = qsa_gr::load_h16(weight, k);
        v.template select<16, 1>(k) = v.template select<16, 1>(k) * inv * w;
    }
}

// --------------------------------------------------------------------------
// Deterministic top-k over an FP32 score row.
//
// Order-preserving float->uint key transform plus a 4 x 8-bit radix select, i.e.
// the histogram / radix-select scheme of design section 6.8.4. The tie-break is
// fixed to (score descending, qsa_block_id ascending), so the prefill fused
// variant and the decode partition + finalization variant select the same blocks.
// --------------------------------------------------------------------------

/// Monotonic float -> uint32 key; preserves IEEE-754 ordering for finite values.
CM_INLINE uint float_to_ordered_key(float f) {
    vector<float, 1> fv = f;
    const uint bits = fv.format<uint>()[0];
    const uint mask = (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return bits ^ mask;
}

/// Selects up to `k` largest entries of scores[0 .. n_valid) and writes their indices in
/// ASCENDING index order into `out`. Returns the number of written entries.
///
/// Ascending order is not just cosmetic: the prefill sparse-attention query tile merges the
/// selected lists of 16 queries with a k-way merge, which requires each list to be sorted.
/// The selected SET is still exactly the (score descending, qsa_block_id ascending) top-k -
/// every key strictly above the threshold is taken, and the threshold level contributes
/// exactly `k_remaining` entries, taken in ascending block-id order.
CM_INLINE uint radix_select_topk(const float* scores ATTR, uint n_valid, uint k, int32_t* out ATTR) {
    if (k > n_valid) {
        k = n_valid;
    }
    if (k == 0) {
        return 0;
    }
#if QSA_DENSE_BYPASS
    // After clamping k, equality means ALL valid blocks are selected. Do not
    // read scores: a coordinated producer may have left this entire row unwritten.
    if (k == n_valid) {
        for (uint i = 0; i < k; i++) {
            out[i] = (int32_t)i;
        }
        return k;
    }
#endif

    uint prefix = 0;     // resolved high bits of the k-th largest key
    uint hi_shift = 32;  // 32 == no high bits resolved yet
    uint k_remaining = k;

    for (uint pass = 0; pass < 4; pass++) {
        const uint shift = 24 - 8 * pass;
        vector<uint, 256> hist = 0;

        for (uint i = 0; i < n_valid; i++) {
            const uint key = float_to_ordered_key(scores[i]);
            if (hi_shift < 32 && (key >> hi_shift) != (prefix >> hi_shift)) {
                continue;
            }
            hist[(key >> shift) & 0xFFu] += 1;
        }

        uint cnt = 0;
        uint chosen = 0;
        for (int b = 255; b >= 0; b--) {
            const uint c = hist[b];
            if (cnt + c >= k_remaining) {
                chosen = (uint)b;
                break;
            }
            cnt += c;
        }
        k_remaining -= cnt;
        prefix |= (chosen << shift);
        hi_shift = shift;
    }

    // Single ascending pass: everything above the threshold key, plus exactly k_remaining
    // entries at the threshold level.
    uint written = 0;
    uint equal_left = k_remaining;
    for (uint i = 0; i < n_valid && written < k; i++) {
        const uint key = float_to_ordered_key(scores[i]);
        if (key > prefix) {
            out[written++] = (int32_t)i;
        } else if (key == prefix && equal_left > 0) {
            out[written++] = (int32_t)i;
            equal_left--;
        }
    }
    return written;
}

}  // namespace qsa

#endif  // QSA_COMMON_HPP
