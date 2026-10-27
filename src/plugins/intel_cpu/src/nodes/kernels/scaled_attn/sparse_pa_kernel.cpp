// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "nodes/kernels/scaled_attn/sparse_pa_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "nodes/kernels/scaled_attn/attn_quant_kernel.hpp"
#include "nodes/kernels/scaled_attn/common.hpp"
#include "nodes/kernels/scaled_attn/paged_attn_kernel.hpp"
#include "nodes/kernels/scaled_attn/softmax_kernel.hpp"
#ifdef OPENVINO_ARCH_X86_64
#    include "nodes/kernels/x64/brgemm_kernel.hpp"
#endif
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

using ov::intel_cpu::PlainTensor;

namespace {

// ---- quantized (u8/u4) paged cache access, same layouts as PagedAttention (see SparsePAQuant) ----

// Dequantizes `n` consecutive by-token rows [slot0, slot0 + n) of one (physical block, kv head) into dst[n, D].
template <typename TDST, ov::element::Type_t PREC>
void dequant_rows_by_token(const PlainTensor& cache,
                           size_t phys_block,
                           size_t kv,
                           size_t slot0,
                           size_t n,
                           size_t D,
                           size_t group_size,
                           TDST* dst) {
    constexpr size_t sub_byte_multiplier = get_sub_byte_multiplier(PREC);
    constexpr size_t params_offset = sizeof(float) * 2;
    const size_t gs = group_size ? group_size : D;
    const auto* s = cache.ptr<uint8_t, PREC>(phys_block, kv, slot0);
    for (size_t r = 0; r < n; r++) {
        size_t src_offset = 0;
        for (size_t d = 0; d < D; d += gs) {
            const auto* params = reinterpret_cast<const float*>(s + src_offset);
            attn_dequant_kernel<TDST, PREC>(s + src_offset + params_offset, dst + d, gs, params);
            src_offset += gs / sub_byte_multiplier + params_offset;
        }
        s += src_offset;
        dst += D;
    }
}

// Dequantizes `n` consecutive rows [slot0, slot0 + n) of a by-channel key page: the page head holds the
// per-channel scales and zero points, followed by the quantized [block_size, S] rows.
template <typename TDST, ov::element::Type_t PREC>
void dequant_key_rows_by_channel(const PlainTensor& cache,
                                 size_t phys_block,
                                 size_t kv,
                                 size_t slot0,
                                 size_t n,
                                 size_t S,
                                 TDST* dst) {
    constexpr size_t sub_byte_multiplier = get_sub_byte_multiplier(PREC);
    const auto* page = cache.ptr<uint8_t, PREC>(phys_block, kv);
    const auto* scales = reinterpret_cast<const float*>(page);
    const float* zps = scales + S;
    const uint8_t* data = page + sizeof(float) * 2 * S;
    const size_t row_stride = S / sub_byte_multiplier;
    attn_dequant_by_channel_kernel<TDST, PREC>(data + slot0 * row_stride, dst, n, S, row_stride, S, scales, zps);
}

// The cache precision is a template parameter of the kernels below; each *_rows function picks u8 or u4 at runtime.

template <typename TDST, ov::element::Type_t PREC>
void dequant_k_rows_prec(const SparsePAQuant& quant,
                         const PlainTensor& cache,
                         size_t phys_block,
                         size_t kv,
                         size_t slot0,
                         size_t n,
                         size_t S,
                         TDST* dst) {
    if (quant.key_bychannel) {
        dequant_key_rows_by_channel<TDST, PREC>(cache, phys_block, kv, slot0, n, S, dst);
    } else {
        dequant_rows_by_token<TDST, PREC>(cache, phys_block, kv, slot0, n, S, quant.key_group_size, dst);
    }
}

template <typename TDST>
void dequant_k_rows(const SparsePAQuant& quant,
                    const PlainTensor& cache,
                    size_t phys_block,
                    size_t kv,
                    size_t slot0,
                    size_t n,
                    size_t S,
                    TDST* dst) {
    if (quant.key_prec == ov::element::u4) {
        dequant_k_rows_prec<TDST, ov::element::u4>(quant, cache, phys_block, kv, slot0, n, S, dst);
    } else {
        dequant_k_rows_prec<TDST, ov::element::u8>(quant, cache, phys_block, kv, slot0, n, S, dst);
    }
}

template <typename TDST>
void dequant_v_rows(const SparsePAQuant& quant,
                    const PlainTensor& cache,
                    size_t phys_block,
                    size_t kv,
                    size_t slot0,
                    size_t n,
                    size_t Sv,
                    TDST* dst) {
    if (quant.value_prec == ov::element::u4) {
        dequant_rows_by_token<TDST, ov::element::u4>(cache, phys_block, kv, slot0, n, Sv, quant.value_group_size, dst);
    } else {
        dequant_rows_by_token<TDST, ov::element::u8>(cache, phys_block, kv, slot0, n, Sv, quant.value_group_size, dst);
    }
}

template <ov::element::Type_t PREC>
void dot_key_rows_by_token_prec(const SparsePAQuant& quant,
                                const PlainTensor& cache,
                                size_t phys_block,
                                size_t kv,
                                size_t slot0,
                                size_t rows,
                                size_t S,
                                float* q,
                                float* scores) {
    auto* k = cache.ptr<uint8_t, PREC>(phys_block, kv, slot0);
    const size_t gs = quant.key_group_size ? quant.key_group_size : S;
    dot_product_block_quantized<float, PREC>(q, k, scores, S, false, rows, gs);
}

// scores[j] = dot(q, key row slot0 + j) for by-token key rows, straight from the quantized cache.
void dot_key_rows_by_token(const SparsePAQuant& quant,
                           const PlainTensor& cache,
                           size_t phys_block,
                           size_t kv,
                           size_t slot0,
                           size_t rows,
                           size_t S,
                           float* q,
                           float* scores) {
    if (quant.key_prec == ov::element::u4) {
        dot_key_rows_by_token_prec<ov::element::u4>(quant, cache, phys_block, kv, slot0, rows, S, q, scores);
    } else {
        dot_key_rows_by_token_prec<ov::element::u8>(quant, cache, phys_block, kv, slot0, rows, S, q, scores);
    }
}

template <ov::element::Type_t PREC>
void acc_value_rows_by_token_prec(const SparsePAQuant& quant,
                                  const PlainTensor& cache,
                                  size_t phys_block,
                                  size_t kv,
                                  size_t slot0,
                                  size_t rows,
                                  size_t Sv,
                                  float* weights,
                                  float* out) {
    auto* v = cache.ptr<uint8_t, PREC>(phys_block, kv, slot0);
    const size_t gs = quant.value_group_size ? quant.value_group_size : Sv;
    attn_acc_value_block_quantized<float, PREC>(out, weights, v, Sv, false, rows, gs);
}

// out += sum_j weights[j] * (value row slot0 + j), straight from the quantized cache (value is always by-token).
void acc_value_rows_by_token(const SparsePAQuant& quant,
                             const PlainTensor& cache,
                             size_t phys_block,
                             size_t kv,
                             size_t slot0,
                             size_t rows,
                             size_t Sv,
                             float* weights,
                             float* out) {
    if (quant.value_prec == ov::element::u4) {
        acc_value_rows_by_token_prec<ov::element::u4>(quant, cache, phys_block, kv, slot0, rows, Sv, weights, out);
    } else {
        acc_value_rows_by_token_prec<ov::element::u8>(quant, cache, phys_block, kv, slot0, rows, Sv, weights, out);
    }
}

// ---- everything the sparse attention work items read from the arguments, see sparse_pa_compute() ----

struct SparsePAArgs {
    const PlainTensor& query;
    const PlainTensor& key_cache;
    const PlainTensor& value_cache;
    const PlainTensor& block_indices;
    const PlainTensor& block_indices_begins;
    const PlainTensor& subsequence_begins;
    const PlainTensor& past_lens;
    const PlainTensor& sel_indices;
    const PlainTensor& sel_count;
    PlainTensor& out;
    size_t q_head_num;
    size_t k_head_num;
    size_t k_head_size;  // == q_head_size
    size_t v_head_size;
    size_t key_block_size;
    size_t page_size;  // tokens per physical cache block
    size_t sel_head_num;
    bool causal;
    float scale;
    const SparsePAQuant& quant;
    ov::element::Type amx_prec;
    void* gemv_qk1;
    void* gemv_qk4;
    size_t amx_unit_blocks;
    size_t max_run_rows;  // most key rows of one run, see SparsePAScratchLayout
    const SparsePABrgemm& brgemm;
    uint8_t* scratch;
    size_t scratch_bytes_per_thread;
    size_t q_head_pack_size;  // query heads of one work item, chosen per call
};

// State of one work item: one query token and a pack of `pack_size` consecutive query heads that read the same K/V rows
// and use the same selected blocks. It holds where the sequence lives in the caches, the pack's scratch and, per query
// head, the running online-softmax state that the attention_over_* functions below accumulate into.
struct PackWork {
    const PlainTensor& key_cache;
    const PlainTensor& value_cache;
    const int32_t* block_table;  // physical block ids of the sequence
    const SparsePAQuant& quant;
    size_t k_head_size;
    size_t v_head_size;
    size_t k_head;     // the K/V head every query head of the pack reads
    size_t pack_size;  // query heads of the pack
    int64_t key_block_size;
    int64_t page_size;
    int64_t token_pos;  // logical position of the query token
    int64_t valid_max;  // exclusive bound on the valid logical positions of the sequence
    int64_t max_run_rows;
    bool causal;
    float scale;
    const void* query;  // [pack_size, k_head_size] query rows in the data type of the query
    float* q;           // the same rows in f32
    float* acc;         // [pack_size, v_head_size] attention accumulators
    float* running_max;
    float* running_sum;
    float* run_scores;  // [pack_size, max_run_rows] scores, then weights, of a run of key rows
    float* k_run;       // quantized caches: [max_run_rows, k_head_size] dequantized key run
    float* v_run;       // quantized caches: [max_run_rows, v_head_size] dequantized value run
};

// Index of the sequence whose tokens contain `token`.
size_t find_sequence(const PlainTensor& subsequence_begins, size_t token) {
    const auto* begins = subsequence_begins.ptr<int32_t>();
    const auto* end = begins + subsequence_begins.size(0);
    return static_cast<size_t>(std::upper_bound(begins, end, static_cast<int32_t>(token)) - begins) - 1;
}

// Maps a logical KV position of the sequence to the physical (block, slot) address in the caches through the shared
// page table. A selected key_block_size-wide span never crosses a physical block boundary (page_size % key_block_size
// == 0 is enforced by the op and the node).
std::pair<size_t, size_t> physical_address(const PackWork& w, int64_t logical_pos) {
    return {static_cast<size_t>(w.block_table[logical_pos / w.page_size]),
            static_cast<size_t>(logical_pos % w.page_size)};
}

// scores[r] *= scale for r < rows; returns the maximum of the scaled scores.
float scale_and_max(float* scores, size_t rows, float scale) {
    size_t r = 0;
    float max_value = -std::numeric_limits<float>::infinity();
#if defined(HAVE_AVX512F)
    const auto v_scale = _mm512_set1_ps(scale);
    auto v_max = _mm512_set1_ps(max_value);
    for (; r + 16 <= rows; r += 16) {
        const auto v = _mm512_mul_ps(_mm512_loadu_ps(scores + r), v_scale);
        _mm512_storeu_ps(scores + r, v);
        v_max = _mm512_max_ps(v_max, v);
    }
    max_value = _mm512_reduce_max_ps(v_max);
#endif
    for (; r < rows; r++) {
        scores[r] *= scale;
        max_value = std::max(max_value, scores[r]);
    }
    return max_value;
}

// ---- The calculation phases of attention over a run of key rows, for one query head of the pack ----
//   1. qk_scores_*:     scores   = query . keys^T
//   2. online_softmax*: weights  = softmax(scale * scores), rescaling the head's running state (max, sum, accumulator)
//   3. pv_accumulate_*: acc     += weights . values
// attention_over_* run the phases for a run, the selected blocks or the causal tail.

// Phase 2: online-softmax step of query head `head` over its scores[0, rows) of one run of key rows: rescales the
// head's state to the new maximum and turns the scores into weights in place (vectorized exp).
void online_softmax(PackWork& w, size_t head, size_t rows) {
    float* scores = w.run_scores + head * static_cast<size_t>(w.max_run_rows);
    float* acc = w.acc + head * w.v_head_size;
    const float new_max = std::max(w.running_max[head], scale_and_max(scores, rows, w.scale));
    const float corr = std::exp(w.running_max[head] - new_max);
    w.running_sum[head] *= corr;
    for (size_t d = 0; d < w.v_head_size; d++) {
        acc[d] *= corr;
    }
    float sum = 0.0F;
    exp_reduce_sum(scores, new_max, rows, sum);
    w.running_sum[head] += sum;
    w.running_max[head] = new_max;
}

// Phase 1, plain caches: the rows are contiguous in the cache, so PagedAttention's block kernel reads them in place.
template <typename T>
void qk_scores_plain(const PackWork& w, size_t head, T* key_rows, size_t rows) {
    constexpr auto prec = ov::intel_cpu::precision_of<T>::value;
    dot_product_block<T, prec>(const_cast<T*>(static_cast<const T*>(w.query)) + head * w.k_head_size,
                               key_rows,
                               w.run_scores + head * static_cast<size_t>(w.max_run_rows),
                               w.k_head_size,
                               rows,
                               0);
}

// Phase 3, plain caches.
template <typename T>
void pv_accumulate_plain(PackWork& w, size_t head, T* value_rows, size_t rows) {
    constexpr auto prec = ov::intel_cpu::precision_of<T>::value;
    attn_acc_value_block<T, prec>(w.acc + head * w.v_head_size,
                                  w.run_scores + head * static_cast<size_t>(w.max_run_rows),
                                  value_rows,
                                  w.v_head_size,
                                  rows,
                                  0);
}

// Plain caches: attention over up to `n` consecutive logical positions [kpos0, kpos0 + n), all inside one physical
// page, for every query head of the pack: no per-row conversion and one page-table lookup.
template <typename T>
void attention_over_run_plain(PackWork& w, int64_t kpos0, int64_t n) {
    const auto [phys_block, slot] = physical_address(w, kpos0);
    const auto rows = static_cast<size_t>(n);
    T* key_rows = w.key_cache.ptr<T>(phys_block, w.k_head, slot);
    T* value_rows = w.value_cache.ptr<T>(phys_block, w.k_head, slot);
    for (size_t head = 0; head < w.pack_size; head++) {
        qk_scores_plain<T>(w, head, key_rows, rows);
        online_softmax(w, head, rows);
        pv_accumulate_plain<T>(w, head, value_rows, rows);
    }
}

// Quantized caches. A pack of one query head consumes by-token rows straight from the quantized cache with
// PagedAttention's fused block kernels (by-channel key rows are dequantized first, since those kernels walk a whole
// page from its head); a pack of several query heads dequantizes the run's K and V rows to f32 once for all of its
// heads.

// Phase 0: dequantize the rows a fused kernel cannot read in place.
void dequantize_run(PackWork& w, size_t phys_block, size_t slot, size_t rows, bool fused_keys, bool fused_values) {
    if (!fused_keys) {
        dequant_k_rows<float>(w.quant, w.key_cache, phys_block, w.k_head, slot, rows, w.k_head_size, w.k_run);
    }
    if (!fused_values) {
        dequant_v_rows<float>(w.quant, w.value_cache, phys_block, w.k_head, slot, rows, w.v_head_size, w.v_run);
    }
}

// Phase 1, quantized caches.
void qk_scores_quant(PackWork& w, size_t head, size_t phys_block, size_t slot, size_t rows, bool fused_keys) {
    float* q = w.q + head * w.k_head_size;
    float* scores = w.run_scores + head * static_cast<size_t>(w.max_run_rows);
    if (fused_keys) {
        dot_key_rows_by_token(w.quant, w.key_cache, phys_block, w.k_head, slot, rows, w.k_head_size, q, scores);
    } else {
        dot_product_block<float, ov::element::f32>(q, w.k_run, scores, w.k_head_size, rows, 0);
    }
}

// Phase 3, quantized caches.
void pv_accumulate_quant(PackWork& w, size_t head, size_t phys_block, size_t slot, size_t rows, bool fused_values) {
    float* scores = w.run_scores + head * static_cast<size_t>(w.max_run_rows);
    float* acc = w.acc + head * w.v_head_size;
    if (fused_values) {
        acc_value_rows_by_token(w.quant, w.value_cache, phys_block, w.k_head, slot, rows, w.v_head_size, scores, acc);
    } else {
        attn_acc_value_block<float, ov::element::f32>(acc, scores, w.v_run, w.v_head_size, rows, 0);
    }
}

// Quantized caches: attention over a run of rows inside one physical page.
void attention_over_run_quant(PackWork& w, int64_t kpos0, int64_t n) {
    const auto [phys_block, slot] = physical_address(w, kpos0);
    const auto rows = static_cast<size_t>(n);
    const bool fused_keys = w.pack_size == 1 && !w.quant.key_bychannel;
    const bool fused_values = w.pack_size == 1;
    dequantize_run(w, phys_block, slot, rows, fused_keys, fused_values);
    for (size_t head = 0; head < w.pack_size; head++) {
        qk_scores_quant(w, head, phys_block, slot, rows, fused_keys);
        online_softmax(w, head, rows);
        pv_accumulate_quant(w, head, phys_block, slot, rows, fused_values);
    }
}

// Number of rows of the run [kpos0, kpos0 + n) that the query may see.
int64_t visible_rows(const PackWork& w, int64_t kpos0, int64_t n) {
    const int64_t limit = w.causal ? std::min(w.valid_max, w.token_pos + 1) : w.valid_max;
    return kpos0 < 0 ? 0 : std::max<int64_t>(std::min(n, limit - kpos0), 0);
}

// Attention over up to `n` consecutive logical positions [kpos0, kpos0 + n) that lie inside one physical page, clamped
// to the positions the query may see.
template <typename T>
void attention_over_run(PackWork& w, int64_t kpos0, int64_t n) {
    n = visible_rows(w, kpos0, n);
    if (n == 0) {
        return;
    }
    if (w.quant.enabled()) {
        attention_over_run_quant(w, kpos0, n);
    } else {
        attention_over_run_plain<T>(w, kpos0, n);
    }
}

// Attention over the key_block_size-wide block starting at `block_start`.
template <typename T>
void attention_over_block(PackWork& w, int64_t block_start) {
    attention_over_run<T>(w, block_start, w.key_block_size);
}

// Attention over the selected complete blocks: adjacent blocks of the same physical page form one run (up to
// max_run_rows rows), so the block kernels see as many rows per call as possible.
template <typename T>
void attention_over_selected_blocks(PackWork& w, const int32_t* block_ids, size_t nblocks) {
    const int64_t page_blocks = w.page_size / w.key_block_size;
    const int64_t max_run_blocks = std::max<int64_t>(w.max_run_rows / w.key_block_size, 1);
    size_t i = 0;
    while (i < nblocks) {
        const int64_t first = block_ids[i];
        int64_t run_blocks = 1;
        while (i + run_blocks < nblocks && run_blocks < max_run_blocks &&
               block_ids[i + run_blocks] == first + run_blocks &&
               (first + run_blocks) / page_blocks == first / page_blocks) {
            run_blocks++;
        }
        attention_over_run<T>(w, first * w.key_block_size, run_blocks * w.key_block_size);
        i += static_cast<size_t>(run_blocks);
    }
}

// sel_indices/sel_count only ever carry *complete* selected blocks -- the producer never encodes the current
// incomplete trailing block. When causal, the consumer derives and attends that tail itself from the query's own
// logical position; it is empty at a block boundary.
template <typename T>
void attention_over_causal_tail(PackWork& w) {
    if (w.causal) {
        const int64_t tail_start = ((w.token_pos + 1) / w.key_block_size) * w.key_block_size;
        attention_over_run<T>(w, tail_start, w.token_pos - tail_start + 1);
    }
}

#ifdef OPENVINO_ARCH_X86_64
// Decomposition of one (seq, token, sh) row's selected blocks into AMX-mergeable chunks, backed
// entirely by caller-owned scratch (see SparsePAScratchLayout) -- no heap vectors, no
// thread_local. Capacity of each array must be >= nblocks (bounded by selected_block_indices' K_max
// dimension), enough room for the worst case of every block becoming its own chunk/leftover.
// Duplicated from sparse_sdpa_kernel.cpp's identical struct rather than shared, matching this
// repo's convention for these small kernel-local helpers.
struct DecomposedRuns {
    int64_t* chunk_block_start = nullptr;
    int64_t* chunk_n_units = nullptr;
    size_t chunk_count = 0;
    int64_t* leftover_block_start = nullptr;
    size_t leftover_count = 0;
};

// Paged-cache counterpart of sparse_sdpa_kernel.cpp's decompose_runs(): identical ascending-run /
// 4-unit / 1-unit / leftover greedy split, with one addition -- a run is also capped at a
// physical-page boundary, since two logically-adjacent selected blocks are only guaranteed
// memory-contiguous (and therefore safe to feed into one larger AMX call) when they land in the
// same physical page. Because page_size % key_block_size == 0 is enforced by the PagedSparseAttention
// shape inference (for the plain f32/f16/bf16 caches this node accepts), an absolute selected block id B
// always belongs to physical page B / page_sel_blocks -- a pure arithmetic property of this sequence's own logical
// block-id space, independent of the page-table permutation. The page-id comparison below MUST use sel_ids[...] (this
// logical numbering), never the physical block number returned by token_phys() -- the two numberings are unrelated, and
// mixing them would silently produce wrong merges.
void decompose_runs_paged(const int32_t* sel_ids,
                          size_t nblocks,
                          size_t Bk,
                          size_t unit_blocks,
                          size_t page_sel_blocks,
                          DecomposedRuns& result) {
    result.chunk_count = 0;
    result.leftover_count = 0;
    size_t i = 0;
    while (i < nblocks) {
        size_t run_len = 1;
        while (i + run_len < nblocks && sel_ids[i + run_len] == sel_ids[i] + static_cast<int32_t>(run_len) &&
               static_cast<size_t>(sel_ids[i + run_len]) / page_sel_blocks ==
                   static_cast<size_t>(sel_ids[i]) / page_sel_blocks) {
            run_len++;
        }
        size_t j = 0;
        while (run_len - j >= 4 * unit_blocks) {
            result.chunk_block_start[result.chunk_count] =
                static_cast<int64_t>(sel_ids[i + j]) * static_cast<int64_t>(Bk);
            result.chunk_n_units[result.chunk_count] = 4;
            result.chunk_count++;
            j += 4 * unit_blocks;
        }
        while (run_len - j >= unit_blocks) {
            result.chunk_block_start[result.chunk_count] =
                static_cast<int64_t>(sel_ids[i + j]) * static_cast<int64_t>(Bk);
            result.chunk_n_units[result.chunk_count] = 1;
            result.chunk_count++;
            j += unit_blocks;
        }
        while (j < run_len) {
            result.leftover_block_start[result.leftover_count++] =
                static_cast<int64_t>(sel_ids[i + j]) * static_cast<int64_t>(Bk);
            j += 1;
        }
        i += run_len;
    }
}

// What the AMX path of one work item needs besides PackWork.
template <typename AmxT>
struct AmxWork {
    ov::Extensions::Cpu::JitMatMulVecAMX* gemv1;  // one unit of rows
    ov::Extensions::Cpu::JitMatMulVecAMX* gemv4;  // four units of rows
    size_t unit_blocks;                           // key blocks per unit of 16-multiple rows
    const AmxT* q;                                // [pack_size, k_head_size] query rows
    AmxT* k_dq;                                   // quantized caches: dequantized key chunk
    AmxT* v_dq;                                   // quantized caches: dequantized value chunk
    float* raw;                                   // [pack_size, 4 * unit rows] chunk scores, then weights
};

// Phase 1 for an AMX chunk of `n_units` units of rows: raw[head] = query . keys^T on AMX (matrix-vector).
template <typename AmxT>
void qk_scores_amx_chunk(const PackWork& w,
                         const AmxWork<AmxT>& amx,
                         size_t head,
                         const AmxT* k_amx,
                         size_t n_units,
                         float* raw) {
    auto& kernel = n_units == 4 ? *amx.gemv4 : *amx.gemv1;
    kernel(amx.q + head * w.k_head_size, k_amx, raw);
}

// Phase 2 for an AMX chunk whose raw scores are computed: online-softmax step over the visible prefix of the chunk, the
// rows after it get weight 0 so that phase 3 needs no per-row branch. Returns false for a fully masked chunk (only the
// last chunk can be, under causal), which contributes nothing.
bool online_softmax_chunk(PackWork& w, size_t head, float* raw, int64_t block_start, size_t chunk_len) {
    const int64_t limit = w.causal ? std::min(w.valid_max, w.token_pos + 1) : w.valid_max;
    const auto valid_rows = static_cast<size_t>(std::clamp<int64_t>(limit - block_start, 0, chunk_len));
    if (valid_rows == 0) {
        return false;
    }
    float* acc = w.acc + head * w.v_head_size;
    const float new_max = std::max(w.running_max[head], scale_and_max(raw, valid_rows, w.scale));
    const float corr = std::exp(w.running_max[head] - new_max);
    w.running_sum[head] *= corr;
    for (size_t d = 0; d < w.v_head_size; d++) {
        acc[d] *= corr;
    }
    float sum = 0.0F;
    exp_reduce_sum(raw, new_max, valid_rows, sum);
    w.running_sum[head] += sum;
    w.running_max[head] = new_max;
    std::fill(raw + valid_rows, raw + chunk_len, 0.0F);
    return true;
}

// Phase 3 for an AMX chunk: acc += weights . values with the vectorized batched FMA.
template <typename AmxT>
void pv_accumulate_amx_chunk(PackWork& w, size_t head, const float* weights, AmxT* v_amx, size_t chunk_len) {
    float* acc = w.acc + head * w.v_head_size;
    if constexpr (std::is_same_v<AmxT, ov::bfloat16>) {
        attn_acc_value_block<ov::bfloat16, ov::element::bf16>(acc, weights, v_amx, w.v_head_size, chunk_len, 0);
    } else {
        attn_acc_value_block<ov::float16, ov::element::f16>(acc, weights, v_amx, w.v_head_size, chunk_len, 0);
    }
}

// Attention over the AMX-eligible selected-block chunks of one work item (block-wise online softmax), updating every
// query head's running state so the causal tail can continue it afterwards. Plain caches are read in place; quantized
// caches are dequantized once per chunk -- for the whole pack of query heads -- into AMX-typed scratch.
// decompose_runs_paged() guarantees every row of a chunk lives in one physical page, so the page is resolved once per
// chunk.
template <typename AmxT>
void attention_over_amx_chunks(PackWork& w, const AmxWork<AmxT>& amx, const DecomposedRuns& runs) {
    if (runs.chunk_count == 0) {
        return;
    }
    const size_t unit_rows = amx.unit_blocks * static_cast<size_t>(w.key_block_size);
    const size_t raw_stride = 4 * unit_rows;
    // Tile shape is derived from head_size only (identical for gemv1/gemv4), so one tile_config()/tile_release()
    // bracket covers calls into either kernel below.
    amx.gemv1->tile_config();
    for (size_t idx = 0; idx < runs.chunk_count; idx++) {
        const int64_t block_start = runs.chunk_block_start[idx];
        const size_t n_units = static_cast<size_t>(runs.chunk_n_units[idx]);
        const size_t chunk_len = n_units * unit_rows;
        const auto [phys_block, slot_start] = physical_address(w, block_start);
        const AmxT* k_amx = nullptr;
        AmxT* v_amx = nullptr;
        if (w.quant.enabled()) {
            dequant_k_rows<AmxT>(w.quant,
                                 w.key_cache,
                                 phys_block,
                                 w.k_head,
                                 slot_start,
                                 chunk_len,
                                 w.k_head_size,
                                 amx.k_dq);
            dequant_v_rows<AmxT>(w.quant,
                                 w.value_cache,
                                 phys_block,
                                 w.k_head,
                                 slot_start,
                                 chunk_len,
                                 w.v_head_size,
                                 amx.v_dq);
            k_amx = amx.k_dq;
            v_amx = amx.v_dq;
        } else {
            k_amx = w.key_cache.ptr<AmxT>(phys_block, w.k_head, slot_start);
            v_amx = w.value_cache.ptr<AmxT>(phys_block, w.k_head, slot_start);
        }
        for (size_t head = 0; head < w.pack_size; head++) {
            float* raw = amx.raw + head * raw_stride;
            qk_scores_amx_chunk<AmxT>(w, amx, head, k_amx, n_units, raw);
            if (online_softmax_chunk(w, head, raw, block_start, chunk_len)) {
                pv_accumulate_amx_chunk<AmxT>(w, head, raw, v_amx, chunk_len);
            }
        }
    }
    amx.gemv1->tile_release();
}

// Selected blocks through AMX: merge adjacent blocks into chunks, then run the blocks that could not be merged (a too
// short run, the remainder after merging whole units, or a run capped at a page boundary) one by one.
template <typename AmxT>
void attention_over_selected_blocks_amx(PackWork& w,
                                        const AmxWork<AmxT>& amx,
                                        int64_t* runs_scratch,
                                        size_t max_blocks,
                                        const int32_t* block_ids,
                                        size_t nblocks) {
    DecomposedRuns runs;
    runs.chunk_block_start = runs_scratch;
    runs.chunk_n_units = runs_scratch + max_blocks;
    runs.leftover_block_start = runs_scratch + 2 * max_blocks;
    const size_t page_sel_blocks = static_cast<size_t>(w.page_size / w.key_block_size);
    decompose_runs_paged(block_ids,
                         nblocks,
                         static_cast<size_t>(w.key_block_size),
                         amx.unit_blocks,
                         page_sel_blocks,
                         runs);
    attention_over_amx_chunks<AmxT>(w, amx, runs);
    for (size_t i = 0; i < runs.leftover_count; i++) {
        attention_over_block<AmxT>(w, runs.leftover_block_start[i]);
    }
}
#endif  // OPENVINO_ARCH_X86_64

// Attention over the selected blocks of one work item (not the tail), through AMX when it applies and run by run
// otherwise.
template <typename T>
void attention_over_selection(const SparsePAArgs& a,
                              PackWork& w,
                              [[maybe_unused]] uint8_t* ws,
                              [[maybe_unused]] const SparsePAScratchLayout& layout,
                              [[maybe_unused]] size_t token,
                              [[maybe_unused]] size_t first_head,
                              [[maybe_unused]] int64_t sequence_new_tokens,
                              const int32_t* block_ids,
                              size_t nblocks) {
#ifdef OPENVINO_ARCH_X86_64
    // AMX only exists for 16-bit data. Decode (one query token) reads quantized caches with the run-based kernels;
    // prefill dequantizes chunks into AMX scratch. Plain caches use AMX for both.
    if constexpr (!std::is_same_v<T, float>) {
        const bool use_amx =
            a.amx_prec != ov::element::dynamic && nblocks > 0 && (!w.quant.enabled() || sequence_new_tokens > 1);
        if (use_amx) {
            const AmxWork<T> amx{static_cast<ov::Extensions::Cpu::JitMatMulVecAMX*>(a.gemv_qk1),
                                 static_cast<ov::Extensions::Cpu::JitMatMulVecAMX*>(a.gemv_qk4),
                                 a.amx_unit_blocks,
                                 a.query.ptr<T>(token, first_head * a.k_head_size),
                                 reinterpret_cast<T*>(ws + layout.k_dq),
                                 reinterpret_cast<T*>(ws + layout.v_dq),
                                 reinterpret_cast<float*>(ws + layout.raw)};
            attention_over_selected_blocks_amx<T>(w,
                                                  amx,
                                                  reinterpret_cast<int64_t*>(ws + layout.runs),
                                                  a.sel_indices.size(2),
                                                  block_ids,
                                                  nblocks);
            return;
        }
    }
#endif
    attention_over_selected_blocks<T>(w, block_ids, nblocks);
}

#ifdef OPENVINO_ARCH_X86_64
// ---- prefill with AMX matrix products over the whole selection of a work item ----

constexpr size_t kBrgemmChunkRows = 32;  // rows per brgemm call: the N of the QK kernel and the K of the PV kernels

// Copies (plain caches) or dequantizes (quantized caches) the visible part of the run [kpos0, kpos0 + n) into
// k_dst / v_dst as data type T; returns the number of rows written.
template <typename T>
size_t gather_rows(const PackWork& w, int64_t kpos0, int64_t n, T* k_dst, T* v_dst) {
    n = visible_rows(w, kpos0, n);
    if (n == 0) {
        return 0;
    }
    const auto [phys_block, slot] = physical_address(w, kpos0);
    const auto rows = static_cast<size_t>(n);
    if (w.quant.enabled()) {
        dequant_k_rows<T>(w.quant, w.key_cache, phys_block, w.k_head, slot, rows, w.k_head_size, k_dst);
        dequant_v_rows<T>(w.quant, w.value_cache, phys_block, w.k_head, slot, rows, w.v_head_size, v_dst);
    } else {
        std::memcpy(k_dst, w.key_cache.ptr<T>(phys_block, w.k_head, slot), rows * w.k_head_size * sizeof(T));
        std::memcpy(v_dst, w.value_cache.ptr<T>(phys_block, w.k_head, slot), rows * w.v_head_size * sizeof(T));
    }
    return rows;
}

// Buffers of the brgemm path of one work item, carved from the thread's scratch.
template <typename T>
struct BrgemmBuffers {
    ov::intel_cpu::BrgemmKernel* qk;
    ov::intel_cpu::BrgemmKernel* pv;
    ov::intel_cpu::BrgemmKernel* pv_acc;
    T* k_rows;       // [stride, k_head_size] gathered key rows
    T* v_rows;       // [stride, v_head_size] gathered value rows
    float* weights;  // [pack_size, stride] scores, then weights (in place as T)
    void* wsp;
    void* scratch_a;
    void* scratch_b;  // packed B of one chunk of rows
    size_t stride;
};

// Phase 0: gather the item's rows -- the selected blocks, adjacent blocks of one page in one copy, then the causal tail
// -- and zero pad them to a multiple of kBrgemmChunkRows. Returns the number of rows (without padding).
template <typename T>
size_t gather_selection(const PackWork& w, const BrgemmBuffers<T>& b, const int32_t* block_ids, size_t nblocks) {
    size_t rows = 0;
    const int64_t page_blocks = w.page_size / w.key_block_size;
    const int64_t max_run_blocks = std::max<int64_t>(w.max_run_rows / w.key_block_size, 1);
    size_t i = 0;
    while (i < nblocks) {
        const int64_t first = block_ids[i];
        int64_t run_blocks = 1;
        while (i + run_blocks < nblocks && run_blocks < max_run_blocks &&
               block_ids[i + run_blocks] == first + run_blocks &&
               (first + run_blocks) / page_blocks == first / page_blocks) {
            run_blocks++;
        }
        rows += gather_rows<T>(w,
                               first * w.key_block_size,
                               run_blocks * w.key_block_size,
                               b.k_rows + rows * w.k_head_size,
                               b.v_rows + rows * w.v_head_size);
        i += static_cast<size_t>(run_blocks);
    }
    const int64_t tail_start = ((w.token_pos + 1) / w.key_block_size) * w.key_block_size;
    rows += gather_rows<T>(w,
                           tail_start,
                           w.token_pos - tail_start + 1,
                           b.k_rows + rows * w.k_head_size,
                           b.v_rows + rows * w.v_head_size);
    OPENVINO_ASSERT(rows <= b.stride, "[CPU] sparse_pa: the selection has more rows than the brgemm buffers hold");
    const size_t rows_pad = ov::intel_cpu::rnd_up(rows, kBrgemmChunkRows);
    std::memset(b.k_rows + rows * w.k_head_size, 0, (rows_pad - rows) * w.k_head_size * sizeof(T));
    std::memset(b.v_rows + rows * w.v_head_size, 0, (rows_pad - rows) * w.v_head_size * sizeof(T));
    return rows;
}

// Phase 1: weights[head, row] = query[head] . keys[row] for the pack's heads and every chunk of rows. The keys of a
// chunk are packed into the AMX layout once for all the heads of the pack.
template <typename T>
void qk_scores_brgemm(const PackWork& w, const BrgemmBuffers<T>& b, size_t rows_pad) {
    for (size_t row = 0; row < rows_pad; row += kBrgemmChunkRows) {
        b.qk->copy_buffer_b(b.k_rows + row * w.k_head_size, b.scratch_b);
        b.qk->executeGemm(true,
                          const_cast<void*>(w.query),
                          b.scratch_b,
                          b.weights + row,
                          nullptr,
                          nullptr,
                          b.wsp,
                          b.scratch_a);
    }
}

// Phase 2: softmax of every head over all its rows (vectorized). The weights are converted to T in place; the padding
// rows get weight 0. Leaves the denominators in w.running_sum (the output is normalized at the end).
template <typename T>
void softmax_brgemm(PackWork& w, const BrgemmBuffers<T>& b, size_t rows, size_t rows_pad) {
    for (size_t head = 0; head < w.pack_size; head++) {
        float* scores = b.weights + head * b.stride;
        const float max_score = scale_and_max(scores, rows, w.scale);
        float sum = 0.0F;
        exp_reduce_sum(scores, max_score, rows, sum);
        w.running_sum[head] = sum;
        std::fill(scores + rows, scores + rows_pad, 0.0F);
        cvt_copy(reinterpret_cast<T*>(scores), scores, 1, rows_pad, 0, 0);
    }
}

// Phase 3: w.acc[head] = weights[head] . values for the pack's heads, accumulating over the chunks of rows. The values
// of a chunk are packed into the AMX layout once for all the heads of the pack.
template <typename T>
void pv_accumulate_brgemm(PackWork& w, const BrgemmBuffers<T>& b, size_t rows_pad) {
    auto* weights = reinterpret_cast<T*>(b.weights);
    for (size_t row = 0; row < rows_pad; row += kBrgemmChunkRows) {
        b.pv->copy_buffer_b(b.v_rows + row * w.v_head_size, b.scratch_b);
        (row == 0 ? b.pv : b.pv_acc)
            ->executeGemm(true, weights + row, b.scratch_b, w.acc, nullptr, nullptr, b.wsp, b.scratch_a);
    }
}

// Attention of the pack's query heads over the item's whole selection (the selected blocks and the causal tail) with
// matrix products on AMX (oneDNN brgemm, M = the pack's query heads), like PagedAttention's prefill but over gathered
// rows. The whole selection is processed in one pass, so no online-softmax rescaling is needed. It leaves the
// unnormalized attention in w.acc and the softmax denominators in w.running_sum; the caller normalizes and stores the
// output.
template <typename T>
void attention_over_selection_brgemm(const SparsePAArgs& a,
                                     PackWork& w,
                                     uint8_t* ws,
                                     const SparsePAScratchLayout& layout,
                                     const int32_t* block_ids,
                                     size_t nblocks) {
    const BrgemmBuffers<T> b{static_cast<ov::intel_cpu::BrgemmKernel*>(a.brgemm.qk),
                             static_cast<ov::intel_cpu::BrgemmKernel*>(a.brgemm.pv),
                             static_cast<ov::intel_cpu::BrgemmKernel*>(a.brgemm.pv_acc),
                             reinterpret_cast<T*>(ws + layout.k_rows),
                             reinterpret_cast<T*>(ws + layout.v_rows),
                             reinterpret_cast<float*>(ws + layout.weights),
                             ws + layout.brg_wsp,
                             ws + layout.brg_a,
                             ws + layout.brg_b,
                             a.brgemm.weights_stride};
    const size_t rows = gather_selection<T>(w, b, block_ids, nblocks);
    if (rows == 0) {
        return;  // nothing visible: zero output
    }
    const size_t rows_pad = ov::intel_cpu::rnd_up(rows, kBrgemmChunkRows);
    qk_scores_brgemm<T>(w, b, rows_pad);
    softmax_brgemm<T>(w, b, rows, rows_pad);
    pv_accumulate_brgemm<T>(w, b, rows_pad);
}
#endif  // OPENVINO_ARCH_X86_64

// One work item: sparse attention of one query token and query head pack `pack`.
template <typename T>
void attention_work_item(const SparsePAArgs& a, const SparsePAScratchLayout& layout, size_t token, size_t pack) {
    const auto* subsequence_begins = a.subsequence_begins.ptr<int32_t>();
    const size_t seq = find_sequence(a.subsequence_begins, token);
    if (seq + 1 >= a.subsequence_begins.size(0)) {
        return;  // not a token of any sequence
    }
    const int32_t token_begin = subsequence_begins[seq];
    const int32_t token_end = subsequence_begins[seq + 1];
    const int32_t block_begin = a.block_indices_begins.ptr<int32_t>()[seq];
    OPENVINO_ASSERT(a.block_indices_begins.ptr<int32_t>()[seq + 1] > block_begin,
                    "[CPU] sparse_pa: each sequence must have at least one cache block");
    const int64_t past_len = a.past_lens.ptr<int32_t>()[seq];

    // The query heads of a pack share the K head and the selection head of the pack's first query head.
    const size_t first_head = pack * a.q_head_pack_size;
    const size_t k_head = first_head / (a.q_head_num / a.k_head_num);
    const size_t sel_head = first_head / (a.q_head_num / a.sel_head_num);

    // Per-thread scratch, see SparsePAScratchLayout.
    uint8_t* ws = a.scratch + static_cast<size_t>(parallel_get_thread_num()) * a.scratch_bytes_per_thread;
    PackWork w{a.key_cache,
               a.value_cache,
               a.block_indices.ptr<int32_t>() + block_begin,
               a.quant,
               a.k_head_size,
               a.v_head_size,
               k_head,
               a.q_head_pack_size,
               static_cast<int64_t>(a.key_block_size),
               static_cast<int64_t>(a.page_size),
               past_len + (static_cast<int32_t>(token) - token_begin),
               // Enforced unconditionally (not just under causal) so a non-causal gather never reads an unallocated
               // or stale slot in the physical pages.
               past_len + (token_end - token_begin),
               static_cast<int64_t>(a.max_run_rows),
               a.causal,
               a.scale,
               a.query.ptr<T>(token, first_head * a.k_head_size),
               reinterpret_cast<float*>(ws + layout.q),
               reinterpret_cast<float*>(ws + layout.acc),
               reinterpret_cast<float*>(ws + layout.running_max),
               reinterpret_cast<float*>(ws + layout.running_sum),
               reinterpret_cast<float*>(ws + layout.scores),
               reinterpret_cast<float*>(ws + layout.k_run),
               reinterpret_cast<float*>(ws + layout.v_run)};
    cvt_copy(w.q, a.query.ptr<T>(token, first_head * a.k_head_size), 1, w.pack_size * a.k_head_size, 0, 0);
    std::fill_n(w.acc, w.pack_size * a.v_head_size, 0.0F);
    std::fill_n(w.running_max, w.pack_size, -std::numeric_limits<float>::infinity());
    std::fill_n(w.running_sum, w.pack_size, 0.0F);

    // One query per selection row, so the row of a token is its global token index.
    const int32_t* block_ids = a.sel_indices.ptr<int32_t>(token, sel_head);
    const auto nblocks = static_cast<size_t>(*a.sel_count.ptr<int32_t>(token, sel_head));
    bool done = false;
#ifdef OPENVINO_ARCH_X86_64
    // Prefill (more than one new token) of a pack of query heads with 16-bit data on AMX: matrix products over the
    // whole selection. Decode and single-head packs keep the run-by-run paths below.
    if constexpr (!std::is_same_v<T, float>) {
        if (a.brgemm.enabled() && a.q_head_pack_size > 1 && a.causal && token_end - token_begin > 1) {
            attention_over_selection_brgemm<T>(a, w, ws, layout, block_ids, nblocks);
            done = true;
        }
    }
#endif
    if (!done) {
        attention_over_selection<T>(a, w, ws, layout, token, first_head, token_end - token_begin, block_ids, nblocks);
        attention_over_causal_tail<T>(w);
    }

    for (size_t head = 0; head < w.pack_size; head++) {
        float* acc = w.acc + head * a.v_head_size;
        const float inv_sum = w.running_sum[head] > 0.0F ? 1.0F / w.running_sum[head] : 0.0F;
        for (size_t d = 0; d < a.v_head_size; d++) {
            acc[d] *= inv_sum;
        }
        cvt_copy(a.out.ptr<T>(token, (first_head + head) * a.v_head_size), acc, 1, a.v_head_size, 0, 0);
    }
}

// Sparse attention of every (global query token, query head pack) pair, in parallel.
template <typename T>
void sparse_pa_impl(const SparsePAArgs& a, const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    const SparsePAScratchLayout layout(a.k_head_size,
                                       a.v_head_size,
                                       a.key_block_size,
                                       a.max_run_rows,
                                       a.sel_indices.size(2),
                                       a.amx_unit_blocks,
                                       max_q_head_pack_size(a.q_head_num, a.k_head_num, a.sel_head_num),
                                       sizeof(T),
                                       a.quant,
                                       sparse_pa_brgemm_sizes(a.brgemm));
    OPENVINO_ASSERT(layout.total <= a.scratch_bytes_per_thread, "[CPU] sparse_pa: scratch is too small");
    cpu_parallel->parallel_for2d(a.query.size(0), a.q_head_num / a.q_head_pack_size, [&](size_t token, size_t pack) {
        attention_work_item<T>(a, layout, token, pack);
    });
}

}  // namespace

void sparse_pa_compute(const ov::intel_cpu::PlainTensor& query,
                       const ov::intel_cpu::PlainTensor& key_cache,
                       const ov::intel_cpu::PlainTensor& value_cache,
                       const ov::intel_cpu::PlainTensor& block_indices,
                       const ov::intel_cpu::PlainTensor& block_indices_begins,
                       const ov::intel_cpu::PlainTensor& subsequence_begins,
                       const ov::intel_cpu::PlainTensor& past_lens,
                       const ov::intel_cpu::PlainTensor& sel_indices,
                       const ov::intel_cpu::PlainTensor& sel_count,
                       ov::intel_cpu::PlainTensor& out,
                       size_t q_head_num,
                       size_t k_head_num,
                       size_t k_head_size,
                       size_t v_head_size,
                       size_t key_block_size,
                       size_t sel_head_num,
                       bool causal,
                       float scale,
                       const SparsePAQuant& quant,
                       ov::element::Type amx_prec,
                       void* gemv_qk1,
                       void* gemv_qk4,
                       size_t amx_unit_blocks,
                       size_t max_run_rows,
                       const SparsePABrgemm& brgemm,
                       uint8_t* scratch,
                       size_t scratch_bytes_per_thread,
                       const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    const auto data_prc = query.get_precision();
    OPENVINO_ASSERT(out.get_precision() == data_prc, "[CPU] sparse_pa: query/output precisions must match");
    OPENVINO_ASSERT(
        quant.enabled() || (key_cache.get_precision() == data_prc && value_cache.get_precision() == data_prc),
        "[CPU] sparse_pa: plain caches must have the query precision");
    OPENVINO_ASSERT(scratch != nullptr, "[CPU] sparse_pa: scratch must not be null");
    OPENVINO_ASSERT(subsequence_begins.size(0) >= 1, "[CPU] sparse_pa: subsequence_begins must be non-empty");

    // Query heads of one work item: all that share both K/V rows and the selected blocks (a pack), which lets K/V be
    // dequantized or read once for the pack. A pack means fewer work items, so with too little work to keep every
    // thread busy (e.g. decode of a few sequences) every query head is its own work item instead, like PagedAttention's
    // loop_hk choice.
    const size_t max_pack_size = max_q_head_pack_size(q_head_num, k_head_num, sel_head_num);
    const size_t pack_items = query.size(0) * (q_head_num / max_pack_size);
    const size_t q_head_pack_size =
        pack_items >= 2 * static_cast<size_t>(cpu_parallel->get_num_worker_threads()) ? max_pack_size : 1;

    // The physical page size: the key cache's dim 2 for plain caches; for quantized caches the node passes the real
    // size, since by-channel key pages carry extra parameter rows in dim 2.
    const size_t page_size = quant.block_size ? quant.block_size : key_cache.size(2);
    const SparsePAArgs args{query,
                            key_cache,
                            value_cache,
                            block_indices,
                            block_indices_begins,
                            subsequence_begins,
                            past_lens,
                            sel_indices,
                            sel_count,
                            out,
                            q_head_num,
                            k_head_num,
                            k_head_size,
                            v_head_size,
                            key_block_size,
                            page_size,
                            sel_head_num,
                            causal,
                            scale,
                            quant,
                            amx_prec,
                            gemv_qk1,
                            gemv_qk4,
                            amx_unit_blocks,
                            max_run_rows,
                            brgemm,
                            scratch,
                            scratch_bytes_per_thread,
                            q_head_pack_size};

    // The data type is dispatched once; everything below works on T.
    if (data_prc == ov::element::f32) {
        sparse_pa_impl<float>(args, cpu_parallel);
    } else if (data_prc == ov::element::f16) {
        sparse_pa_impl<ov::float16>(args, cpu_parallel);
    } else if (data_prc == ov::element::bf16) {
        sparse_pa_impl<ov::bfloat16>(args, cpu_parallel);
    } else {
        OPENVINO_THROW("[CPU] sparse_pa: unsupported precision ", data_prc);
    }
}

}  // namespace ov::Extensions::Cpu::XARCH
