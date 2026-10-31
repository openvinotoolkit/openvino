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
#include "nodes/kernels/scaled_attn/sparse_brgemm_kernel.hpp"
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

// ---- the items of a call ----
// Tokens are split by algorithm: multi-token sequences (prefill) take the brgemm path (AMX over a head pack) when the
// node built the kernels; all others take the fallback SIMD / GEMV paths. `token_ids` lists brgemm tokens first.
// With little fallback work, every query head is its own item (`fallback_pack` 1), like PagedAttention's loop_hk.
struct WorkList {
    const int32_t* token_ids = nullptr;
    size_t brgemm_tokens = 0;
    size_t fallback_tokens = 0;
    size_t brgemm_pack = 1;    ///< query heads of one item of a brgemm token
    size_t fallback_pack = 1;  ///< query heads of one item of a fallback token
};

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
    const SparseBrgemm& brgemm;
    uint8_t* scratch;
    size_t scratch_bytes_per_thread;
    WorkList work;
};

// State of one work item: a query token and a pack of `pack_size` consecutive query heads sharing K/V rows and selected
// blocks. Holds the sequence's cache location, the pack's scratch and each head's running online-softmax state.
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

// ---- The phases of attention over a run of key rows, for one query head of the pack ----
//   1. qk_scores_*:     scores   = query . keys^T
//   2. online_softmax*: weights  = softmax(scale * scores), rescaling the running (max, sum, acc)
//   3. pv_accumulate_*: acc     += weights . values
// attention_over_* run them for a run, the selected blocks or the causal tail.

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

// Quantized caches. A one-head pack reads by-token rows directly with PagedAttention's fused block kernels (by-channel
// keys are dequantized first: those kernels walk a whole page). A multi-head pack dequantizes the run's K and V to f32
// once for all its heads.

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
void attention_selected_blocks_simd(PackWork& w, const int32_t* block_ids, size_t nblocks) {
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
void attention_causal_tail_legacy(PackWork& w) {
    if (w.causal) {
        const int64_t tail_start = ((w.token_pos + 1) / w.key_block_size) * w.key_block_size;
        attention_over_run<T>(w, tail_start, w.token_pos - tail_start + 1);
    }
}

#ifdef OPENVINO_ARCH_X86_64
// A row's selected blocks (any order) split into KV chunks plus leftovers, in caller scratch (>= nblocks per array).
// A chunk's K rows feed an AMX QK^T, its V rows P@V; blocks are adjacent in one page (in place) or gathered.
// Duplicated from sparse_sdpa_kernel.cpp, as kernel-local helpers are in this repo.
struct KvChunks {
    int64_t* block_starts = nullptr;  // logical start (id * key_block_size) of each chunk block, chunk after chunk
    size_t block_count = 0;
    int64_t* chunk_units = nullptr;  // merge units (1 or 4) per chunk
    size_t chunk_count = 0;
    int64_t* leftover_block_start = nullptr;  // blocks that fill no whole unit: attended block by block
    size_t leftover_count = 0;
};

// Paged counterpart of split_into_kv_chunks(): adjacent blocks form an in-place chunk only within one physical page.
// Compare pages by the logical ids in sel_ids (id / page_sel_blocks), never by the physical block numbers of the page
// table: the two numberings are unrelated.
void split_into_kv_chunks_paged(const int32_t* sel_ids,
                                size_t nblocks,
                                size_t Bk,
                                size_t unit_blocks,
                                size_t page_sel_blocks,
                                KvChunks& result) {
    result.chunk_count = 0;
    result.block_count = 0;
    result.leftover_count = 0;
    const auto start_of = [&](size_t i) {
        return static_cast<int64_t>(sel_ids[i]) * static_cast<int64_t>(Bk);
    };
    // Pass 1. Example with unit_blocks = 1, ids {5,6,7,8,9, 20, 30,31}: runs are {5..9}, {20}, {30,31}.
    // Run {5..9} gives one 4-unit chunk {5..8}; block 9 and the short runs become pending leftovers.
    size_t i = 0;
    while (i < nblocks) {
        // The run is the maximal stretch of consecutive ids from position i that stays inside one physical page.
        size_t run_len = 1;
        while (i + run_len < nblocks && sel_ids[i + run_len] == sel_ids[i] + static_cast<int32_t>(run_len) &&
               static_cast<size_t>(sel_ids[i + run_len]) / page_sel_blocks ==
                   static_cast<size_t>(sel_ids[i]) / page_sel_blocks) {
            run_len++;
        }
        // Carve whole chunks off the run, biggest first (4 units, then 1); j counts the run blocks consumed.
        size_t j = 0;
        for (const size_t units : {size_t{4}, size_t{1}}) {
            const size_t blocks = units * unit_blocks;
            while (run_len - j >= blocks) {
                for (size_t k = 0; k < blocks; k++) {
                    result.block_starts[result.block_count++] = start_of(i + j + k);
                }
                result.chunk_units[result.chunk_count++] = static_cast<int64_t>(units);
                j += blocks;
            }
        }
        // What does not fill a unit is kept for pass 2.
        for (; j < run_len; j++) {
            result.leftover_block_start[result.leftover_count++] = start_of(i + j);  // pending, see below
        }
        i += run_len;
    }
    // Pass 2. The pending blocks, in selection order, are not adjacent: form the same chunks from them; the K/V rows
    // of such a chunk are gathered by id. `taken` counts the pending blocks consumed.
    const size_t pending = result.leftover_count;
    size_t taken = 0;
    for (const size_t units : {size_t{4}, size_t{1}}) {
        const size_t blocks = units * unit_blocks;
        while (pending - taken >= blocks) {
            std::copy_n(result.leftover_block_start + taken, blocks, result.block_starts + result.block_count);
            result.block_count += blocks;
            result.chunk_units[result.chunk_count++] = static_cast<int64_t>(units);
            taken += blocks;
        }
    }
    // Move the blocks that fit no chunk to the front of the leftover list.
    if (taken > 0) {
        std::copy(result.leftover_block_start + taken,
                  result.leftover_block_start + pending,
                  result.leftover_block_start);
    }
    result.leftover_count = pending - taken;
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

// Phase 2 for an AMX chunk with raw scores computed: online-softmax over its first `valid_rows` rows; later rows get
// weight 0, so phase 3 needs no per-row branch. Returns false for a fully masked chunk (causal, last chunk only).
bool online_softmax_rows(PackWork& w, size_t head, float* raw, size_t valid_rows, size_t chunk_len) {
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

// The same for the chunk of adjacent rows starting at block_start: the rows the query attends are a prefix of it.
bool online_softmax_chunk(PackWork& w, size_t head, float* raw, int64_t block_start, size_t chunk_len) {
    const int64_t limit = w.causal ? std::min(w.valid_max, w.token_pos + 1) : w.valid_max;
    const auto valid_rows = static_cast<size_t>(std::clamp<int64_t>(limit - block_start, 0, chunk_len));
    return online_softmax_rows(w, head, raw, valid_rows, chunk_len);
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

// Copies (or dequantizes) the key rows of a gathered chunk's blocks, from any pages, into contiguous scratch. Quantized
// values go to v_buf; plain values are read in place by the caller.
template <typename AmxT>
void gather_chunk_rows(const PackWork& w, const int64_t* starts, size_t n_blocks, AmxT* k_buf, AmxT* v_buf) {
    const auto rows = static_cast<size_t>(w.key_block_size);
    for (size_t j = 0; j < n_blocks; j++) {
        const auto [phys_block, slot] = physical_address(w, starts[j]);
        AmxT* k_dst = k_buf + j * rows * w.k_head_size;
        if (w.quant.enabled()) {
            dequant_k_rows<AmxT>(w.quant, w.key_cache, phys_block, w.k_head, slot, rows, w.k_head_size, k_dst);
            dequant_v_rows<AmxT>(w.quant,
                                 w.value_cache,
                                 phys_block,
                                 w.k_head,
                                 slot,
                                 rows,
                                 w.v_head_size,
                                 v_buf + j * rows * w.v_head_size);
        } else {
            std::memcpy(k_dst, w.key_cache.ptr<AmxT>(phys_block, w.k_head, slot), rows * w.k_head_size * sizeof(AmxT));
        }
    }
}

// Attends the KV chunks of one work item, updating each head's running softmax state. Adjacent chunks of one page read
// plain caches in place; the others are gathered by id into AMX-typed scratch (plain values stay in place).
template <typename AmxT>
void attention_over_amx_chunks(PackWork& w, const AmxWork<AmxT>& amx, const KvChunks& chunks) {
    if (chunks.chunk_count == 0) {
        return;
    }
    const size_t unit_rows = amx.unit_blocks * static_cast<size_t>(w.key_block_size);
    const size_t raw_stride = 4 * unit_rows;
    const auto Bk = w.key_block_size;
    // Tile shape is derived from head_size only (identical for gemv1/gemv4), so one tile_config()/tile_release()
    // bracket covers calls into either kernel below.
    amx.gemv1->tile_config();
    size_t first_block = 0;  // of the chunk, in chunks.block_starts
    for (size_t idx = 0; idx < chunks.chunk_count; idx++) {
        const size_t n_units = static_cast<size_t>(chunks.chunk_units[idx]);
        const size_t n_blocks = n_units * amx.unit_blocks;
        const int64_t* starts = chunks.block_starts + first_block;
        first_block += n_blocks;
        const size_t chunk_len = n_units * unit_rows;
        bool adjacent = true;  // consecutive blocks of one physical page
        for (size_t j = 1; j < n_blocks; j++) {
            adjacent = adjacent && starts[j] == starts[0] + static_cast<int64_t>(j) * Bk &&
                       starts[j] / w.page_size == starts[0] / w.page_size;
        }

        if (adjacent) {
            const int64_t block_start = starts[0];
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
            continue;
        }

        // Selected blocks are complete and, under causal, before the query, so all their rows are attended; a block
        // that is not (a producer outside the contract) takes the generic path, which clamps to the visible rows.
        bool attended = true;
        for (size_t j = 0; j < n_blocks; j++) {
            attended = attended && visible_rows(w, starts[j], Bk) == Bk;
        }
        if (!attended) {
            for (size_t j = 0; j < n_blocks; j++) {
                attention_over_block<AmxT>(w, starts[j]);
            }
            continue;
        }
        gather_chunk_rows<AmxT>(w, starts, n_blocks, amx.k_dq, amx.v_dq);
        for (size_t head = 0; head < w.pack_size; head++) {
            float* raw = amx.raw + head * raw_stride;
            qk_scores_amx_chunk<AmxT>(w, amx, head, amx.k_dq, n_units, raw);
            if (!online_softmax_rows(w, head, raw, chunk_len, chunk_len)) {
                continue;
            }
            if (w.quant.enabled()) {
                pv_accumulate_amx_chunk<AmxT>(w, head, raw, amx.v_dq, chunk_len);
                continue;
            }
            for (size_t j = 0; j < n_blocks; j++) {
                const auto [phys_block, slot] = physical_address(w, starts[j]);
                pv_accumulate_amx_chunk<AmxT>(w,
                                              head,
                                              raw + j * static_cast<size_t>(Bk),
                                              w.value_cache.ptr<AmxT>(phys_block, w.k_head, slot),
                                              static_cast<size_t>(Bk));
            }
        }
    }
    amx.gemv1->tile_release();
}

// Selected blocks through AMX: chunks of adjacent blocks (used in place) and of gathered blocks (any order), then the
// blocks that fill no whole merge unit one by one.
template <typename AmxT>
void attention_selected_blocks_gemv(PackWork& w,
                                    const AmxWork<AmxT>& amx,
                                    int64_t* kv_chunk_scratch,
                                    size_t max_blocks,
                                    const int32_t* block_ids,
                                    size_t nblocks) {
    KvChunks kv_chunks;
    kv_chunks.block_starts = kv_chunk_scratch;
    kv_chunks.chunk_units = kv_chunk_scratch + max_blocks;
    kv_chunks.leftover_block_start = kv_chunk_scratch + 2 * max_blocks;
    const size_t page_sel_blocks = static_cast<size_t>(w.page_size / w.key_block_size);
    split_into_kv_chunks_paged(block_ids,
                               nblocks,
                               static_cast<size_t>(w.key_block_size),
                               amx.unit_blocks,
                               page_sel_blocks,
                               kv_chunks);
    attention_over_amx_chunks<AmxT>(w, amx, kv_chunks);
    for (size_t i = 0; i < kv_chunks.leftover_count; i++) {
        attention_over_block<AmxT>(w, kv_chunks.leftover_block_start[i]);
    }
}
#endif  // OPENVINO_ARCH_X86_64

// Attention over the selected blocks of one work item (not the tail), through AMX when it applies and run by run
// otherwise.
template <typename T>
void attention_selected_blocks_legacy(const SparsePAArgs& a,
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
            attention_selected_blocks_gemv<T>(w,
                                              amx,
                                              reinterpret_cast<int64_t*>(ws + layout.kv_chunks),
                                              a.sel_indices.size(2),
                                              block_ids,
                                              nblocks);
            return;
        }
    }
#endif
    attention_selected_blocks_simd<T>(w, block_ids, nblocks);
}

#ifdef OPENVINO_ARCH_X86_64
// ---- prefill with AMX matrix products over the whole selection of a work item ----

// The rows of the paged caches for the shared tile loop (sparse_brgemm_kernel.hpp): a run of rows is contiguous inside
// one physical page; plain caches are copied, quantized ones dequantized, into the tile.
struct PagedRowSource {
    const PackWork& w;

    [[nodiscard]] int64_t visible(int64_t kpos0, int64_t n) const {
        return visible_rows(w, kpos0, n);
    }

    template <typename T>
    void gather(int64_t kpos0, size_t rows, T* k_dst, T* v_dst) const {
        const auto [phys_block, slot] = physical_address(w, kpos0);
        if (w.quant.enabled()) {
            dequant_k_rows<T>(w.quant, w.key_cache, phys_block, w.k_head, slot, rows, w.k_head_size, k_dst);
            dequant_v_rows<T>(w.quant, w.value_cache, phys_block, w.k_head, slot, rows, w.v_head_size, v_dst);
        } else {
            std::memcpy(k_dst, w.key_cache.ptr<T>(phys_block, w.k_head, slot), rows * w.k_head_size * sizeof(T));
            std::memcpy(v_dst, w.value_cache.ptr<T>(phys_block, w.k_head, slot), rows * w.v_head_size * sizeof(T));
        }
    }
};

// Attention of the pack's heads over the whole selection (blocks and causal tail) with AMX brgemm, M = the pack's
// heads, over gathered rows (see attention_over_selection_tiles). Leaves unnormalized attention in w.acc and the
// softmax denominators in w.running_sum; the caller normalizes and stores.
template <typename T>
void attention_full_selection_brgemm(const SparsePAArgs& a,
                                     PackWork& w,
                                     uint8_t* ws,
                                     const SparsePAScratchLayout& layout,
                                     const int32_t* block_ids,
                                     size_t nblocks) {
    const auto& lay = layout.brgemm;
    const BrgemmBuffers<T> b{static_cast<ov::intel_cpu::BrgemmKernel*>(a.brgemm.qk),
                             static_cast<ov::intel_cpu::BrgemmKernel*>(a.brgemm.pv_overwrite),
                             static_cast<ov::intel_cpu::BrgemmKernel*>(a.brgemm.pv_accumulate),
                             reinterpret_cast<T*>(ws + lay.k_rows),
                             reinterpret_cast<T*>(ws + lay.v_rows),
                             {reinterpret_cast<float*>(ws + lay.weights), a.brgemm.tile_rows},
                             ws + lay.brg_wsp,
                             ws + lay.brg_a,
                             ws + lay.brg_b};
    const BrgemmPack
        pack{w.query, w.pack_size, w.k_head_size, w.v_head_size, w.scale, w.acc, w.running_max, w.running_sum};
    const int64_t tail_start = ((w.token_pos + 1) / w.key_block_size) * w.key_block_size;
    const BrgemmSelection selection{block_ids,
                                    nblocks,
                                    0,
                                    w.key_block_size,
                                    w.page_size / w.key_block_size,
                                    std::max<int64_t>(w.max_run_rows / w.key_block_size, 1),
                                    tail_start,
                                    w.token_pos - tail_start + 1};
    attention_over_selection_tiles<T>(PagedRowSource{w}, pack, b, selection);
}
#endif  // OPENVINO_ARCH_X86_64

// One work item: attention of one query token and one head pack, on the brgemm path (whole selection at once) or the
// legacy path (selected blocks, then the causal tail). A legacy "chunk" is 1 or 4 GEMV row units, not a brgemm block.
template <typename T>
void attention_work_item(const SparsePAArgs& a,
                         const SparsePAScratchLayout& layout,
                         size_t token,
                         size_t pack,
                         size_t pack_size) {
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
    const size_t first_head = pack * pack_size;
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
               pack_size,
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
               reinterpret_cast<float*>(ws + layout.pack.acc),
               reinterpret_cast<float*>(ws + layout.pack.running_max),
               reinterpret_cast<float*>(ws + layout.pack.running_sum),
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
    OPENVINO_ASSERT(nblocks <= a.sel_indices.size(2),
                    "[CPU] sparse_pa: block_count exceeds the width of block_indices");
    bool done = false;
#ifdef OPENVINO_ARCH_X86_64
    // Prefill (more than one new token) of a pack of query heads with 16-bit data on AMX: matrix products over the
    // whole selection. Decode and single-head packs keep the run-by-run paths below.
    if constexpr (!std::is_same_v<T, float>) {
        if (a.brgemm.enabled() && pack_size > 1 && a.causal && token_end - token_begin > 1) {
            attention_full_selection_brgemm<T>(a, w, ws, layout, block_ids, nblocks);
            done = true;
        }
    }
#endif
    if (!done) {
        attention_selected_blocks_legacy<T>(a,
                                            w,
                                            ws,
                                            layout,
                                            token,
                                            first_head,
                                            token_end - token_begin,
                                            block_ids,
                                            nblocks);
        attention_causal_tail_legacy<T>(w);
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

// Sparse attention of every (global query token, query head pack) item of the work list, in parallel: the brgemm items
// first, then the fallback items.
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
                                       sparse_brgemm_sizes(a.brgemm));
    OPENVINO_ASSERT(layout.total <= a.scratch_bytes_per_thread, "[CPU] sparse_pa: scratch is too small");
    const auto& work = a.work;
    const size_t brgemm_per_token = a.q_head_num / work.brgemm_pack;
    const size_t fallback_per_token = a.q_head_num / work.fallback_pack;
    const size_t brgemm_items = work.brgemm_tokens * brgemm_per_token;
    cpu_parallel->parallel_for(brgemm_items + work.fallback_tokens * fallback_per_token, [&](size_t item) {
        if (item < brgemm_items) {
            attention_work_item<T>(a,
                                   layout,
                                   work.token_ids[item / brgemm_per_token],
                                   item % brgemm_per_token,
                                   work.brgemm_pack);
        } else {
            const size_t fallback_item = item - brgemm_items;
            attention_work_item<T>(a,
                                   layout,
                                   work.token_ids[work.brgemm_tokens + fallback_item / fallback_per_token],
                                   fallback_item % fallback_per_token,
                                   work.fallback_pack);
        }
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
                       const SparseBrgemm& brgemm,
                       uint8_t* scratch,
                       size_t scratch_bytes_per_thread,
                       int32_t* token_ids,
                       const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    const auto data_prc = query.get_precision();
    OPENVINO_ASSERT(out.get_precision() == data_prc, "[CPU] sparse_pa: query/output precisions must match");
    OPENVINO_ASSERT(
        quant.enabled() || (key_cache.get_precision() == data_prc && value_cache.get_precision() == data_prc),
        "[CPU] sparse_pa: plain caches must have the query precision");
    OPENVINO_ASSERT(scratch != nullptr && token_ids != nullptr, "[CPU] sparse_pa: scratch must not be null");
    OPENVINO_ASSERT(subsequence_begins.size(0) >= 1, "[CPU] sparse_pa: subsequence_begins must be non-empty");

    // The work list: the tokens of sequences with several new tokens first (brgemm items when the node built the
    // kernels), then the rest (fallback items); both id lists share the node's `token_ids` buffer (one id per token).
    const size_t max_pack_size = max_q_head_pack_size(q_head_num, k_head_num, sel_head_num);
    const auto* begins = subsequence_begins.ptr<int32_t>();
    const size_t sequences = subsequence_begins.size(0) - 1;
    const size_t tokens = query.size(0);
    const auto in_range = [tokens](int32_t token) {
        return static_cast<size_t>(token) < tokens;
    };
    WorkList work;
    work.token_ids = token_ids;
    for (size_t seq = 0; seq < sequences; seq++) {
        for (int32_t token = begins[seq]; token < begins[seq + 1] && in_range(token); token++) {
            if (brgemm.enabled() && begins[seq + 1] - begins[seq] > 1) {
                token_ids[work.brgemm_tokens++] = token;
            }
        }
    }
    for (size_t seq = 0; seq < sequences; seq++) {
        if (!brgemm.enabled() || begins[seq + 1] - begins[seq] <= 1) {
            for (int32_t token = begins[seq]; token < begins[seq + 1] && in_range(token); token++) {
                token_ids[work.brgemm_tokens + work.fallback_tokens++] = token;
            }
        }
    }
    work.brgemm_pack = max_pack_size;
    const size_t fallback_items = work.fallback_tokens * (q_head_num / max_pack_size);
    work.fallback_pack =
        fallback_items >= 2 * static_cast<size_t>(cpu_parallel->get_num_worker_threads()) ? max_pack_size : 1;

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
                            work};

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
