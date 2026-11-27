// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "nodes/kernels/sparse_attn/sparse_pa_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

#include "cpu_parallel.hpp"
#include "nodes/kernels/scaled_attn/attn_quant_kernel.hpp"
#include "nodes/kernels/scaled_attn/common.hpp"
#include "nodes/kernels/scaled_attn/executor_pa_common.hpp"
#include "nodes/kernels/scaled_attn/paged_attn_kernel.hpp"
#include "nodes/kernels/scaled_attn/softmax_kernel.hpp"
#include "nodes/kernels/sparse_attn/sparse_attn_kernel.hpp"
#include "nodes/kernels/sparse_attn/sparse_brgemm.hpp"
#include "nodes/kernels/sparse_attn/sparse_brgemm_kernel.hpp"
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
// Brgemm tokens (sequences of several new tokens, and decode tokens with a large selection) take the AMX head-pack
// path, the others SIMD items; `token_ids` lists brgemm tokens first. With little SIMD work every query head is its
// own item (`fallback_pack` 1).
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

// ---- The phases of attention over a run of key rows, for the heads of the pack (one head is a pack of one) ----
//   1. qk_scores_gqa:     scores   = query . keys^T
//   2. online_softmax:    weights  = softmax(scale * scores), rescaling the running (max, sum, acc)
//   3. pv_accumulate_gqa: acc     += weights . values
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

// Plain caches: attention over up to `n` consecutive logical positions [kpos0, kpos0 + n), all inside one physical
// page, for every query head of the pack: no per-row conversion and one page-table lookup.
template <typename T>
void attention_over_run_plain(PackWork& w, int64_t kpos0, int64_t n) {
    const auto [phys_block, slot] = physical_address(w, kpos0);
    const auto rows = static_cast<size_t>(n);
    T* key_rows = w.key_cache.ptr<T>(phys_block, w.k_head, slot);
    T* value_rows = w.value_cache.ptr<T>(phys_block, w.k_head, slot);
    const auto score_stride = static_cast<size_t>(w.max_run_rows);
    qk_scores_gqa<T>(w.q, w.pack_size, w.k_head_size, key_rows, rows, w.run_scores, score_stride);
    for (size_t head = 0; head < w.pack_size; head++) {
        online_softmax(w, head, rows);
    }
    pv_accumulate_gqa<T>(w.acc, w.pack_size, w.v_head_size, w.run_scores, score_stride, value_rows, rows);
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

// Quantized caches: attention over a run of rows inside one physical page. A fused phase (one-head pack) reads the
// quantized rows in place, the other one uses the dequantized f32 run through the head-pack kernels.
void attention_over_run_quant(PackWork& w, int64_t kpos0, int64_t n) {
    const auto [phys_block, slot] = physical_address(w, kpos0);
    const auto rows = static_cast<size_t>(n);
    const bool fused_keys = w.pack_size == 1 && !w.quant.key_bychannel;
    const bool fused_values = w.pack_size == 1;
    const auto score_stride = static_cast<size_t>(w.max_run_rows);
    dequantize_run(w, phys_block, slot, rows, fused_keys, fused_values);
    if (fused_keys) {
        dot_key_rows_by_token(w.quant, w.key_cache, phys_block, w.k_head, slot, rows, w.k_head_size, w.q, w.run_scores);
    } else {
        qk_scores_gqa<float>(w.q, w.pack_size, w.k_head_size, w.k_run, rows, w.run_scores, score_stride);
    }
    for (size_t head = 0; head < w.pack_size; head++) {
        online_softmax(w, head, rows);
    }
    if (fused_values) {
        acc_value_rows_by_token(w.quant,
                                w.value_cache,
                                phys_block,
                                w.k_head,
                                slot,
                                rows,
                                w.v_head_size,
                                w.run_scores,
                                w.acc);
    } else {
        pv_accumulate_gqa<float>(w.acc, w.pack_size, w.v_head_size, w.run_scores, score_stride, w.v_run, rows);
    }
}

// Number of rows of the run [kpos0, kpos0 + n) that the query may see.
int64_t visible_rows(const PackWork& w, int64_t kpos0, int64_t n) {
    const int64_t limit = w.causal ? std::min(w.valid_max, w.token_pos + 1) : w.valid_max;
    return kpos0 < 0 ? 0 : std::max<int64_t>(std::min(n, limit - kpos0), 0);
}

// Attention over up to `n` consecutive logical positions [kpos0, kpos0 + n) that lie inside one physical page, clamped
// to the positions the query may see.
void attention_over_run(PackWork& w, int64_t kpos0, int64_t n) {
    n = visible_rows(w, kpos0, n);
    if (n == 0) {
        return;
    }
    if (w.quant.enabled()) {
        attention_over_run_quant(w, kpos0, n);
    } else {
        // The query rows are f32 here, so only the cache rows need their own type (f16 next to f32 data, like
        // PagedAttention; the rows are converted in registers).
        visit_float_type(w.key_cache.get_precision(), [&](auto cache_tag) {
            attention_over_run_plain<decltype(cache_tag)>(w, kpos0, n);
        });
    }
}

// Attention over the selected complete blocks: adjacent blocks of the same physical page form one run (up to
// max_run_rows rows), so the block kernels see as many rows per call as possible.
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
        attention_over_run(w, first * w.key_block_size, run_blocks * w.key_block_size);
        i += static_cast<size_t>(run_blocks);
    }
}

// sel_indices/sel_count only ever carry *complete* selected blocks -- the producer never encodes the current
// incomplete trailing block. When causal, the consumer derives and attends that tail itself from the query's own
// logical position; it is empty at a block boundary.
void attention_causal_tail_simd(PackWork& w) {
    if (w.causal) {
        const int64_t tail_start = ((w.token_pos + 1) / w.key_block_size) * w.key_block_size;
        attention_over_run(w, tail_start, w.token_pos - tail_start + 1);
    }
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
    void rows(int64_t kpos0, size_t n, const T** k_ptr, const T** v_ptr, T* k_stage, T* v_stage) const {
        const auto [phys_block, slot] = physical_address(w, kpos0);
        const T* k_src = k_stage;
        const T* v_src = v_stage;
        if (w.quant.enabled()) {
            dequant_k_rows<T>(w.quant, w.key_cache, phys_block, w.k_head, slot, n, w.k_head_size, k_stage);
            dequant_v_rows<T>(w.quant, w.value_cache, phys_block, w.k_head, slot, n, w.v_head_size, v_stage);
        } else {
            k_src = w.key_cache.ptr<T>(phys_block, w.k_head, slot);
            v_src = w.value_cache.ptr<T>(phys_block, w.k_head, slot);
        }
        for (size_t i = 0; i < n; i++) {
            k_ptr[i] = k_src + i * w.k_head_size;
            v_ptr[i] = v_src + i * w.v_head_size;
        }
    }
};

// Attention of the pack's heads over the whole selection (blocks and causal tail) with AMX brgemm, M = the pack's
// heads, over gathered rows (see attention_over_selection_tiles). Leaves unnormalized attention in w.acc and the
// softmax denominators in w.running_sum; the caller normalizes and stores.
template <typename T>
void attention_full_selection_brgemm(
    const SparsePAArgs& a,
    PackWork& w,
    uint8_t* ws,  // NOLINT(readability-non-const-parameter): the buffers it carves are written
    const SparsePAScratchLayout& layout,
    const int32_t* block_ids,
    size_t nblocks) {
    const auto& lay = layout.brgemm;
    const BrgemmBuffers<T> b{static_cast<ov::intel_cpu::BrgemmKernel*>(a.brgemm.qk),
                             static_cast<ov::intel_cpu::BrgemmKernel*>(a.brgemm.pv_overwrite),
                             static_cast<ov::intel_cpu::BrgemmKernel*>(a.brgemm.pv_accumulate),
                             reinterpret_cast<T*>(ws + lay.k_rows),
                             reinterpret_cast<T*>(ws + lay.v_rows),
                             reinterpret_cast<const T**>(ws + lay.k_ptrs),
                             reinterpret_cast<const T**>(ws + lay.v_ptrs),
                             {reinterpret_cast<float*>(ws + lay.weights), a.brgemm.tile_rows},
                             ws + lay.brg_wsp,
                             ws + lay.brg_a,
                             ws + lay.brg_b};
    const BrgemmPack
        pack{w.query, w.pack_size, w.k_head_size, w.v_head_size, w.scale, w.acc, w.running_max, w.running_sum};
    const int64_t tail_start = ((w.token_pos + 1) / w.key_block_size) * w.key_block_size;
    const BrgemmSelection selection{block_ids, nblocks, 0, w.key_block_size, tail_start, w.token_pos - tail_start + 1};
    attention_over_selection_tiles<T>(PagedRowSource{w}, pack, b, selection);
}
#endif  // OPENVINO_ARCH_X86_64

// One work item: attention of one query token and one head pack, on the brgemm path (whole selection at once) or the
// SIMD path (selected blocks run by run, then the causal tail).
template <typename T>
void attention_work_item(const SparsePAArgs& a,
                         const SparsePAScratchLayout& layout,
                         size_t token,
                         size_t pack,
                         size_t pack_size,
                         bool brgemm_item) {
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
    // The brgemm items (prefill, or a decode with enough selected bytes, see kBrgemmDecodeMinBytes) of a pack of query
    // heads with 16-bit data on AMX: matrix products over the whole selection.
    if constexpr (!std::is_same_v<T, float>) {
        if (brgemm_item && a.brgemm.enabled() && pack_size > 1 && a.causal) {
            attention_full_selection_brgemm<T>(a, w, ws, layout, block_ids, nblocks);
            done = true;
        }
    }
#endif
    if (!done) {
        attention_selected_blocks_simd(w, block_ids, nblocks);
        attention_causal_tail_simd(w);
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
                                       a.max_run_rows,
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
                                   work.brgemm_pack,
                                   true);
        } else {
            const size_t fallback_item = item - brgemm_items;
            attention_work_item<T>(a,
                                   layout,
                                   work.token_ids[work.brgemm_tokens + fallback_item / fallback_per_token],
                                   fallback_item % fallback_per_token,
                                   work.fallback_pack,
                                   false);
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
                       size_t max_run_rows,
                       const SparseBrgemm& brgemm,
                       uint8_t* scratch,
                       size_t scratch_bytes_per_thread,
                       int32_t* token_ids,
                       const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    const auto data_prc = query.get_precision();
    OPENVINO_ASSERT(out.get_precision() == data_prc, "[CPU] sparse_pa: query/output precisions must match");
    // Plain caches: the query precision, or f16 next to f32 data like PagedAttention (read in place, no conversion
    // pass). The AMX tiles are built for the query precision, so a cache of another type uses the SIMD heads.
    const bool cache_is_data = key_cache.get_precision() == data_prc && value_cache.get_precision() == data_prc;
    OPENVINO_ASSERT(quant.enabled() || cache_is_data ||
                        (data_prc == ov::element::f32 && key_cache.get_precision() == ov::element::f16 &&
                         value_cache.get_precision() == ov::element::f16),
                    "[CPU] sparse_pa: plain caches must have the query precision, or f16 with f32 data");
    const bool tiles = brgemm.enabled() && (quant.enabled() || cache_is_data);
    OPENVINO_ASSERT(!brgemm.enabled() || brgemm.prec == data_prc,
                    "[CPU] sparse_pa: the brgemm kernels must be built for the query precision");
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
    // A decode token (a sequence of one new token) is a brgemm item too when its selection holds enough K / V bytes
    // (plain caches; a quantized cache would be dequantized into the tile first).
    const bool decode_brgemm =
        tiles && !quant.enabled() &&
        sel_indices.size(2) * key_block_size * (k_head_size + v_head_size) * brgemm.prec.size() >=
            kBrgemmDecodeMinBytes;
    const auto is_brgemm_sequence = [&](size_t seq) {
        return tiles && (begins[seq + 1] - begins[seq] > 1 || decode_brgemm);
    };
    WorkList work;
    work.token_ids = token_ids;
    for (size_t seq = 0; seq < sequences; seq++) {
        for (int32_t token = begins[seq]; token < begins[seq + 1] && in_range(token); token++) {
            if (is_brgemm_sequence(seq)) {
                token_ids[work.brgemm_tokens++] = token;
            }
        }
    }
    for (size_t seq = 0; seq < sequences; seq++) {
        if (!is_brgemm_sequence(seq)) {
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
