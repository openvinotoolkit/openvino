// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "nodes/kernels/sparse_attn/sparse_pa_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>

#include "cpu_parallel.hpp"
#include "nodes/kernels/scaled_attn/attn_brgemm_common.hpp"
#include "nodes/kernels/scaled_attn/attn_memcpy.hpp"
#include "nodes/kernels/scaled_attn/attn_quant.hpp"
#include "nodes/kernels/scaled_attn/attn_quant_kernel.hpp"
#include "nodes/kernels/scaled_attn/cache_spec.hpp"
#include "nodes/kernels/scaled_attn/common.hpp"
#include "nodes/kernels/scaled_attn/executor_pa_common.hpp"
#include "nodes/kernels/scaled_attn/paged_attn_kernel.hpp"
#include "nodes/kernels/scaled_attn/softmax_kernel.hpp"
#include "nodes/kernels/scaled_attn/visit_float_type.hpp"
#include "nodes/kernels/sparse_attn/sparse_attn_common.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

using ov::intel_cpu::CacheSpec;
using ov::intel_cpu::is_quantized_cache;
using ov::intel_cpu::MemoryPtr;
using ov::intel_cpu::PlainTensor;

namespace {

// The cache formats of the node, with the same layouts as PagedAttention (see executor_pa.cpp):
//  - by-token (key or value): each row is a sequence of groups `[scale f32][zp f32][group data]`, so the row holds
//    `S + 8 * groups` elements (u8) or `S + 16 * groups` (u4); a group size of 0 means one group per row.
//  - by-channel (key only): each page is `f32 scale[S] | f32 zp[S] | data[block_size, S]`.
// The caches are either both quantized or both plain f32/f16/bf16. The executor writes the current-step K/V into the
// caches before the attention, so the attention only reads them.

/// @brief Byte layout of the per-thread scratch of the sparse PA kernel. The scratch of a call holds one such block per
/// worker thread (so nothing is allocated while computing), carved up with this layout. Every part is 64-byte aligned.
/// Sizes are for the largest query-head pack, max_q_head_pack_size().
struct SparsePAScratchLayout {
    // Shared by every path: the state of the head pack (accumulators, running softmax maximum / sum).
    PackStateLayout pack;

    // SIMD path (run by run).
    size_t q = 0;       ///< float[pack, k_head_size]: query rows of the pack, f32
    size_t scores = 0;  ///< float[pack, max_run_rows]: scores, then weights, of a run of key rows
    size_t k_run = 0;   ///< quantized caches: float[max_run_rows, k_head_size] dequantized key run
    size_t v_run = 0;   ///< quantized caches, pack > 1: float[max_run_rows, v_head_size] dequantized value run

    // Brgemm path: the gather tile, the scores / weights and the brgemm scratch.
    TileBrgemmLayout brgemm;

    size_t total = 0;  ///< bytes per thread

    /// @param max_run_rows most key rows of one run: the page size when adjacent selected blocks are merged inside a
    /// page, key_block_size when they are not
    /// @param max_pack_size max_q_head_pack_size()
    /// @param data_size size in bytes of the query/key/value data type
    SparsePAScratchLayout(size_t k_head_size,
                          size_t v_head_size,
                          size_t max_run_rows,
                          size_t max_pack_size,
                          size_t data_size,
                          const CacheSpec& key_spec,
                          const TileBrgemmSizes& brgemm_sizes = {})
        : pack(0, k_head_size, v_head_size, max_pack_size, data_size, false),
          q(pack.total),
          scores(q + aligned(max_pack_size * k_head_size * sizeof(float))),
          k_run(scores + aligned(max_pack_size * max_run_rows * sizeof(float))),
          // quantized keys are dequantized to f32 for by-channel keys and for packs, values for packs
          v_run(
              k_run +
              aligned((is_quantized_cache(key_spec) && (key_spec.by_channel || max_pack_size > 1) ? max_run_rows : 0) *
                      k_head_size * sizeof(float))),
          brgemm(v_run + aligned((is_quantized_cache(key_spec) && max_pack_size > 1 ? max_run_rows : 0) * v_head_size *
                                 sizeof(float)),
                 k_head_size,
                 v_head_size,
                 max_pack_size,
                 data_size,
                 brgemm_sizes),
          total(brgemm.total) {}
};

// ---- quantized (u8/u4) paged cache access, same layouts as PagedAttention (see CacheSpec) ----
// Each function picks u8 or u4 at runtime (visit_quant_precision) and runs the kernel of that precision.

// Dequantizes `n` consecutive rows [slot0, slot0 + n) of one (physical block, kv head) of the key cache into dst[n, S]:
// by-token rows carry their own groups, a by-channel page holds the per-channel scales and zero points of all its rows.
template <typename TDST>
void dequant_k_rows(const SparsePAConfig& cfg,
                    const PlainTensor& cache,
                    size_t phys_block,
                    size_t kv,
                    size_t slot0,
                    size_t n,
                    size_t S,
                    TDST* dst) {
    visit_quant_precision(cfg.key_spec.precision, [&](auto prec) {
        constexpr ov::element::Type_t PREC = decltype(prec)::value;
        if (cfg.key_spec.by_channel) {
            attn_dequant_page_rows_by_channel<TDST, PREC>(cache.ptr<uint8_t, PREC>(phys_block, kv),
                                                          dst,
                                                          slot0,
                                                          n,
                                                          S,
                                                          S);
        } else {
            attn_dequant_rows_by_token<TDST, PREC>(cache.ptr<uint8_t, PREC>(phys_block, kv, slot0),
                                                   dst,
                                                   n,
                                                   S,
                                                   cfg.key_spec.group_size,
                                                   S);
        }
    });
}

// Dequantizes `n` consecutive by-token value rows [slot0, slot0 + n) of one (physical block, kv head) into dst[n, Sv].
template <typename TDST>
void dequant_v_rows(const SparsePAConfig& cfg,
                    const PlainTensor& cache,
                    size_t phys_block,
                    size_t kv,
                    size_t slot0,
                    size_t n,
                    size_t Sv,
                    TDST* dst) {
    visit_quant_precision(cfg.value_spec.precision, [&](auto prec) {
        constexpr ov::element::Type_t PREC = decltype(prec)::value;
        attn_dequant_rows_by_token<TDST, PREC>(cache.ptr<uint8_t, PREC>(phys_block, kv, slot0),
                                               dst,
                                               n,
                                               Sv,
                                               cfg.value_spec.group_size,
                                               Sv);
    });
}

// scores[j] = dot(q, key row slot0 + j) for by-token key rows, straight from the quantized cache.
void dot_key_rows_by_token(const SparsePAConfig& cfg,
                           const PlainTensor& cache,
                           size_t phys_block,
                           size_t kv,
                           size_t slot0,
                           size_t rows,
                           size_t S,
                           float* q,
                           float* scores) {
    visit_quant_precision(cfg.key_spec.precision, [&](auto prec) {
        constexpr ov::element::Type_t PREC = decltype(prec)::value;
        auto* k = cache.ptr<uint8_t, PREC>(phys_block, kv, slot0);
        const size_t gs = cfg.key_spec.group_size ? cfg.key_spec.group_size : S;
        dot_product_block_quantized<float, PREC>(q, k, scores, S, false, rows, gs);
    });
}

// out += sum_j weights[j] * (value row slot0 + j), straight from the quantized cache (value is always by-token).
void acc_value_rows_by_token(const SparsePAConfig& cfg,
                             const PlainTensor& cache,
                             size_t phys_block,
                             size_t kv,
                             size_t slot0,
                             size_t rows,
                             size_t Sv,
                             float* weights,
                             float* out) {
    visit_quant_precision(cfg.value_spec.precision, [&](auto prec) {
        constexpr ov::element::Type_t PREC = decltype(prec)::value;
        auto* v = cache.ptr<uint8_t, PREC>(phys_block, kv, slot0);
        const size_t gs = cfg.value_spec.group_size ? cfg.value_spec.group_size : Sv;
        attn_acc_value_block_quantized<float, PREC>(out, weights, v, Sv, false, rows, gs);
    });
}

// ---- the items of a call ----
// Brgemm tokens (the tokens of sequences of several new tokens) take the AMX head-pack path, the others SIMD items;
// `token_ids` lists brgemm tokens first. With little SIMD work every query head is its own item (`fallback_pack` 1).
struct WorkList {
    const int32_t* token_ids = nullptr;
    size_t brgemm_tokens = 0;
    size_t fallback_tokens = 0;
    size_t brgemm_pack = 1;    ///< query heads of one item of a brgemm token
    size_t fallback_pack = 1;  ///< query heads of one item of a fallback token
    size_t kv_chunks = 1;      ///< slices of the selected blocks of a fallback item (a decode of few items is split)
};

// A compressed cache is dequantized once per call for prefill sequences when their tokens gather at least as many rows
// as their pages hold, so a row many tokens select is not dequantized per token. Returns the block_indices entries
// covered (0: per-row path).
size_t dequantized_cache_pages(const SparsePATensors& t, const SparsePAConfig& cfg) {
    if (!cfg.brgemm.handle().enabled() || !cfg.causal || !is_quantized_cache(cfg.key_spec) ||
        t.query.get_precision() != cfg.brgemm.handle().prec) {
        return 0;
    }
    const auto* begins = t.subsequence_begins.ptr<int32_t>();
    const auto* blocks = t.block_indices_begins.ptr<int32_t>();
    const size_t sequences = t.subsequence_begins.size(0) - 1;
    size_t tokens = 0;
    size_t pages = 0;
    for (size_t seq = 0; seq < sequences; seq++) {
        const auto new_tokens = static_cast<size_t>(begins[seq + 1] - begins[seq]);
        if (new_tokens > 1) {
            tokens += new_tokens;
            pages += static_cast<size_t>(blocks[seq + 1] - blocks[seq]);
        }
    }
    const size_t page_size = cfg.page_size ? cfg.page_size : t.key_cache.size(2);
    return pages > 0 && tokens * t.sel_indices.size(2) * cfg.key_block_size >= pages * page_size
               ? static_cast<size_t>(blocks[sequences])
               : 0;
}

// Where the dequantized pages of `entries` block_indices entries lie in their region of the scratch: the identity page
// table, then the key rows and the value rows ([entries, Hkv, page_size, head size] each).
struct DequantizedLayout {
    size_t table = 0;  ///< byte offsets in the region
    size_t key = 0;
    size_t value = 0;
    size_t bytes = 0;

    DequantizedLayout(size_t entries, const SparsePATensors& t, const SparsePAConfig& cfg) {
        const size_t page_size = cfg.page_size ? cfg.page_size : t.key_cache.size(2);
        const size_t page_rows = entries * cfg.num_kv_heads * page_size;
        ScratchPartition region;
        table = region.reserve(entries * sizeof(int32_t));
        key = region.reserve(page_rows * cfg.k_head_size * cfg.data_precision.size());
        value = region.reserve(page_rows * cfg.v_head_size * cfg.data_precision.size());
        bytes = region.bytes();
    }
};

/// @brief The scratch of a call, laid out in one place so that its size and the places of its parts cannot disagree:
///   | one block per worker thread | one token id per token | partial states OR dequantized pages |
/// A call has partial states only when it splits a decode (no brgemm token) and dequantized pages only for a prefill
/// (brgemm tokens), never both, so they share the memory. All the offsets are in bytes from the start of the scratch.
struct SparsePAScratch {
    size_t threads = 0;  ///< num_threads blocks of thread_bytes, see SparsePAScratchLayout
    size_t thread_bytes = 0;
    size_t token_ids = 0;  ///< int32[tokens]
    size_t partials =
        0;  ///< float[num_threads, max_pack * (v_head_size + 2)]: the partial softmax states of a split decode
    size_t dequantized = 0;          ///< the pages of DequantizedLayout
    size_t dequantized_entries = 0;  ///< block_indices entries they cover, 0: none
    DequantizedLayout dequantized_layout;
    size_t total = 0;

    SparsePAScratch(const SparsePAConfig& cfg, const SparsePATensors& t, size_t num_threads, size_t block_bytes)
        : thread_bytes(block_bytes),
          dequantized_entries(dequantized_cache_pages(t, cfg)),
          dequantized_layout(dequantized_entries, t, cfg) {
        const size_t max_pack = max_q_head_pack_size(cfg.num_heads, cfg.num_kv_heads, cfg.num_selection_heads);
        ScratchPartition scratch;
        threads = scratch.reserve(num_threads * thread_bytes);
        token_ids = scratch.reserve(t.query.size(0) * sizeof(int32_t));
        partials = dequantized = scratch.reserve(
            std::max(num_threads * max_pack * (cfg.v_head_size + 2) * sizeof(float), dequantized_layout.bytes));
        total = scratch.bytes();
    }
};

// ---- everything the sparse attention work items read from the arguments, see SparsePAExecutor ----

struct SparsePAArgs {
    const SparsePAConfig& cfg;  // fixed with the executor: head sizes, key_block_size, causal, specs, brgemm
    const SparsePATensors& t;   // the tensors of the call: shapes give the page size, the heads of the selection ...
    WorkList work;
    const MemoryPtr& scratch_memory;  // the scratch pad memory of the call: the owner of its parts
    uint8_t* scratch;                 // its data
    const SparsePAScratch& layout;    // where the parts of the scratch are
};

// State of one work item: a query token and a pack of `pack_size` consecutive query heads sharing K/V rows and selected
// blocks. Holds the sequence's cache location, the pack's scratch and each head's running online-softmax state.
struct PackWork {
    const SparsePAConfig& cfg;
    const SparsePATensors& t;
    const int32_t* block_table;  // physical block ids of the sequence
    size_t k_head;               // the K/V head every query head of the pack reads
    size_t pack_size;            // query heads of the pack
    int64_t token_pos;           // logical position of the query token
    int64_t valid_max;           // exclusive bound on the valid logical positions of the sequence
    const void* query;           // [pack_size, k_head_size] query rows in the data type of the query
    float* q;                    // the same rows in f32
    float* acc;                  // [pack_size, v_head_size] attention accumulators
    float* running_max;
    float* running_sum;
    float* run_scores;  // [pack_size, max_run_rows] scores, then weights, of a run of key rows
    float* k_run;       // quantized caches: [max_run_rows, k_head_size] dequantized key run
    float* v_run;       // quantized caches: [max_run_rows, v_head_size] dequantized value run
    // Tokens per physical cache block: the config holds it for quantized caches (by-channel key pages carry extra
    // parameter rows in dim 2 of the cache); a plain cache has it in dim 2.
    int64_t page_size;
    int64_t key_block_size;

    /// @param ws the thread's scratch block, laid out by `layout`
    PackWork(const SparsePAArgs& args,
             const SparsePAScratchLayout& layout,
             uint8_t* ws,
             const int32_t* table,
             size_t head,
             size_t pack,
             int64_t pos,
             int64_t valid,
             const void* query_rows)
        : cfg(args.cfg),
          t(args.t),
          block_table(table),
          k_head(head),
          pack_size(pack),
          token_pos(pos),
          valid_max(valid),
          query(query_rows),
          q(reinterpret_cast<float*>(ws + layout.q)),
          acc(reinterpret_cast<float*>(ws + layout.pack.acc)),
          running_max(reinterpret_cast<float*>(ws + layout.pack.running_max)),
          running_sum(reinterpret_cast<float*>(ws + layout.pack.running_sum)),
          run_scores(reinterpret_cast<float*>(ws + layout.scores)),
          k_run(reinterpret_cast<float*>(ws + layout.k_run)),
          v_run(reinterpret_cast<float*>(ws + layout.v_run)),
          page_size(static_cast<int64_t>(cfg.page_size ? cfg.page_size : t.key_cache.size(2))),
          key_block_size(static_cast<int64_t>(cfg.key_block_size)) {}
};

// Maps a logical KV position of the sequence to the physical (block, slot) address in the caches through the shared
// page table. A selected key_block_size-wide span never crosses a physical block boundary (page_size % key_block_size
// == 0 is enforced by the node).
std::pair<size_t, size_t> physical_address(const PackWork& w, int64_t logical_pos) {
    return {static_cast<size_t>(w.block_table[logical_pos / w.page_size]),
            static_cast<size_t>(logical_pos % w.page_size)};
}

// ---- The phases of attention over a run of key rows, for the heads of the pack (one head is a pack of one) ----
//   1. dot_product_block_multi_head:    scores  = query . keys^T
//   2. online_softmax:                  weights = softmax(scale * scores), rescaling the running (max, sum, acc)
//   3. attn_acc_value_block_multi_head: acc    += weights . values
// attention_over_* run them for a run, the selected blocks or the causal tail.

// Phase 2: online-softmax step of query head `head` over its scores[0, rows) of one run of key rows (see
// online_softmax_step()).
void online_softmax(PackWork& w, size_t head, size_t rows) {
    online_softmax_step(w.run_scores + head * w.cfg.max_run_rows,
                        rows,
                        w.t.scale,
                        w.running_max[head],
                        w.running_sum[head],
                        w.acc + head * w.cfg.v_head_size,
                        w.cfg.v_head_size);
}

// Plain caches: attention over up to `n` consecutive logical positions [kpos0, kpos0 + n), all inside one physical
// page, for every query head of the pack: no per-row conversion and one page-table lookup.
template <typename T>
void attention_over_run_plain(PackWork& w, int64_t kpos0, int64_t n) {
    const auto [phys_block, slot] = physical_address(w, kpos0);
    const auto rows = static_cast<size_t>(n);
    T* key_rows = w.t.key_cache.ptr<T>(phys_block, w.k_head, slot);
    T* value_rows = w.t.value_cache.ptr<T>(phys_block, w.k_head, slot);
    const auto score_stride = w.cfg.max_run_rows;
    dot_product_block_multi_head<T>(w.q, w.pack_size, w.cfg.k_head_size, key_rows, rows, w.run_scores, score_stride);
    for (size_t head = 0; head < w.pack_size; head++) {
        online_softmax(w, head, rows);
    }
    attn_acc_value_block_multi_head<T>(w.acc,
                                       w.pack_size,
                                       w.cfg.v_head_size,
                                       w.run_scores,
                                       score_stride,
                                       value_rows,
                                       rows);
}

// Quantized caches. A one-head pack reads by-token rows directly with PagedAttention's fused block kernels (by-channel
// keys are dequantized first: those kernels walk a whole page). A multi-head pack dequantizes the run's K and V to f32
// once for all its heads.

// Phase 0: dequantize the rows a fused kernel cannot read in place.
void dequantize_run(PackWork& w, size_t phys_block, size_t slot, size_t rows, bool fused_keys, bool fused_values) {
    if (!fused_keys) {
        dequant_k_rows<float>(w.cfg, w.t.key_cache, phys_block, w.k_head, slot, rows, w.cfg.k_head_size, w.k_run);
    }
    if (!fused_values) {
        dequant_v_rows<float>(w.cfg, w.t.value_cache, phys_block, w.k_head, slot, rows, w.cfg.v_head_size, w.v_run);
    }
}

// Quantized caches: attention over a run of rows inside one physical page. A fused phase (one-head pack) reads the
// quantized rows in place, the other one uses the dequantized f32 run through the head-pack kernels.
void attention_over_run_quant(PackWork& w, int64_t kpos0, int64_t n) {
    const auto [phys_block, slot] = physical_address(w, kpos0);
    const auto rows = static_cast<size_t>(n);
    const bool fused_keys = w.pack_size == 1 && !w.cfg.key_spec.by_channel;
    const bool fused_values = w.pack_size == 1;
    const auto score_stride = w.cfg.max_run_rows;
    dequantize_run(w, phys_block, slot, rows, fused_keys, fused_values);
    if (fused_keys) {
        dot_key_rows_by_token(w.cfg,
                              w.t.key_cache,
                              phys_block,
                              w.k_head,
                              slot,
                              rows,
                              w.cfg.k_head_size,
                              w.q,
                              w.run_scores);
    } else {
        dot_product_block_multi_head<float>(w.q,
                                            w.pack_size,
                                            w.cfg.k_head_size,
                                            w.k_run,
                                            rows,
                                            w.run_scores,
                                            score_stride);
    }
    for (size_t head = 0; head < w.pack_size; head++) {
        online_softmax(w, head, rows);
    }
    if (fused_values) {
        acc_value_rows_by_token(w.cfg,
                                w.t.value_cache,
                                phys_block,
                                w.k_head,
                                slot,
                                rows,
                                w.cfg.v_head_size,
                                w.run_scores,
                                w.acc);
    } else {
        attn_acc_value_block_multi_head<float>(w.acc,
                                               w.pack_size,
                                               w.cfg.v_head_size,
                                               w.run_scores,
                                               score_stride,
                                               w.v_run,
                                               rows);
    }
}

// Number of rows of the run [kpos0, kpos0 + n) that the query may see.
int64_t visible_rows(const PackWork& w, int64_t kpos0, int64_t n) {
    const int64_t limit = w.cfg.causal ? std::min(w.valid_max, w.token_pos + 1) : w.valid_max;
    return XARCH::visible_rows(kpos0, n, 0, limit);
}

// Attention over up to `n` consecutive logical positions [kpos0, kpos0 + n) that lie inside one physical page, clamped
// to the positions the query may see.
void attention_over_run(PackWork& w, int64_t kpos0, int64_t n) {
    n = visible_rows(w, kpos0, n);
    if (n == 0) {
        return;
    }
    if (is_quantized_cache(w.cfg.key_spec)) {
        attention_over_run_quant(w, kpos0, n);
    } else {
        // The query rows are f32 here, so only the cache rows need their own type (f16 next to f32 data, like
        // PagedAttention; the rows are converted in registers).
        visit_float_type(w.t.key_cache.get_precision(), [&](auto cache_tag) {
            attention_over_run_plain<decltype(cache_tag)>(w, kpos0, n);
        });
    }
}

// Attention over the selected complete blocks: adjacent blocks of the same physical page form one run (up to
// max_run_rows rows), so the block kernels see as many rows per call as possible.
void attention_selected_blocks_simd(PackWork& w, const int32_t* block_ids, size_t nblocks) {
    const int64_t page_blocks = w.page_size / w.key_block_size;
    const int64_t max_run_blocks = std::max<int64_t>(static_cast<int64_t>(w.cfg.max_run_rows) / w.key_block_size, 1);
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
    if (w.cfg.causal) {
        const int64_t tail_start = causal_tail_start(w.token_pos, 0, w.key_block_size);
        attention_over_run(w, tail_start, w.token_pos - tail_start + 1);
    }
}

#if defined(OPENVINO_ARCH_X86_64) && defined(HAVE_AVX512F)
// ---- prefill with AMX matrix products over the whole selection of a work item ----

// The rows of the paged caches for the shared tile loop (attn_brgemm_tiles.hpp): a run of rows is contiguous inside
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
        if (is_quantized_cache(w.cfg.key_spec)) {
            dequant_k_rows<T>(w.cfg, w.t.key_cache, phys_block, w.k_head, slot, n, w.cfg.k_head_size, k_stage);
            dequant_v_rows<T>(w.cfg, w.t.value_cache, phys_block, w.k_head, slot, n, w.cfg.v_head_size, v_stage);
        } else {
            k_src = w.t.key_cache.ptr<T>(phys_block, w.k_head, slot);
            v_src = w.t.value_cache.ptr<T>(phys_block, w.k_head, slot);
        }
        for (size_t i = 0; i < n; i++) {
            k_ptr[i] = k_src + i * w.cfg.k_head_size;
            v_ptr[i] = v_src + i * w.cfg.v_head_size;
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
    const TileBrgemm& brgemm = a.cfg.brgemm.handle();
    const auto b = make_brgemm_buffers<T>(brgemm, lay, ws);
    const BrgemmPack pack{w.query,
                          w.pack_size,
                          w.cfg.k_head_size,
                          w.cfg.v_head_size,
                          w.t.scale,
                          w.acc,
                          w.running_max,
                          w.running_sum};
    const int64_t tail_start = causal_tail_start(w.token_pos, 0, w.key_block_size);
    const BrgemmSelection selection{block_ids, nblocks, 0, w.key_block_size, tail_start, w.token_pos - tail_start + 1};
    attention_over_selection_tiles<T>(PagedRowSource{w}, pack, b, selection);
}
#endif  // OPENVINO_ARCH_X86_64 && HAVE_AVX512F

// One work item: attention of one query token and one head pack, on the brgemm path (whole selection at once) or the
// SIMD path (the `kv_chunk`-th of `kv_chunks` slices of the selected blocks, run by run; the last slice also takes the
// causal tail). The result is written out, or left in `partial` as a softmax state when the blocks are split.
template <typename T>
void attention_work_item(const SparsePAArgs& a,
                         const SparsePAScratchLayout& layout,
                         size_t token,
                         size_t pack,
                         size_t pack_size,
                         [[maybe_unused]] bool brgemm_item,
                         size_t kv_chunk,
                         size_t kv_chunks,
                         float* partial) {
    const auto* subsequence_begins = a.t.subsequence_begins.ptr<int32_t>();
    // The sequence whose tokens contain `token`.
    const auto* const begins_end = subsequence_begins + a.t.subsequence_begins.size(0);
    const size_t seq =
        static_cast<size_t>(std::upper_bound(subsequence_begins, begins_end, static_cast<int32_t>(token)) -
                            subsequence_begins) -
        1;
    if (seq + 1 >= a.t.subsequence_begins.size(0)) {
        return;  // not a token of any sequence
    }
    const int32_t token_begin = subsequence_begins[seq];
    const int32_t token_end = subsequence_begins[seq + 1];
    const int32_t block_begin = a.t.block_indices_begins.ptr<int32_t>()[seq];
    OPENVINO_ASSERT(a.t.block_indices_begins.ptr<int32_t>()[seq + 1] > block_begin,
                    "[CPU] sparse_pa: each sequence must have at least one cache block");
    const int64_t past_len = a.t.past_lens.ptr<int32_t>()[seq];

    // The query heads of a pack share the K head and the selection head of the pack's first query head.
    const size_t first_head = pack * pack_size;
    const size_t k_head = first_head / (a.cfg.num_heads / a.t.key_cache.size(1));
    const size_t sel_head = first_head / (a.cfg.num_heads / a.t.sel_indices.size(1));

    // Per-thread scratch, see SparsePAScratchLayout.
    uint8_t* ws = a.scratch + a.layout.threads + static_cast<size_t>(parallel_get_thread_num()) * a.layout.thread_bytes;
    PackWork w(a,
               layout,
               ws,
               a.t.block_indices.ptr<int32_t>() + block_begin,
               k_head,
               pack_size,
               past_len + (static_cast<int32_t>(token) - token_begin),
               // Enforced unconditionally (not just under causal) so a non-causal gather never reads an unallocated
               // or stale slot in the physical pages.
               past_len + (token_end - token_begin),
               a.t.query.ptr<T>(token, first_head * a.cfg.k_head_size));
    cvt_copy(w.q, a.t.query.ptr<T>(token, first_head * a.cfg.k_head_size), 1, w.pack_size * a.cfg.k_head_size, 0, 0);
    init_pack_state(w.acc, w.running_max, w.running_sum, w.pack_size, a.cfg.v_head_size);

    // One query per selection row, so the row of a token is its global token index.
    const int32_t* block_ids = a.t.sel_indices.ptr<int32_t>(token, sel_head);
    const auto nblocks = static_cast<size_t>(*a.t.sel_count.ptr<int32_t>(token, sel_head));
    OPENVINO_ASSERT(nblocks <= a.t.sel_indices.size(2),
                    "[CPU] sparse_pa: block_count exceeds the width of block_indices");
    bool done = false;
#if defined(OPENVINO_ARCH_X86_64) && defined(HAVE_AVX512F)
    // The brgemm items (prefill) of a pack of query heads with 16-bit data on AMX: matrix products over the whole
    // selection.
    if constexpr (!std::is_same_v<T, float>) {
        if (brgemm_item) {
            attention_full_selection_brgemm<T>(a, w, ws, layout, block_ids, nblocks);
            done = true;
        }
    }
#endif
    if (!done) {
        // This item takes the blocks [first, last) of the selection; the last kv_chunk also takes the causal tail.
        const size_t first = nblocks * kv_chunk / kv_chunks;
        const size_t last = nblocks * (kv_chunk + 1) / kv_chunks;
        attention_selected_blocks_simd(w, block_ids + first, last - first);
        if (kv_chunk + 1 == kv_chunks) {
            attention_causal_tail_simd(w);
        }
    }

    if (partial != nullptr) {
        std::copy_n(w.running_max, w.pack_size, partial);
        std::copy_n(w.running_sum, w.pack_size, partial + w.pack_size);
        std::copy_n(w.acc, w.pack_size * a.cfg.v_head_size, partial + 2 * w.pack_size);
        return;
    }
    store_normalized_pack(w.acc, w.running_sum, w.pack_size, a.cfg.v_head_size, [&](size_t head) {
        return a.t.out.ptr<T>(token, (first_head + head) * a.cfg.v_head_size);
    });
}

// The pages of the compressed caches that a call dequantized: views of the scratch pad memory, which they share the
// ownership of.
struct DequantizedPages {
    PlainTensor key;  // [entries, Hkv, page_size, k_head_size] in the data type, page i = the page of block_indices[i]
    PlainTensor value;  // [entries, Hkv, page_size, v_head_size]
    PlainTensor table;  // [entries] identity page table: the pages are packed in block_indices order, so the page of
                        // entry i is i (the physical block ids of block_indices index the compressed cache, not these)
};

// Dequantizes the pages of prefill sequences into their region of the scratch (see SparsePAScratch) and sets the views
// of them in `pages`.
template <typename T>
void build_dequantized_pages(const SparsePAArgs& a,
                             DequantizedPages& pages,
                             const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    const auto& cfg = a.cfg;
    const auto& t = a.t;
    const auto& region = a.layout.dequantized_layout;
    const size_t entries = a.layout.dequantized_entries;
    const size_t k_head_num = t.key_cache.size(1);
    const size_t page_size = cfg.page_size ? cfg.page_size : t.key_cache.size(2);
    pages.table.resize_view<int32_t>({entries}, a.scratch_memory, a.layout.dequantized + region.table);
    pages.key.resize_view<T>({entries, k_head_num, page_size, cfg.k_head_size},
                             a.scratch_memory,
                             a.layout.dequantized + region.key);
    pages.value.resize_view<T>({entries, k_head_num, page_size, cfg.v_head_size},
                               a.scratch_memory,
                               a.layout.dequantized + region.value);
    auto* table = pages.table.ptr<int32_t>();
    std::iota(table, table + entries, 0);
    const auto* begins = t.subsequence_begins.ptr<int32_t>();
    const auto* blocks = t.block_indices_begins.ptr<int32_t>();
    const size_t sequences = t.subsequence_begins.size(0) - 1;
    cpu_parallel->parallel_for2d(entries, k_head_num, [&](size_t entry, size_t hk) {
        const int32_t phys_block = t.block_indices.ptr<int32_t>()[entry];
        const auto seq = static_cast<size_t>(
            std::upper_bound(blocks, blocks + sequences + 1, static_cast<int32_t>(entry)) - blocks - 1);
        if (phys_block < 0 || seq >= sequences || begins[seq + 1] - begins[seq] <= 1) {
            return;  // an unused page, or one of a sequence that does not use the dequantized pages
        }
        dequant_k_rows<T>(cfg,
                          t.key_cache,
                          static_cast<size_t>(phys_block),
                          hk,
                          0,
                          page_size,
                          cfg.k_head_size,
                          pages.key.ptr<T>(entry, hk));
        dequant_v_rows<T>(cfg,
                          t.value_cache,
                          static_cast<size_t>(phys_block),
                          hk,
                          0,
                          page_size,
                          cfg.v_head_size,
                          pages.value.ptr<T>(entry, hk));
    });
}

// Sparse attention of every (global query token, query head pack, kv_chunk) item of the work list, in parallel: the
// brgemm items first, then the fallback items. A split decode leaves partial softmax states in `partials` for a second
// merging pass; the brgemm items of a compressed cache read its dequantized pages.
template <typename T>
void sparse_pa_impl(const SparsePAArgs& a, const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    const size_t max_pack = max_q_head_pack_size(a.cfg.num_heads, a.cfg.num_kv_heads, a.cfg.num_selection_heads);
    const SparsePAScratchLayout layout(a.cfg.k_head_size,
                                       a.cfg.v_head_size,
                                       a.cfg.max_run_rows,
                                       max_pack,
                                       sizeof(T),
                                       a.cfg.key_spec,
                                       tile_brgemm_sizes(a.cfg.brgemm.handle()));
    OPENVINO_ASSERT(layout.total <= a.layout.thread_bytes, "[CPU] sparse_pa: scratch is too small");
    const auto& work = a.work;
    auto* const partials = reinterpret_cast<float*>(a.scratch + a.layout.partials);

    // The brgemm items read dequantized copies of the pages of a compressed cache, see build_dequantized_pages. They
    // share their memory with the partial states, which a call has only without brgemm tokens.
    DequantizedPages dequantized;
    bool use_dequantized = false;
    if constexpr (!std::is_same_v<T, float>) {
        if (a.layout.dequantized_entries > 0 && work.brgemm_tokens > 0) {
            build_dequantized_pages<T>(a, dequantized, cpu_parallel);
            use_dequantized = true;
        }
    }
    const SparsePATensors dequantized_t{a.t.query,
                                        a.t.key,
                                        a.t.value,
                                        dequantized.key,
                                        dequantized.value,
                                        dequantized.table,
                                        a.t.block_indices_begins,
                                        a.t.subsequence_begins,
                                        a.t.past_lens,
                                        a.t.sel_indices,
                                        a.t.sel_count,
                                        a.t.out,
                                        a.t.scale};
    SparsePAConfig dequantized_cfg = a.cfg;  // the dequantized pages are plain caches
    dequantized_cfg.key_spec = {};
    dequantized_cfg.value_spec = {};
    const SparsePAArgs dequantized_args{dequantized_cfg, dequantized_t, work, a.scratch_memory, a.scratch, a.layout};

    const size_t brgemm_per_token = a.cfg.num_heads / work.brgemm_pack;
    const size_t fallback_per_token = a.cfg.num_heads / work.fallback_pack;
    const size_t brgemm_items = work.brgemm_tokens * brgemm_per_token;
    const size_t partial_floats = work.fallback_pack * (a.cfg.v_head_size + 2);
    // Items are (token, pack, kv_chunk): the brgemm tokens first, each with packs of brgemm_pack heads and one chunk,
    // then the fallback tokens; a split decode leaves its state in the slot of the item.
    cpu_parallel->parallel_for(brgemm_items + work.fallback_tokens * fallback_per_token * work.kv_chunks,
                               [&](size_t item) {
                                   if (item < brgemm_items) {
                                       attention_work_item<T>(use_dequantized ? dequantized_args : a,
                                                              layout,
                                                              work.token_ids[item / brgemm_per_token],
                                                              item % brgemm_per_token,
                                                              work.brgemm_pack,
                                                              true,
                                                              0,
                                                              1,
                                                              nullptr);
                                   } else {
                                       const size_t local = item - brgemm_items;
                                       const size_t per_token = fallback_per_token * work.kv_chunks;
                                       attention_work_item<T>(
                                           a,
                                           layout,
                                           work.token_ids[work.brgemm_tokens + local / per_token],
                                           (local % per_token) / work.kv_chunks,
                                           work.fallback_pack,
                                           false,
                                           local % work.kv_chunks,
                                           work.kv_chunks,
                                           work.kv_chunks > 1 ? partials + local * partial_floats : nullptr);
                                   }
                               });
    if (work.kv_chunks == 1) {
        return;
    }
    // Second pass: merge the partial states of every fallback (token, pack) item into the output.
    cpu_parallel->parallel_for(work.fallback_tokens * fallback_per_token, [&](size_t token_pack) {
        const size_t token = work.token_ids[work.brgemm_tokens + token_pack / fallback_per_token];
        const size_t first_head = (token_pack % fallback_per_token) * work.fallback_pack;
        auto* const merged = reinterpret_cast<float*>(
            a.scratch + a.layout.threads + static_cast<size_t>(parallel_get_thread_num()) * a.layout.thread_bytes +
            layout.pack.acc);
        merge_decode_partials(partials + token_pack * work.kv_chunks * partial_floats,
                              work.kv_chunks,
                              work.fallback_pack,
                              a.cfg.v_head_size,
                              merged);
        for (size_t head = 0; head < work.fallback_pack; head++) {
            cvt_copy(a.t.out.ptr<T>(token, (first_head + head) * a.cfg.v_head_size),
                     merged + head * a.cfg.v_head_size,
                     1,
                     a.cfg.v_head_size,
                     0,
                     0);
        }
    });
}

class PaExecutor final : public SparsePAExecutor {
public:
    // The heads, head sizes and the data type are fixed by the config, so the scratch of a thread is computed once
    // here.
    explicit PaExecutor(const SparsePAConfig& config)
        : m_cfg(config),
          m_thread_block_bytes(
              SparsePAScratchLayout(
                  config.k_head_size,
                  config.v_head_size,
                  config.max_run_rows,
                  max_q_head_pack_size(config.num_heads, config.num_kv_heads, config.num_selection_heads),
                  config.data_precision.size(),
                  config.key_spec,
                  tile_brgemm_sizes(config.brgemm.handle()))
                  .total) {}

    [[nodiscard]] size_t scratch_bytes(const SparsePATensors& t,
                                       const ov::intel_cpu::CpuParallelPtr& cpu_parallel) const override {
        return SparsePAScratch(m_cfg,
                               t,
                               static_cast<size_t>(cpu_parallel->get_num_worker_threads()),
                               m_thread_block_bytes)
            .total;
    }

    void execute(const SparsePATensors& t,
                 const MemoryPtr& scratch,
                 const ov::intel_cpu::CpuParallelPtr& cpu_parallel) override {
        // Phase 1 completes before phase 2, the sparse attention, reads the caches.
        write_kv(t, cpu_parallel);
        run(t, scratch, cpu_parallel);
    }

private:
    // Phase 1: writes the current K/V into the paged caches (parallel over token x kv head), quantizing them for
    // u8 / u4 caches. The slot mapping and the by-channel requantization buffer are members, as in the PagedAttention
    // executor.
    void write_kv(const SparsePATensors& t, const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
        const auto& c = m_cfg;
        const auto& key_cache = t.key_cache;
        const auto& past_lens = t.past_lens;
        const auto& subsequence_begins = t.subsequence_begins;
        const auto& block_indices = t.block_indices;
        const auto& block_indices_begins = t.block_indices_begins;
        const size_t block_size = c.page_size ? c.page_size : key_cache.size(2);

        // Physical slot (block * block_size + offset) of every new token, as PagedAttention builds it.
        const size_t tokens = t.key.size(0);
        m_slot_mapping.resize<int32_t>({tokens});
        size_t idx = 0;
        for (size_t seq = 0; seq < past_lens.size(0); seq++) {
            const int32_t q_len = subsequence_begins.ptr<int32_t>()[seq + 1] - subsequence_begins.ptr<int32_t>()[seq];
            const int32_t past = past_lens.ptr<int32_t>()[seq];
            const int32_t first_block = block_indices_begins.ptr<int32_t>()[seq];
            for (int32_t j = 0; j < q_len; j++) {
                const auto pos = static_cast<size_t>(past) + static_cast<size_t>(j);
                const auto block = static_cast<size_t>(block_indices.ptr<int32_t>()[first_block + pos / block_size]);
                m_slot_mapping.ptr<int32_t>()[idx++] = static_cast<int32_t>(block * block_size + pos % block_size);
            }
        }

        const auto key_4d = t.key.reshape({tokens, c.num_kv_heads, 1, c.k_head_size});
        const auto value_4d = t.value.reshape({tokens, c.num_kv_heads, 1, c.v_head_size});
        if (!is_quantized_cache(c.key_spec)) {
            paged_attn_memcpy(key_4d, value_4d, key_cache, t.value_cache, m_slot_mapping, cpu_parallel);
            return;
        }
        const QuantizeParam quant_param{c.key_spec.by_channel,
                                        false,
                                        false,
                                        c.key_spec.group_size,
                                        c.value_spec.group_size};
        if (c.key_spec.by_channel) {
            // One page of dequantized keys per worker thread (only the keys can be quantized by channel).
            m_temp_buffer.resize<float>(
                {static_cast<size_t>(cpu_parallel->get_num_worker_threads()), block_size, c.k_head_size});
        }
        paged_attn_quantkv(key_4d,
                           value_4d,
                           key_cache,
                           t.value_cache,
                           past_lens,
                           subsequence_begins,
                           block_indices,
                           block_indices_begins,
                           m_slot_mapping,
                           m_temp_buffer,
                           quant_param,
                           cpu_parallel);
    }

    void run(const SparsePATensors& t,
             const MemoryPtr& scratch_memory,
             const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
        const TileBrgemm& brgemm = m_cfg.brgemm.handle();
        uint8_t* const scratch = scratch_memory->getDataAs<uint8_t>();
        const SparsePAScratch scratch_layout(m_cfg,
                                             t,
                                             static_cast<size_t>(cpu_parallel->get_num_worker_threads()),
                                             m_thread_block_bytes);
        OPENVINO_ASSERT(scratch_memory->getSize() >= scratch_layout.total, "[CPU] sparse_pa: scratch is too small");
        auto* const token_ids = reinterpret_cast<int32_t*>(scratch + scratch_layout.token_ids);

        const auto data_prc = t.query.get_precision();
        OPENVINO_ASSERT(t.out.get_precision() == data_prc, "[CPU] sparse_pa: query/output precisions must match");
        // Plain caches: the query precision, or f16 next to f32 data like PagedAttention (read in place, no conversion
        // pass). The AMX tiles are built for the query precision, so a cache of another type uses the SIMD heads.
        const bool cache_is_data = t.key_cache.get_precision() == data_prc && t.value_cache.get_precision() == data_prc;
        OPENVINO_ASSERT(is_quantized_cache(m_cfg.key_spec) || cache_is_data ||
                            (data_prc == ov::element::f32 && t.key_cache.get_precision() == ov::element::f16 &&
                             t.value_cache.get_precision() == ov::element::f16),
                        "[CPU] sparse_pa: plain caches must have the query precision, or f16 with f32 data");
        // A brgemm item needs the kernels (they exist for causal packs of several heads only) and a cache the tiles
        // read.
        const bool tiles = brgemm.enabled() && m_cfg.causal && (is_quantized_cache(m_cfg.key_spec) || cache_is_data);
        OPENVINO_ASSERT(!brgemm.enabled() || brgemm.prec == data_prc,
                        "[CPU] sparse_pa: the brgemm kernels must be built for the query precision");
        OPENVINO_ASSERT(scratch != nullptr, "[CPU] sparse_pa: scratch must not be null");
        OPENVINO_ASSERT(t.subsequence_begins.size(0) >= 1, "[CPU] sparse_pa: subsequence_begins must be non-empty");

        // The work list: the tokens of sequences with several new tokens first (brgemm items when the brgemm kernels
        // are enabled), then the rest (fallback items); both id lists share the `token_ids` region at the end of the
        // scratch (one id per token).
        const size_t max_pack_size =
            max_q_head_pack_size(m_cfg.num_heads, m_cfg.num_kv_heads, m_cfg.num_selection_heads);
        const auto* begins = t.subsequence_begins.ptr<int32_t>();
        const size_t sequences = t.subsequence_begins.size(0) - 1;
        const size_t tokens = t.query.size(0);
        const auto in_range = [tokens](int32_t token) {
            return static_cast<size_t>(token) < tokens;
        };
        // Only the prefill (a sequence of several new tokens) is a brgemm item; a decode token is a SIMD item.
        const auto is_brgemm_sequence = [&](size_t seq) {
            return tiles && begins[seq + 1] - begins[seq] > 1;
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
        // Few fallback items on many threads: split their blocks into kv_chunks (not with brgemm tokens in the call,
        // which run in the same loop).
        const auto num_threads = static_cast<size_t>(cpu_parallel->get_num_worker_threads());
        const size_t fallback_items = work.fallback_tokens * (m_cfg.num_heads / max_pack_size);
        work.kv_chunks =
            work.brgemm_tokens == 0 ? decode_kv_chunk_count(fallback_items, t.sel_indices.size(2), num_threads) : 1;
        work.fallback_pack = (work.kv_chunks > 1 || fallback_items >= 2 * num_threads) ? max_pack_size : 1;

        const SparsePAArgs args{m_cfg, t, work, scratch_memory, scratch, scratch_layout};

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

    SparsePAConfig m_cfg;
    size_t m_thread_block_bytes;  // bytes of the scratch block of one worker thread
    PlainTensor m_slot_mapping;   // int32 [tokens], grown on demand
    PlainTensor m_temp_buffer;    // f32 [threads, page rows, k_head_size]: by-channel key requantization
};

}  // namespace

std::shared_ptr<ov::Extensions::Cpu::SparsePAExecutor> make_sparse_pa_executor(
    const ov::Extensions::Cpu::SparsePAConfig& config) {
    return std::make_shared<PaExecutor>(config);
}

}  // namespace ov::Extensions::Cpu::XARCH
