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
#include <type_traits>
#include <utility>

#include "cache/multi_cache.h"
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

#if defined(OPENVINO_ARCH_X86_64) && defined(HAVE_AVX512F)
#    include "nodes/kernels/scaled_attn/attn_brgemm_tiles.hpp"
#endif

namespace ov::Extensions::Cpu::XARCH {

using ov::intel_cpu::CacheSpec;
using ov::intel_cpu::is_quantized_cache;
using ov::intel_cpu::MemoryPtr;
using ov::intel_cpu::PlainTensor;

namespace {

// Cache formats, as in PagedAttention (see executor_pa.cpp): by token (key or value) a row is groups of
// `[scale f32][zp f32][data]` (group size 0: one group per row); by channel (key only) a page is
// `f32 scale[S] | f32 zp[S] | data[block_size, S]`. Both caches are compressed or both plain.

/// @brief Byte layout of the scratch block of one worker thread, for the largest query-head pack (every part 64-byte
/// aligned). The scratch of a call holds one block per thread, so nothing is allocated while computing. A thread runs
/// one item at a time, on the SIMD or the AMX path, so the buffers of the two paths overlap after the shared state.
struct SparsePAScratchLayout {
    PackStateLayout pack;  // accumulators and running softmax maximum / sum of the head pack, used by both paths

    // SIMD path
    size_t q = 0;           // float[pack, k_head_size]: the query rows
    size_t scores = 0;      // float[pack, max_attend_kv_rows]: scores, then weights
    size_t k_rows_f32 = 0;  // compressed keys: float[max_attend_kv_rows, k_head_size]
    size_t v_rows_f32 = 0;  // compressed values, pack > 1: float[max_attend_kv_rows, v_head_size]

    TileBrgemmLayout brgemm;  // AMX path: the KV tile, scores / weights and the brgemm scratch

    size_t total = 0;  // bytes per thread

    SparsePAScratchLayout(size_t k_head_size,
                          size_t v_head_size,
                          size_t max_attend_kv_rows,
                          size_t max_pack_size,
                          size_t data_size,
                          const CacheSpec& key_spec,
                          const TileBrgemmSizes& brgemm_sizes = {})
        : pack(0, k_head_size, v_head_size, max_pack_size, data_size, false) {
        ScratchPartition block;
        block.reserve(pack.total);
        q = block.reserve(max_pack_size * k_head_size * sizeof(float));
        scores = block.reserve(max_pack_size * max_attend_kv_rows * sizeof(float));
        // compressed keys are dequantized to f32 for by-channel keys and for packs, values for packs
        const bool compressed = is_quantized_cache(key_spec);
        const bool dequantize_keys = compressed && (key_spec.by_channel || max_pack_size > 1);
        k_rows_f32 = block.reserve((dequantize_keys ? max_attend_kv_rows : 0) * k_head_size * sizeof(float));
        v_rows_f32 =
            block.reserve((compressed && max_pack_size > 1 ? max_attend_kv_rows : 0) * v_head_size * sizeof(float));
        // The AMX buffers start where the SIMD ones do.
        brgemm = TileBrgemmLayout(q, k_head_size, v_head_size, max_pack_size, data_size, brgemm_sizes);
        total = std::max(block.bytes(), brgemm.total);
    }
};

// Compressed (u8 / u4) paged cache access; each function picks the precision at runtime (visit_quant_precision).

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
// `token_ids` lists brgemm tokens first. With little SIMD work every query head is its own item (`simd_q_head_pack` 1).
struct WorkList {
    const int32_t* token_ids = nullptr;
    size_t brgemm_tokens = 0;
    size_t simd_tokens = 0;
    size_t brgemm_q_head_pack = 1;  ///< query heads of one item of a brgemm token
    size_t simd_q_head_pack = 1;    ///< query heads of one item of a SIMD token
    size_t kv_chunks = 1;           ///< slices of the selected blocks of a SIMD item (a decode of few items is split)
};

// The most bytes of dequantized pages a call may take from the scratch: the scratch only grows, so its peak stays
// allocated. A call above it dequantizes row by row.
constexpr size_t kMaxDequantizedBytes = size_t{256} << 20;

// The pages of a compressed cache that a call dequantizes.
struct DequantizedPlan {
    size_t entries = 0;  ///< block_indices entries the page table covers, 0: no dequantized pages (per-row path)
    size_t pages = 0;    ///< pages dequantized: those of the prefill sequences only
    size_t bytes = 0;    ///< of the key and value rows of those pages
};

// The bytes of the key and value rows of `pages` pages.
size_t dequantized_page_bytes(size_t pages, const SparsePATensors& t, const SparsePAConfig& cfg) {
    const size_t page_size = cfg.page_size ? cfg.page_size : t.key_cache.size(2);
    return pages * cfg.num_kv_heads * page_size * (cfg.k_head_size + cfg.v_head_size) * cfg.data_precision.size();
}

// A compressed cache is dequantized once per call for prefill sequences when their tokens gather at least as many rows
// as their pages hold, so a row many tokens select is not dequantized per token. The decode sequences keep to the
// per-row path and take no pages.
DequantizedPlan plan_dequantized_pages(const SparsePATensors& t, const SparsePAConfig& cfg) {
    if (!cfg.brgemm.handle().enabled() || !cfg.causal || !is_quantized_cache(cfg.key_spec) ||
        t.query.get_precision() != cfg.brgemm.handle().prec) {
        return {};
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
    const size_t bytes = dequantized_page_bytes(pages, t, cfg);
    if (pages == 0 || tokens * t.sel_indices.size(2) * cfg.key_block_size < pages * page_size ||
        bytes > kMaxDequantizedBytes) {
        return {};
    }
    return {static_cast<size_t>(blocks[sequences]), pages, bytes};
}

// Where the dequantized pages lie in their region of the scratch: the page table (one entry per block_indices entry),
// then the key rows and the value rows of the pages ([pages, Hkv, page_size, head size] each).
struct DequantizedLayout {
    size_t table = 0;  ///< byte offsets in the region
    size_t key = 0;
    size_t value = 0;
    size_t bytes = 0;

    DequantizedLayout(const DequantizedPlan& plan, const SparsePATensors& t, const SparsePAConfig& cfg) {
        const size_t page_size = cfg.page_size ? cfg.page_size : t.key_cache.size(2);
        const size_t page_rows = plan.pages * cfg.num_kv_heads * page_size;
        ScratchPartition region;
        table = region.reserve(plan.entries * sizeof(int32_t));
        key = region.reserve(page_rows * cfg.k_head_size * cfg.data_precision.size());
        value = region.reserve(page_rows * cfg.v_head_size * cfg.data_precision.size());
        bytes = region.bytes();
    }
};

/// @brief The scratch of a call: | one block per worker thread | one token id per token | partial states OR dequantized
/// pages |. Partial states exist only for a kv-chunk split, pages only with AMX tokens, so they share the memory.
struct SparsePAScratch {
    size_t threads = 0;  ///< num_threads blocks of thread_bytes, see SparsePAScratchLayout
    size_t thread_bytes = 0;
    size_t token_ids = 0;  ///< int32[tokens]
    size_t partials =
        0;  ///< float[num_threads, max_pack * (v_head_size + 2)]: the partial softmax states of a split decode
    size_t dequantized = 0;  ///< the pages of DequantizedLayout
    DequantizedPlan dequantized_plan;
    DequantizedLayout dequantized_layout;
    size_t total = 0;

    SparsePAScratch(const SparsePAConfig& cfg, const SparsePATensors& t, size_t num_threads, size_t block_bytes)
        : thread_bytes(block_bytes),
          dequantized_plan(plan_dequantized_pages(t, cfg)),
          dequantized_layout(dequantized_plan, t, cfg) {
        const size_t max_pack = max_q_head_pack_size(cfg.num_heads, cfg.num_kv_heads, cfg.num_selection_heads);
        ScratchPartition scratch;
        threads = scratch.reserve(num_threads * thread_bytes);
        token_ids = scratch.reserve(t.query.size(0) * sizeof(int32_t));
        partials = dequantized = scratch.reserve(
            std::max(num_threads * max_pack * (cfg.v_head_size + 2) * sizeof(float), dequantized_layout.bytes));
        total = scratch.bytes();
    }
};

// The sequence a token belongs to: its page table and the positions it may read.
struct TokenSeq {
    const int32_t* block_table = nullptr;  // physical block ids of the sequence
    int64_t token_pos = 0;                 // logical position of the query token
    int64_t valid_max = 0;                 // exclusive bound on the valid logical positions of the sequence
};

// Finds the sequence of `token`; false if it belongs to none. `block_indices` is the page table the call reads.
bool locate_token(const SparsePATensors& t, const PlainTensor& block_indices, size_t token, TokenSeq& seq) {
    const auto* begins = t.subsequence_begins.ptr<int32_t>();
    const auto* const begins_end = begins + t.subsequence_begins.size(0);
    const size_t s =
        static_cast<size_t>(std::upper_bound(begins, begins_end, static_cast<int32_t>(token)) - begins) - 1;
    if (s + 1 >= t.subsequence_begins.size(0)) {
        return false;
    }
    const int32_t block_begin = t.block_indices_begins.ptr<int32_t>()[s];
    OPENVINO_ASSERT(t.block_indices_begins.ptr<int32_t>()[s + 1] > block_begin,
                    "[CPU] sparse_pa: each sequence must have at least one cache block");
    const int64_t past_len = t.past_lens.ptr<int32_t>()[s];
    seq.block_table = block_indices.ptr<int32_t>() + block_begin;
    seq.token_pos = past_len + (static_cast<int32_t>(token) - begins[s]);
    // Enforced unconditionally (not just under causal) so a non-causal gather never reads an unallocated or stale slot.
    seq.valid_max = past_len + (begins[s + 1] - begins[s]);
    return true;
}

#if defined(OPENVINO_ARCH_X86_64) && defined(HAVE_AVX512F)
// Row source of gather_selected_kv() for the paged caches: the rows of a call lie in one physical page. The caches are
// the originals or their dequantized pages; when `compressed` they are dequantized row by row.
struct PagedRowSource {
    const SparsePAConfig& cfg;
    const PlainTensor& key_cache;
    const PlainTensor& value_cache;
    bool compressed;
    const TokenSeq& seq;
    size_t k_head;
    int64_t page_size;
    int64_t visible_end;

    [[nodiscard]] int64_t visible(int64_t kpos0, int64_t n) const {
        return XARCH::visible_rows(kpos0, n, 0, visible_end);
    }

    template <typename T>
    void locate_kv_rows(int64_t kpos0, size_t n, const T** k_ptr, const T** v_ptr, T* k_stage, T* v_stage) const {
        const auto phys_block = static_cast<size_t>(seq.block_table[kpos0 / page_size]);
        const auto slot = static_cast<size_t>(kpos0 % page_size);
        const T* k_src = k_stage;
        const T* v_src = v_stage;
        if (compressed) {
            dequant_k_rows<T>(cfg, key_cache, phys_block, k_head, slot, n, cfg.k_head_size, k_stage);
            dequant_v_rows<T>(cfg, value_cache, phys_block, k_head, slot, n, cfg.v_head_size, v_stage);
        } else {
            k_src = key_cache.ptr<T>(phys_block, k_head, slot);
            v_src = value_cache.ptr<T>(phys_block, k_head, slot);
        }
        for (size_t i = 0; i < n; i++) {
            k_ptr[i] = k_src + i * cfg.k_head_size;
            v_ptr[i] = v_src + i * cfg.v_head_size;
        }
    }
};
#endif  // OPENVINO_ARCH_X86_64 && HAVE_AVX512F

// The pages of the compressed caches that a call dequantized: views of the scratch pad memory, which they share the
// ownership of.
struct DequantizedPages {
    PlainTensor key;    // [pages, Hkv, page_size, k_head_size] in the data type
    PlainTensor value;  // [pages, Hkv, page_size, v_head_size]
    PlainTensor table;  // [entries] page table: the page of entry i of block_indices (the physical block ids of
                        // block_indices index the compressed cache, not these pages)
};

// Dequantizes the pages of prefill sequences into their region of the scratch (see SparsePAScratch) and sets the views
// of them in `pages`. The pages are packed in block_indices order; the table maps an entry of block_indices to its
// page (-1: an entry of a sequence that does not use the pages).
template <typename T>
void build_dequantized_pages(const SparsePAConfig& cfg,
                             const SparsePATensors& t,
                             const SparsePAScratch& parts,
                             const MemoryPtr& scratch_memory,
                             DequantizedPages& pages,
                             const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    const auto& region = parts.dequantized_layout;
    const size_t entries = parts.dequantized_plan.entries;
    const size_t page_count = parts.dequantized_plan.pages;
    const size_t k_head_num = t.key_cache.size(1);
    const size_t page_size = cfg.page_size ? cfg.page_size : t.key_cache.size(2);
    pages.table.resize_view<int32_t>({entries}, scratch_memory, parts.dequantized + region.table);
    pages.key.resize_view<T>({page_count, k_head_num, page_size, cfg.k_head_size},
                             scratch_memory,
                             parts.dequantized + region.key);
    pages.value.resize_view<T>({page_count, k_head_num, page_size, cfg.v_head_size},
                               scratch_memory,
                               parts.dequantized + region.value);
    auto* table = pages.table.ptr<int32_t>();
    std::fill_n(table, entries, -1);
    const auto* begins = t.subsequence_begins.ptr<int32_t>();
    const auto* blocks = t.block_indices_begins.ptr<int32_t>();
    const size_t sequences = t.subsequence_begins.size(0) - 1;
    int32_t next_page = 0;
    for (size_t seq = 0; seq < sequences; seq++) {
        if (begins[seq + 1] - begins[seq] > 1) {
            for (int32_t entry = blocks[seq]; entry < blocks[seq + 1]; entry++) {
                table[entry] = next_page++;
            }
        }
    }
    OPENVINO_ASSERT(static_cast<size_t>(next_page) == page_count, "[CPU] sparse_pa: dequantized page count mismatch");
    cpu_parallel->parallel_for2d(entries, k_head_num, [&](size_t entry, size_t hk) {
        if (table[entry] < 0) {
            return;  // an entry of a sequence that does not use the dequantized pages
        }
        const auto page = static_cast<size_t>(table[entry]);
        const auto phys_block = static_cast<size_t>(t.block_indices.ptr<int32_t>()[entry]);
        dequant_k_rows<T>(cfg, t.key_cache, phys_block, hk, 0, page_size, cfg.k_head_size, pages.key.ptr<T>(page, hk));
        dequant_v_rows<T>(cfg,
                          t.value_cache,
                          phys_block,
                          hk,
                          0,
                          page_size,
                          cfg.v_head_size,
                          pages.value.ptr<T>(page, hk));
    });
}

class PaExecutor final : public SparsePAExecutor {
public:
    // The scratch block of a worker thread depends on the config only, so it is laid out once here.
    explicit PaExecutor(const SparsePAConfig& config)
        : m_cfg(config),
          m_thread(config.k_head_size,
                   config.v_head_size,
                   config.max_attend_kv_rows,
                   max_q_head_pack_size(config.num_heads, config.num_kv_heads, config.num_selection_heads),
                   config.data_precision.size(),
                   config.key_spec,
                   tile_brgemm_sizes(config.brgemm.handle())) {}

    [[nodiscard]] ov::intel_cpu::impl_desc_type impl_type() const override {
        return m_cfg.brgemm.handle().enabled() ? ov::intel_cpu::impl_desc_type::brgemm_avx512_amx : variant_impl_type();
    }

    [[nodiscard]] size_t scratch_bytes(const SparsePATensors& t,
                                       const ov::intel_cpu::CpuParallelPtr& cpu_parallel) const override {
        return SparsePAScratch(m_cfg, t, static_cast<size_t>(cpu_parallel->get_num_worker_threads()), m_thread.total)
            .total;
    }

    void execute(const SparsePATensors& t,
                 const MemoryPtr& scratch,
                 const ov::intel_cpu::CpuParallelPtr& cpu_parallel) override {
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
        const size_t sequences = subsequence_begins.size(0) - 1;
        for (size_t seq = 0; seq < sequences; seq++) {
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
                                             m_thread.total);
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
        // are enabled), then the rest (SIMD items); both id lists share the `token_ids` region of the scratch
        // (one id per token).
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
                    token_ids[work.brgemm_tokens + work.simd_tokens++] = token;
                }
            }
        }
        work.brgemm_q_head_pack = max_pack_size;
        // Few SIMD items on many threads: split their blocks into kv_chunks (not when AMX tokens share the loop).
        const auto num_threads = static_cast<size_t>(cpu_parallel->get_num_worker_threads());
        const size_t simd_items = work.simd_tokens * (m_cfg.num_heads / max_pack_size);
        work.kv_chunks = work.brgemm_tokens == 0 ? kv_chunk_count(simd_items, t.sel_indices.size(2), num_threads) : 1;
        work.simd_q_head_pack = (work.kv_chunks > 1 || simd_items >= 2 * num_threads) ? max_pack_size : 1;

        if (data_prc == ov::element::f32) {
            attend<float>(t, work, scratch_layout, scratch_memory, cpu_parallel);
        } else if (data_prc == ov::element::f16) {
            attend<ov::float16>(t, work, scratch_layout, scratch_memory, cpu_parallel);
        } else if (data_prc == ov::element::bf16) {
            attend<ov::bfloat16>(t, work, scratch_layout, scratch_memory, cpu_parallel);
        } else {
            OPENVINO_THROW("[CPU] sparse_pa: unsupported precision ", data_prc);
        }
    }

#if defined(OPENVINO_ARCH_X86_64) && defined(HAVE_AVX512F)
    // AMX item: the attention of one query token and a pack of query heads over its whole selection, as matrix
    // products over KV tiles (see attn_brgemm_tiles.hpp). `pages`: the dequantized pages of a compressed cache, or
    // null.
    template <typename T>
    void attention_amx_item(const SparsePATensors& t,
                            const WorkList& work,
                            const SparsePAScratch& parts,
                            uint8_t* scratch,
                            const DequantizedPages* pages,
                            size_t token,
                            size_t pack_index) const {
        TokenSeq seq;
        if (!locate_token(t, pages != nullptr ? pages->table : t.block_indices, token, seq)) {
            return;
        }
        const size_t pack_size = work.brgemm_q_head_pack;
        const size_t Dh = m_cfg.k_head_size;
        const size_t Dv = m_cfg.v_head_size;
        // The query heads of a pack share the K head and the selection head of the pack's first query head.
        const size_t first_head = pack_index * pack_size;
        const size_t k_head = first_head / (m_cfg.num_heads / t.key_cache.size(1));
        const size_t sel_head = first_head / (m_cfg.num_heads / t.sel_indices.size(1));
        const auto key_block_size = static_cast<int64_t>(m_cfg.key_block_size);

        uint8_t* ws = scratch + parts.threads + static_cast<size_t>(parallel_get_thread_num()) * parts.thread_bytes;
        auto* acc = reinterpret_cast<float*>(ws + m_thread.pack.acc);
        auto* running_max = reinterpret_cast<float*>(ws + m_thread.pack.running_max);
        auto* running_sum = reinterpret_cast<float*>(ws + m_thread.pack.running_sum);
        init_pack_state(acc, running_max, running_sum, pack_size, Dv);

        const auto buffers = make_brgemm_buffers<T>(m_cfg.brgemm.handle(), m_thread.brgemm, ws);
        // The query rows of the pack are contiguous in the query tensor: used in place.
        const BrgemmPack
            pack{t.query.ptr<T>(token, first_head * Dh), pack_size, Dh, Dv, t.scale, acc, running_max, running_sum};
        const auto page_size = static_cast<int64_t>(m_cfg.page_size ? m_cfg.page_size : t.key_cache.size(2));
        const PagedRowSource src{m_cfg,
                                 pages != nullptr ? pages->key : t.key_cache,
                                 pages != nullptr ? pages->value : t.value_cache,
                                 pages == nullptr && is_quantized_cache(m_cfg.key_spec),
                                 seq,
                                 k_head,
                                 page_size,
                                 m_cfg.causal ? std::min(seq.valid_max, seq.token_pos + 1) : seq.valid_max};

        // The rows [kpos0, kpos0 + n) the query attends: gathered into the KV tile (step 1), which goes through step 2
        // whenever it is full. The online softmax carries the state across tiles.
        TileState tile;
        const auto attend_rows = [&](int64_t kpos0, int64_t n) {
            n = src.visible(kpos0, n);
            while (n > 0) {
                const int64_t gathered = gather_selected_kv<T>(src, pack, buffers, tile, kpos0, n);
                kpos0 += gathered;
                n -= gathered;
                if (tile.rows == buffers.tile_rows()) {
                    attention_over_kv_tile<T>(pack, buffers, tile);
                }
            }
        };
        // One query per selection row, so the row of a token is its global token index.
        const int32_t* block_ids = t.sel_indices.ptr<int32_t>(token, sel_head);
        const auto nblocks = static_cast<size_t>(*t.sel_count.ptr<int32_t>(token, sel_head));
        OPENVINO_ASSERT(nblocks <= t.sel_indices.size(2),
                        "[CPU] sparse_pa: block_count exceeds the width of block_indices");
        for (size_t i = 0; i < nblocks; i++) {
            attend_rows(block_ids[i] * key_block_size, key_block_size);
        }
        const int64_t tail_start = causal_tail_start(seq.token_pos, 0, key_block_size);
        attend_rows(tail_start, seq.token_pos - tail_start + 1);
        attention_over_kv_tile<T>(pack, buffers, tile);  // the last, partial tile

        store_normalized_pack(acc, running_sum, pack_size, Dv, [&](size_t head) {
            return t.out.ptr<T>(token, (first_head + head) * Dv);
        });
    }
#endif

    // SIMD item: attention of one query token and a pack of query heads (one head is a pack of one) over the
    // `kv_chunk`-th of `kv_chunks` slices of the selected blocks; the last slice also takes the causal tail. Written
    // out, or left in `partial` as a softmax state when split.
    template <typename T>
    void attention_simd_item(const SparsePATensors& t,
                             const WorkList& work,
                             const SparsePAScratch& parts,
                             uint8_t* scratch,
                             size_t token,
                             size_t pack_index,
                             size_t kv_chunk,
                             float* partial) const {
        TokenSeq seq;
        if (!locate_token(t, t.block_indices, token, seq)) {
            return;
        }
        const size_t pack_size = work.simd_q_head_pack;
        const size_t kv_chunks = work.kv_chunks;
        const size_t Dh = m_cfg.k_head_size;
        const size_t Dv = m_cfg.v_head_size;
        const size_t first_head = pack_index * pack_size;
        const size_t k_head = first_head / (m_cfg.num_heads / t.key_cache.size(1));
        const size_t sel_head = first_head / (m_cfg.num_heads / t.sel_indices.size(1));
        const auto key_block_size = static_cast<int64_t>(m_cfg.key_block_size);
        // Tokens per physical page: the config holds it for compressed caches (by-channel key pages carry parameter
        // rows in dim 2); a plain cache has it in dim 2.
        const auto page_size = static_cast<int64_t>(m_cfg.page_size ? m_cfg.page_size : t.key_cache.size(2));
        const size_t max_rows = m_cfg.max_attend_kv_rows;
        const int64_t visible_end = m_cfg.causal ? std::min(seq.valid_max, seq.token_pos + 1) : seq.valid_max;

        uint8_t* ws = scratch + parts.threads + static_cast<size_t>(parallel_get_thread_num()) * parts.thread_bytes;
        auto* q = reinterpret_cast<float*>(ws + m_thread.q);
        auto* acc = reinterpret_cast<float*>(ws + m_thread.pack.acc);
        auto* running_max = reinterpret_cast<float*>(ws + m_thread.pack.running_max);
        auto* running_sum = reinterpret_cast<float*>(ws + m_thread.pack.running_sum);
        auto* scores = reinterpret_cast<float*>(ws + m_thread.scores);
        auto* k_rows_f32 = reinterpret_cast<float*>(ws + m_thread.k_rows_f32);
        auto* v_rows_f32 = reinterpret_cast<float*>(ws + m_thread.v_rows_f32);
        cvt_copy(q, t.query.ptr<T>(token, first_head * Dh), 1, pack_size * Dh, 0, 0);
        init_pack_state(acc, running_max, running_sum, pack_size, Dv);

        // Attention of every head of the pack over up to max_rows consecutive rows [kpos0, kpos0 + n) of one page:
        // scores, online softmax, accumulate V. A one-head pack reads compressed rows in place (by-channel keys are
        // dequantized first), a larger pack dequantizes them to f32 once.
        const auto attend_kv_rows = [&](int64_t kpos0, int64_t n) {
            n = XARCH::visible_rows(kpos0, n, 0, visible_end);
            if (n == 0) {
                return;
            }
            const auto phys_block = static_cast<size_t>(seq.block_table[kpos0 / page_size]);
            const auto slot = static_cast<size_t>(kpos0 % page_size);
            const auto rows = static_cast<size_t>(n);
            const auto softmax = [&] {
                for (size_t head = 0; head < pack_size; head++) {
                    online_softmax_step(scores + head * max_rows,
                                        rows,
                                        t.scale,
                                        running_max[head],
                                        running_sum[head],
                                        acc + head * Dv,
                                        Dv);
                }
            };
            if (!is_quantized_cache(m_cfg.key_spec)) {
                // The query rows are f32, so only the cache rows need their own type (f16 next to f32 data, like
                // PagedAttention; converted in registers).
                visit_float_type(t.key_cache.get_precision(), [&](auto cache_tag) {
                    using C = decltype(cache_tag);
                    dot_product_block_multi_head<C>(q,
                                                    pack_size,
                                                    Dh,
                                                    t.key_cache.ptr<C>(phys_block, k_head, slot),
                                                    rows,
                                                    scores,
                                                    max_rows);
                    softmax();
                    attn_acc_value_block_multi_head<C>(acc,
                                                       pack_size,
                                                       Dv,
                                                       scores,
                                                       max_rows,
                                                       t.value_cache.ptr<C>(phys_block, k_head, slot),
                                                       rows);
                });
                return;
            }
            const bool fused_keys = pack_size == 1 && !m_cfg.key_spec.by_channel;
            const bool fused_values = pack_size == 1;
            if (!fused_keys) {
                dequant_k_rows<float>(m_cfg, t.key_cache, phys_block, k_head, slot, rows, Dh, k_rows_f32);
            }
            if (!fused_values) {
                dequant_v_rows<float>(m_cfg, t.value_cache, phys_block, k_head, slot, rows, Dv, v_rows_f32);
            }
            if (fused_keys) {
                dot_key_rows_by_token(m_cfg, t.key_cache, phys_block, k_head, slot, rows, Dh, q, scores);
            } else {
                dot_product_block_multi_head<float>(q, pack_size, Dh, k_rows_f32, rows, scores, max_rows);
            }
            softmax();
            if (fused_values) {
                acc_value_rows_by_token(m_cfg, t.value_cache, phys_block, k_head, slot, rows, Dv, scores, acc);
            } else {
                attn_acc_value_block_multi_head<float>(acc, pack_size, Dv, scores, max_rows, v_rows_f32, rows);
            }
        };

        // This item takes the blocks [first, last) of the selection. Adjacent blocks of one physical page are attended
        // together, up to max_rows rows at a time.
        const int32_t* block_ids = t.sel_indices.ptr<int32_t>(token, sel_head);
        const auto nblocks = static_cast<size_t>(*t.sel_count.ptr<int32_t>(token, sel_head));
        OPENVINO_ASSERT(nblocks <= t.sel_indices.size(2),
                        "[CPU] sparse_pa: block_count exceeds the width of block_indices");
        const size_t first = nblocks * kv_chunk / kv_chunks;
        const size_t last = nblocks * (kv_chunk + 1) / kv_chunks;
        const int64_t page_blocks = page_size / key_block_size;
        const int64_t max_adjacent_blocks = std::max<int64_t>(static_cast<int64_t>(max_rows) / key_block_size, 1);
        for (size_t i = first; i < last;) {
            const int64_t first_block = block_ids[i];
            int64_t adjacent_blocks = 1;
            while (i + adjacent_blocks < last && adjacent_blocks < max_adjacent_blocks &&
                   block_ids[i + adjacent_blocks] == first_block + adjacent_blocks &&
                   (first_block + adjacent_blocks) / page_blocks == first_block / page_blocks) {
                adjacent_blocks++;
            }
            attend_kv_rows(first_block * key_block_size, adjacent_blocks * key_block_size);
            i += static_cast<size_t>(adjacent_blocks);
        }
        // sel_indices / sel_count hold complete blocks only: when causal, the last chunk attends the trailing tail
        // itself; it is empty at a block boundary.
        if (m_cfg.causal && kv_chunk + 1 == kv_chunks) {
            const int64_t tail_start = causal_tail_start(seq.token_pos, 0, key_block_size);
            attend_kv_rows(tail_start, seq.token_pos - tail_start + 1);
        }

        if (partial != nullptr) {
            std::copy_n(running_max, pack_size, partial);
            std::copy_n(running_sum, pack_size, partial + pack_size);
            std::copy_n(acc, pack_size * Dv, partial + 2 * pack_size);
            return;
        }
        store_normalized_pack(acc, running_sum, pack_size, Dv, [&](size_t head) {
            return t.out.ptr<T>(token, (first_head + head) * Dv);
        });
    }

    // The attention of every (global query token, head pack, kv_chunk) item of the work list, in parallel: the AMX
    // items first, then the SIMD items. A kv-chunk split leaves partial softmax states for a second merging pass; the
    // AMX items of a compressed cache read its dequantized pages.
    template <typename T>
    void attend(const SparsePATensors& t,
                const WorkList& work,
                const SparsePAScratch& parts,
                const MemoryPtr& scratch_memory,
                const ov::intel_cpu::CpuParallelPtr& cpu_parallel) const {
        uint8_t* const scratch = scratch_memory->getDataAs<uint8_t>();
        auto* const partials = reinterpret_cast<float*>(scratch + parts.partials);

        // The pages share their memory with the partial states, which a call has only without AMX tokens.
        DequantizedPages dequantized;
        [[maybe_unused]] const DequantizedPages* pages = nullptr;  // only the AMX items read them
        if constexpr (!std::is_same_v<T, float>) {
            if (parts.dequantized_plan.entries > 0 && work.brgemm_tokens > 0) {
                build_dequantized_pages<T>(m_cfg, t, parts, scratch_memory, dequantized, cpu_parallel);
                pages = &dequantized;
            }
        }

        const size_t brgemm_per_token = m_cfg.num_heads / work.brgemm_q_head_pack;
        const size_t simd_per_token = m_cfg.num_heads / work.simd_q_head_pack;
        const size_t brgemm_items = work.brgemm_tokens * brgemm_per_token;
        const size_t partial_floats = work.simd_q_head_pack * (m_cfg.v_head_size + 2);
        // Items are (token, pack, kv_chunk): the AMX tokens first, each with packs of brgemm_q_head_pack heads and one
        // chunk, then the SIMD tokens; a kv-chunk split leaves its state in the slot of the item.
        cpu_parallel->parallel_for(brgemm_items + work.simd_tokens * simd_per_token * work.kv_chunks, [&](size_t item) {
            if (item < brgemm_items) {
#if defined(OPENVINO_ARCH_X86_64) && defined(HAVE_AVX512F)
                if constexpr (!std::is_same_v<T, float>) {
                    attention_amx_item<T>(t,
                                          work,
                                          parts,
                                          scratch,
                                          pages,
                                          work.token_ids[item / brgemm_per_token],
                                          item % brgemm_per_token);
                }
#endif
                return;
            }
            const size_t local = item - brgemm_items;
            const size_t per_token = simd_per_token * work.kv_chunks;
            attention_simd_item<T>(t,
                                   work,
                                   parts,
                                   scratch,
                                   work.token_ids[work.brgemm_tokens + local / per_token],
                                   (local % per_token) / work.kv_chunks,
                                   local % work.kv_chunks,
                                   work.kv_chunks > 1 ? partials + local * partial_floats : nullptr);
        });
        if (work.kv_chunks == 1) {
            return;
        }
        // Second pass: merge the partial states of every SIMD (token, pack) item into the output.
        cpu_parallel->parallel_for(work.simd_tokens * simd_per_token, [&](size_t token_pack) {
            const size_t token = work.token_ids[work.brgemm_tokens + token_pack / simd_per_token];
            const size_t first_head = (token_pack % simd_per_token) * work.simd_q_head_pack;
            auto* const merged = reinterpret_cast<float*>(
                scratch + parts.threads + static_cast<size_t>(parallel_get_thread_num()) * parts.thread_bytes +
                m_thread.pack.acc);
            merge_kv_chunk_partials(partials + token_pack * work.kv_chunks * partial_floats,
                                    work.kv_chunks,
                                    work.simd_q_head_pack,
                                    m_cfg.v_head_size,
                                    merged);
            for (size_t head = 0; head < work.simd_q_head_pack; head++) {
                cvt_copy(t.out.ptr<T>(token, (first_head + head) * m_cfg.v_head_size),
                         merged + head * m_cfg.v_head_size,
                         1,
                         m_cfg.v_head_size,
                         0,
                         0);
            }
        });
    }

    SparsePAConfig m_cfg;
    SparsePAScratchLayout m_thread;  // the scratch block of one worker thread
    PlainTensor m_slot_mapping;      // int32 [tokens], grown on demand
    PlainTensor m_temp_buffer;       // f32 [threads, page rows, k_head_size]: by-channel key requantization
};

}  // namespace

std::shared_ptr<ov::Extensions::Cpu::SparsePAExecutor> make_sparse_pa_executor(
    const ov::Extensions::Cpu::SparsePAConfig& config,
    const ov::intel_cpu::MultiCachePtr& cache) {
    auto executor_config = config;
    executor_config.brgemm = {};
    // The AMX items are compiled in the AVX-512 variant only: other variants build no kernels, so the work split and
    // the scratch plan no brgemm items, whatever the CPU runs.
#if defined(OPENVINO_ARCH_X86_64) && defined(HAVE_AVX512F)
    const auto precision = config.data_precision;
    const size_t pack = max_q_head_pack_size(config.num_heads, config.num_kv_heads, config.num_selection_heads);
    if (TileBrgemmKernels::amx_type(precision) == precision && config.causal &&
        TileBrgemmKernels::supports_pack(pack)) {
        executor_config.brgemm =
            TileBrgemmKernels::get(cache, pack, config.k_head_size, config.v_head_size, kGatherTileRows, precision);
    }
#else
    static_cast<void>(cache);
#endif
    return std::make_shared<PaExecutor>(executor_config);
}

}  // namespace ov::Extensions::Cpu::XARCH
