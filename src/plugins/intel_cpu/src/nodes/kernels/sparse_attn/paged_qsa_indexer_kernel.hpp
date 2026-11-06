// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// The computation of the PagedQSAIndexer node: the per-token math of the QSAIndexer (see qsa_indexer_kernel.hpp), but
// every cache (index_key_cache, index_block_rope_cache, block_index_keys_cache) is addressed through the shared
// PagedAttention page table (block_indices, block_indices_begins, subsequence_begins, past_lens) instead of a flat
// [B, ...] layout. A call of the node, per sequence:
//   1. appends the raw index keys of its new tokens to the paged key cache;
//   2. pools every block the new tokens complete (mean of its `ratio` raw keys -> RMSNorm -> partial RoPE with the
//      coefficients of the first token of the block) into the paged summary cache;
//   3. scores, for every new token, the complete blocks it can see and keeps the block_topk best block ids, ascending.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "cpu_memory.h"
#include "cpu_parallel.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/paged_qsa_indexer_config.hpp"
#include "nodes/kernels/scaled_attn/common.hpp"
#include "nodes/kernels/sparse_attn/qsa_indexer_common.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"

#ifdef OPENVINO_ARCH_X86_64
#    include "nodes/kernels/sparse_attn/qsa_brgemm_scorer.hpp"
#endif

namespace ov::intel_cpu::node::qsa::paged {

using Config = ov::op::internal::PagedQSAIndexer::Config;
using In = ov::op::internal::PagedQSAIndexer::InputIdx;
using Out = PagedQSAIndexerOutputIdx;

// All floating-point data ports of PagedQSAIndexer (index_query, raw_index_keys, rope_cos, rope_sin, q_norm_weight,
// k_norm_weight, index_key_cache, index_block_rope_cache, block_index_keys_cache) may independently be f32, f16 or
// bf16: these convert only the elements actually touched, so the compute stays f32 without a whole-tensor Reorder on
// every call over caches that grow every step. Every port has its own precision, so it is looked up per call.
template <typename... Idx>
void load_row_f32(const PlainTensor& t, size_t len, float* dst, Idx... idx) {
    const auto prec = t.get_precision();
    if (prec == ov::element::bf16) {
        ov::Extensions::Cpu::XARCH::cvt_copy(dst, t.ptr<ov::bfloat16>(idx...), 1, len, 0, 0);
    } else if (prec == ov::element::f16) {
        ov::Extensions::Cpu::XARCH::cvt_copy(dst, t.ptr<ov::float16>(idx...), 1, len, 0, 0);
    } else {
        ov::Extensions::Cpu::XARCH::cvt_copy(dst, t.ptr<float>(idx...), 1, len, 0, 0);
    }
}

template <typename... Idx>
void store_row_f32(const PlainTensor& t, size_t len, const float* src, Idx... idx) {
    const auto prec = t.get_precision();
    if (prec == ov::element::bf16) {
        ov::Extensions::Cpu::XARCH::cvt_copy(t.ptr<ov::bfloat16>(idx...), src, 1, len, 0, 0);
    } else if (prec == ov::element::f16) {
        ov::Extensions::Cpu::XARCH::cvt_copy(t.ptr<ov::float16>(idx...), src, 1, len, 0, 0);
    } else {
        ov::Extensions::Cpu::XARCH::cvt_copy(t.ptr<float>(idx...), src, 1, len, 0, 0);
    }
}

// Read-only f32 view of a row of a tensor of any of the three float types: the row itself when the tensor is f32 (no
// copy), otherwise the row converted into `scratch`.
template <typename... Idx>
const float* row_view(const PlainTensor& t, size_t len, float* scratch, Idx... idx) {
    if (t.get_precision() == ov::element::f32) {
        return t.ptr<float>(idx...);
    }
    load_row_f32(t, len, scratch, idx...);
    return scratch;
}

// The ports of the node, each taken by its own name.
struct Ports {
    explicit Ports(const MemoryArgs& memory)
        : q_idx(memory.at(paged_qsa_src_arg(In::INDEX_QUERY))),
          k_idx(memory.at(paged_qsa_src_arg(In::RAW_INDEX_KEYS))),
          rope_cos(memory.at(paged_qsa_src_arg(In::ROPE_COS))),
          rope_sin(memory.at(paged_qsa_src_arg(In::ROPE_SIN))),
          q_norm_weight(memory.at(paged_qsa_src_arg(In::Q_NORM_WEIGHT))),
          k_norm_weight(memory.at(paged_qsa_src_arg(In::K_NORM_WEIGHT))),
          past_lens(memory.at(paged_qsa_src_arg(In::PAST_LENS))),
          subsequence_begins(memory.at(paged_qsa_src_arg(In::SUBSEQUENCE_BEGINS))),
          block_indices(memory.at(paged_qsa_src_arg(In::BLOCK_INDICES))),
          block_indices_begins(memory.at(paged_qsa_src_arg(In::BLOCK_INDICES_BEGINS))),
          raw_k_cache(memory.at(paged_qsa_src_arg(In::INDEX_KEY_CACHE))),
          indexer_rope_cache(memory.at(paged_qsa_src_arg(In::INDEX_BLOCK_ROPE_CACHE))),
          summary_cache(memory.at(paged_qsa_src_arg(In::BLOCK_INDEX_KEYS_CACHE))),
          sel_indices(memory.at(paged_qsa_dst_arg(Out::SEL_INDICES))),
          sel_count(memory.at(paged_qsa_dst_arg(Out::SEL_COUNT))) {}

    PlainTensor q_idx, k_idx, rope_cos, rope_sin, q_norm_weight, k_norm_weight;
    PlainTensor past_lens, subsequence_begins, block_indices, block_indices_begins;
    PlainTensor raw_k_cache, indexer_rope_cache, summary_cache;
    PlainTensor sel_indices, sel_count;
};

// The sizes of a call that follow from the shapes of the ports (the others are in Config).
inline size_t num_tokens(const Ports& p) {
    return p.q_idx.size(0);
}
inline size_t num_sequences(const Ports& p) {
    OPENVINO_ASSERT(p.subsequence_begins.size(0) >= 1, "[CPU] paged_qsa_indexer: subsequence_begins must be non-empty");
    return p.subsequence_begins.size(0) - 1;
}
inline size_t index_query_heads(const Ports& p, const Config& c) {
    return p.q_idx.size(1) / c.indexer_head_dim;
}
inline size_t index_key_heads(const Ports& p, const Config& c) {
    return p.k_idx.size(1) / c.indexer_head_dim;
}

// What a call knows about one sequence: where its tokens and its page-table row are, and which index blocks it
// completes. past_blocks = past_len / key_block_size blocks were complete before the call, new_blocks =
// (past_len + q_len) / key_block_size - past_blocks are completed by it. It also carries the paging of the caches.
struct SeqPlan {
    int64_t key_block_size;  // tokens per index block
    int64_t page_size;       // tokens per page of the caches, from the shape of the key cache
    int64_t slots_per_page;  // index blocks per page
    int64_t token_begin;
    int64_t q_len;
    int64_t past_len;
    int64_t block_begin;     // first entry of the sequence in block_indices
    int64_t block_capacity;  // index blocks that fit the sequence's pages
    int64_t past_blocks;
    int64_t new_blocks;

    // The complete blocks visible to the token `v` of the call (counted from the first token of the sequence's call):
    // those up to the block the token completes.
    [[nodiscard]] int64_t blocks_visible(int64_t v) const {
        return (past_len + v + 1) / key_block_size;
    }
    // The complete blocks of the sequence after the call.
    [[nodiscard]] int64_t total_blocks() const {
        return past_blocks + new_blocks;
    }
};

inline SeqPlan seq_plan(const Ports& p, const Config& c, size_t seq) {
    const auto* subsequence_begins = p.subsequence_begins.ptr<int32_t>();
    const auto* block_indices_begins = p.block_indices_begins.ptr<int32_t>();
    const int64_t block_begin = block_indices_begins[seq];
    const int64_t seq_blocks = std::max<int64_t>(block_indices_begins[seq + 1] - block_begin, 0);
    SeqPlan plan{};
    plan.key_block_size = static_cast<int64_t>(c.key_block_size);
    plan.page_size = static_cast<int64_t>(p.raw_k_cache.size(2));
    plan.slots_per_page = plan.page_size / plan.key_block_size;
    plan.token_begin = subsequence_begins[seq];
    plan.q_len = std::max<int64_t>(subsequence_begins[seq + 1] - plan.token_begin, 0);
    plan.past_len = p.past_lens.ptr<int32_t>()[seq];
    plan.block_begin = block_begin;
    plan.block_capacity = seq_blocks * plan.slots_per_page;
    plan.past_blocks = plan.past_len / plan.key_block_size;
    plan.new_blocks = (plan.past_len + plan.q_len) / plan.key_block_size - plan.past_blocks;
    return plan;
}

// What the longest sequence of the call needs: the scores of a token are for at most max_blocks complete blocks, the
// pooling of a call is for at most max_new_blocks blocks of a sequence.
struct SeqStats {
    int64_t max_blocks = 0;
    int64_t max_new_blocks = 0;
    int64_t max_q_len = 0;
};

inline SeqStats seq_stats(const Ports& p, const Config& c) {
    SeqStats stats;
    const auto* block_indices_begins = p.block_indices_begins.ptr<int32_t>();
    for (size_t seq = 0; seq < num_sequences(p); seq++) {
        OPENVINO_ASSERT(block_indices_begins[seq + 1] > block_indices_begins[seq],
                        "[CPU] paged_qsa_indexer: each sequence must have at least one cache block");
        const auto plan = seq_plan(p, c, seq);
        OPENVINO_ASSERT(plan.total_blocks() <= plan.block_capacity,
                        "[CPU] paged_qsa_indexer: the completed blocks do not fit the sequence's cache blocks");
        stats.max_blocks = std::max(stats.max_blocks, plan.total_blocks());
        stats.max_new_blocks = std::max(stats.max_new_blocks, plan.new_blocks);
        stats.max_q_len = std::max(stats.max_q_len, plan.q_len);
    }
    return stats;
}

// A row of a paged cache: the physical page (an entry of block_indices) and the slot inside it.
struct CacheAddress {
    size_t page;
    size_t slot;
};

// Maps a logical token position (relative to the sequence's own 0..S history) to the (page, slot) address into
// INDEX_KEY_CACHE through the shared page table.
inline CacheAddress token_address(const Ports& p, const SeqPlan& plan, int64_t logical_pos) {
    const int64_t page_table_idx = plan.block_begin + logical_pos / plan.page_size;
    const auto page = static_cast<size_t>(*p.block_indices.ptr<int32_t>(static_cast<size_t>(page_table_idx)));
    return {page, static_cast<size_t>(logical_pos % plan.page_size)};
}

// Maps a (sequence-local, 0-based) completed-block index to the (page, slot) address into BLOCK_INDEX_KEYS_CACHE /
// INDEX_BLOCK_ROPE_CACHE.
inline CacheAddress block_address(const Ports& p, const SeqPlan& plan, int64_t block_idx) {
    const int64_t page_table_idx = plan.block_begin + (block_idx * plan.key_block_size) / plan.page_size;
    const auto page = static_cast<size_t>(*p.block_indices.ptr<int32_t>(static_cast<size_t>(page_table_idx)));
    return {page, static_cast<size_t>(block_idx % plan.slots_per_page)};
}

// The buffers of one worker thread's slice of the scratch; the phases run one after the other, so they share them.
struct ThreadBuffers {
    float* q_normed;     // [H_iq, D_idx] normalized + rotated index queries
    float* row_buf;      // [D_idx] a converted row
    float* summary_buf;  // [D_idx] a pooled summary row
    float* pooled;       // [D_idx] the mean of the raw keys of a block
    float* cos_sin;      // [2 * rotary_dim] the interleaved cos / sin row of a block
    float* cos_tmp;      // [rotary_dim] a cos row
    float* sin_tmp;      // [rotary_dim] a sin row
    float* stage;        // [4, D_idx] summary rows converted to f32 (the JIT scoring)
    float* score_f32;    // [blocks] the scores of the blocks of a token and selection head
    BlockScore* scores;  // [blocks] (score * scale, block), what the top-k selects from
    int32_t* chosen;     // [block_topk]
};

struct ThreadLayout {
    size_t q_normed_off = 0;  // floats
    size_t row_buf_off;
    size_t summary_buf_off;
    size_t pooled_off;
    size_t cos_sin_off;
    size_t cos_tmp_off;
    size_t sin_tmp_off;
    size_t stage_off;
    size_t score_f32_off;
    size_t scores_off;  // bytes
    size_t chosen_off;  // bytes
    size_t bytes;

    static constexpr size_t kAlign = 64;
    static constexpr size_t kStageRows = 4;  // the rows the JIT scoring converts at a time

    ThreadLayout(size_t H_iq, size_t D_idx, size_t rotary_dim, size_t max_blocks, size_t block_topk)
        : row_buf_off(q_normed_off + H_iq * D_idx),
          summary_buf_off(row_buf_off + D_idx),
          pooled_off(summary_buf_off + D_idx),
          cos_sin_off(pooled_off + D_idx),
          cos_tmp_off(cos_sin_off + 2 * rotary_dim),
          sin_tmp_off(cos_tmp_off + rotary_dim),
          stage_off(sin_tmp_off + rotary_dim),
          score_f32_off(stage_off + kStageRows * D_idx),
          scores_off(rnd_up((score_f32_off + max_blocks) * sizeof(float), kAlign)),
          chosen_off(rnd_up(scores_off + max_blocks * sizeof(BlockScore), kAlign)),
          bytes(rnd_up(chosen_off + block_topk * sizeof(int32_t), kAlign)) {}

    [[nodiscard]] ThreadBuffers bind(uint8_t* slice) const {
        auto* const floats = reinterpret_cast<float*>(slice);
        return {floats + q_normed_off,
                floats + row_buf_off,
                floats + summary_buf_off,
                floats + pooled_off,
                floats + cos_sin_off,
                floats + cos_tmp_off,
                floats + sin_tmp_off,
                floats + stage_off,
                floats + score_f32_off,
                reinterpret_cast<BlockScore*>(slice + scores_off),
                reinterpret_cast<int32_t*>(slice + chosen_off)};
    }
};

// The scratch of one call, one allocation (a second createScratchPadMem() could free the first's memory):
//   [ token -> sequence array, f32 q / k norm weights | what the brgemm scorer shares between the threads |
//     one slice per worker thread ]
// The scorer lays out its own slices over the same space, after phases 1 and 2 have used theirs.
class Scratch {
public:
    // `brgemm_scorer` is the scorer of the node (null: no brgemm scoring); the call only plans its sizes.
    Scratch(const Ports& p,
            const Config& c,
            const SeqStats& stats,
            const ExecutorContext::CPtr& context,
            const BrgemmBlockScorer* brgemm_scorer)
        : scorer(brgemm_scorer != nullptr && stats.max_blocks > 0 ? brgemm_scorer : nullptr),
          layout(index_query_heads(p, c),
                 c.indexer_head_dim,
                 c.indexer_rotary_dim,
                 static_cast<size_t>(stats.max_blocks),
                 c.block_topk) {
        const size_t D_idx = c.indexer_head_dim;
        const size_t tokens = num_tokens(p);
        size_t scorer_shared_bytes = 0;
        size_t scorer_thread_bytes = 0;
#ifdef OPENVINO_ARCH_X86_64
        if (scorer) {
            scorer_plan = scorer->plan(num_sequences(p), stats.max_q_len, stats.max_blocks);
            scorer_shared_bytes = scorer_plan.shared_bytes;
            scorer_thread_bytes = scorer_plan.thread_bytes;
        }
#endif
        constexpr size_t kAlign = ThreadLayout::kAlign;
        const size_t weights_off = rnd_up(tokens * sizeof(int32_t), kAlign);
        const size_t header_bytes = rnd_up(weights_off + 2 * D_idx * sizeof(float), kAlign);
        const size_t scorer_shared_offset = header_bytes;
        threads_offset = rnd_up(scorer_shared_offset + scorer_shared_bytes, kAlign);
        thread_bytes = std::max(layout.bytes, scorer_thread_bytes);
        const auto num_threads = static_cast<size_t>(CpuParallel::get_num_worker_threads());
        m_memory = context->getScratchPad()->createScratchPadMem(
            std::make_shared<CpuBlockedMemoryDesc>(ov::element::u8,
                                                   ov::intel_cpu::Shape{threads_offset + num_threads * thread_bytes}));
        base = m_memory->getDataAs<uint8_t>();

        token_seq = reinterpret_cast<int32_t*>(base);
        auto* const q_norm = reinterpret_cast<float*>(base + weights_off);
        auto* const k_norm = q_norm + D_idx;
        load_row_f32(p.q_norm_weight, D_idx, q_norm);
        load_row_f32(p.k_norm_weight, D_idx, k_norm);
        q_norm_w = q_norm;
        k_norm_w = k_norm;
        scorer_shared = base + scorer_shared_offset;
        std::fill_n(token_seq, tokens, static_cast<int32_t>(-1));
        for (size_t seq = 0; seq < num_sequences(p); seq++) {
            const auto plan = seq_plan(p, c, seq);
            for (int64_t t = plan.token_begin; t < plan.token_begin + plan.q_len; t++) {
                if (static_cast<size_t>(t) < tokens) {
                    token_seq[t] = static_cast<int32_t>(seq);
                }
            }
        }
    }

    [[nodiscard]] uint8_t* threads_base() const {
        return base + threads_offset;
    }
    // This worker thread's slice of the scratch.
    [[nodiscard]] ThreadBuffers thread() const {
        return layout.bind(threads_base() + static_cast<size_t>(parallel_get_thread_num()) * thread_bytes);
    }

    const BrgemmBlockScorer* scorer;  // null: no brgemm scoring
#ifdef OPENVINO_ARCH_X86_64
    BrgemmBlockScorer::Plan scorer_plan;
#endif
    ThreadLayout layout;
    int32_t* token_seq = nullptr;  // per token: its sequence, -1 for a token that belongs to none
    const float* q_norm_w = nullptr;
    const float* k_norm_w = nullptr;
    uint8_t* scorer_shared = nullptr;

private:
    size_t threads_offset = 0;
    size_t thread_bytes = 0;
    uint8_t* base = nullptr;
    MemoryPtr m_memory;
};

#ifdef OPENVINO_ARCH_X86_64
// The paged layout of a call as the brgemm scorer sees it (see BrgemmBlockScorer): the rows are the sequences, the
// token v of a sequence is the flat token plan.token_begin + v, and the summary blocks of a sequence are addressed
// through the page table: a page holds `slots_per_page` consecutive blocks of a K head, and the pages of a sequence
// are anywhere in the cache. `T` is the precision of the summary cache.
template <typename T>
class PagedSource {
public:
    using Element = T;

    PagedSource(const Ports& ports, const Config& config, const Scratch& scratch)
        : m_p(ports),
          m_c(config),
          m_q_norm_w(scratch.q_norm_w) {}

    [[nodiscard]] int64_t tokens(size_t seq) const {
        return seq_plan(m_p, m_c, seq).q_len;
    }
    // the complete blocks up to the one the token completes
    [[nodiscard]] int64_t blocks_visible(size_t seq, int64_t v) const {
        return seq_plan(m_p, m_c, seq).blocks_visible(v);
    }
    [[nodiscard]] int64_t blocks(size_t seq) const {
        return seq_plan(m_p, m_c, seq).total_blocks();
    }
    [[nodiscard]] const float* query_row(size_t seq, int64_t v, size_t h, float* buf) const {
        return row_view(m_p.q_idx, m_c.indexer_head_dim, buf, token(seq, v), h * m_c.indexer_head_dim);
    }
    [[nodiscard]] const float* rope_cos(size_t seq, int64_t v, float* buf) const {
        return row_view(m_p.rope_cos, m_c.indexer_rotary_dim, buf, token(seq, v));
    }
    [[nodiscard]] const float* rope_sin(size_t seq, int64_t v, float* buf) const {
        return row_view(m_p.rope_sin, m_c.indexer_rotary_dim, buf, token(seq, v));
    }
    [[nodiscard]] const float* q_norm_weight() const {
        return m_q_norm_w;
    }
    // 32 blocks that lie in one page are used in place; blocks of several pages (a small page, or the end of the
    // sequence) are gathered, run by run, into the zero-padded stage.
    [[nodiscard]] const T* block_rows(size_t seq, size_t key_head, size_t first, size_t count, T* stage) const {
        const auto plan = seq_plan(m_p, m_c, seq);
        const auto slots = static_cast<size_t>(plan.slots_per_page);
        const auto [first_page, first_slot] = block_address(m_p, plan, static_cast<int64_t>(first));
        if (count == kScoreBlockRows && first_slot + count <= slots) {
            return m_p.summary_cache.template ptr<T>(first_page, key_head, first_slot);
        }
        const size_t dim = m_c.indexer_head_dim;
        size_t done = 0;
        while (done < count) {
            const auto [page, slot] = block_address(m_p, plan, static_cast<int64_t>(first + done));
            const size_t run = std::min(count - done, slots - slot);
            std::copy_n(m_p.summary_cache.template ptr<T>(page, key_head, slot), run * dim, stage + done * dim);
            done += run;
        }
        std::fill_n(stage + count * dim, (kScoreBlockRows - count) * dim, T{});
        return stage;
    }
    [[nodiscard]] int32_t* selection_indices(size_t seq, int64_t v, size_t sel_head) const {
        return m_p.sel_indices.template ptr<int32_t>(token(seq, v), sel_head, 0);
    }
    [[nodiscard]] int32_t* selection_count(size_t seq, int64_t v, size_t sel_head) const {
        return m_p.sel_count.template ptr<int32_t>(token(seq, v), sel_head);
    }

private:
    [[nodiscard]] size_t token(size_t seq, int64_t v) const {
        return static_cast<size_t>(seq_plan(m_p, m_c, seq).token_begin + v);
    }

    const Ports& m_p;
    const Config& m_c;
    const float* m_q_norm_w;
};
#endif

// ---- Phase 1, per token: append the token's raw index keys to the paged cache, and when it starts a block record the
// interleaved cos/sin row of that first token, so the block's summary (finalized once the block completes, possibly in
// a later call) can be RoPE'ed with the exact coefficients of its first token, regardless of KV-cache slot vs logical
// position drift (continuous batching, chunked prefill, mRoPE, ...). The capacity bound skips a trailing block that
// starts within the allocated page range but has no room to ever complete. ----
inline void store_token_rows(const Ports& p, const Config& c, const Scratch& scratch, size_t token) {
    const auto plan = seq_plan(p, c, static_cast<size_t>(scratch.token_seq[token]));
    const int64_t logical_pos = plan.past_len + (static_cast<int64_t>(token) - plan.token_begin);
    const auto buf = scratch.thread();
    const size_t D_idx = c.indexer_head_dim;
    const size_t rotary_dim = c.indexer_rotary_dim;

    if (logical_pos % plan.key_block_size == 0) {
        const int64_t block_idx = logical_pos / plan.key_block_size;
        if (block_idx < plan.block_capacity) {
            float* const cos_sin = buf.cos_sin;
            const float* const cos_row = row_view(p.rope_cos, rotary_dim, buf.cos_tmp, token);
            const float* const sin_row = row_view(p.rope_sin, rotary_dim, buf.sin_tmp, token);
            for (size_t j = 0; j < rotary_dim; j++) {
                cos_sin[2 * j] = cos_row[j];
                cos_sin[2 * j + 1] = sin_row[j];
            }
            const auto [page, slot] = block_address(p, plan, block_idx);
            store_row_f32(p.indexer_rope_cache, 2 * rotary_dim, cos_sin, page, slot);
        }
    }
    const auto [cur_page, cur_slot] = token_address(p, plan, logical_pos);
    for (size_t key_head = 0; key_head < index_key_heads(p, c); key_head++) {
        const float* const row = row_view(p.k_idx, D_idx, buf.row_buf, token, key_head * D_idx);
        store_row_f32(p.raw_k_cache, D_idx, row, cur_page, key_head, cur_slot);
    }
}

// ---- Phase 2, per (sequence, index-key head, block completed by the call): the summary of the block is the mean of
// its raw keys, RMSNorm'ed and RoPE'd with the coefficients of its first token. The blocks are independent of each
// other. ----
template <typename Backend>
void pool_block(const Backend& backend,
                const Ports& p,
                const Config& c,
                const Scratch& scratch,
                size_t seq,
                size_t key_head,
                size_t j) {
    const auto plan = seq_plan(p, c, seq);
    const auto buf = scratch.thread();
    float* const pooled = buf.pooled;
    const size_t D_idx = c.indexer_head_dim;
    const size_t rotary_dim = c.indexer_rotary_dim;

    const int64_t block_idx = plan.past_blocks + static_cast<int64_t>(j);
    const int64_t block_start = block_idx * plan.key_block_size;
    const auto [block_page, block_slot] = block_address(p, plan, block_idx);
    std::fill_n(pooled, D_idx, 0.0F);
    for (int64_t pos = block_start; pos < block_start + plan.key_block_size; pos++) {
        const auto [p_page, p_slot] = token_address(p, plan, pos);
        const float* const row = row_view(p.raw_k_cache, D_idx, buf.row_buf, p_page, key_head, p_slot);
        for (size_t d = 0; d < D_idx; d++) {
            pooled[d] += row[d];
        }
    }
    const float inv_block_size = 1.0F / static_cast<float>(plan.key_block_size);
    for (size_t d = 0; d < D_idx; d++) {
        pooled[d] *= inv_block_size;
    }
    // the cos / sin of the first token of the block were stored interleaved; the RoPE takes them apart
    const float* const cos_sin = row_view(p.indexer_rope_cache, 2 * rotary_dim, buf.cos_sin, block_page, block_slot);
    for (size_t k = 0; k < rotary_dim; k++) {
        buf.cos_tmp[k] = cos_sin[2 * k];
        buf.sin_tmp[k] = cos_sin[2 * k + 1];
    }
    backend.rms_rope(pooled, buf.summary_buf, scratch.k_norm_w, D_idx, rotary_dim, c.epsilon, buf.cos_tmp, buf.sin_tmp);
    store_row_f32(p.summary_cache, D_idx, buf.summary_buf, block_page, key_head, block_slot);
}

// ---- Phase 3, per token: the query RMSNorm + partial RoPE of every index-query head, then per selection head the
// score of every complete block (the sum over the heads of the selection head of max(0, q . summary)) and the top-k
// blocks in ascending block order. T is the precision of the summary cache. ----
template <typename T, typename Backend>
void select_blocks(const Backend& backend, const Ports& p, const Config& c, const Scratch& scratch, size_t token) {
    const auto plan = seq_plan(p, c, static_cast<size_t>(scratch.token_seq[token]));
    const auto buf = scratch.thread();
    const size_t D_idx = c.indexer_head_dim;
    const size_t rotary_dim = c.indexer_rotary_dim;
    const size_t index_keys = index_key_heads(p, c);
    const size_t index_queries = index_query_heads(p, c);
    const size_t heads_per_key = index_queries / index_keys;
    const size_t selection_heads = c.num_selection_heads;
    const float scale = ov::op::internal::PagedQSAIndexer::effective_scale(c);

    const float* const cos_row = row_view(p.rope_cos, rotary_dim, buf.cos_tmp, token);
    const float* const sin_row = row_view(p.rope_sin, rotary_dim, buf.sin_tmp, token);
    for (size_t h = 0; h < index_queries; h++) {
        const float* const q_row = row_view(p.q_idx, D_idx, buf.row_buf, token, h * D_idx);
        backend.rms_rope(q_row,
                         buf.q_normed + h * D_idx,
                         scratch.q_norm_w,
                         D_idx,
                         rotary_dim,
                         c.epsilon,
                         cos_row,
                         sin_row);
    }

    // The complete blocks visible to this token, including the one this token completes.
    const int64_t n_complete = plan.blocks_visible(static_cast<int64_t>(token) - plan.token_begin);
    const auto slots = static_cast<size_t>(plan.slots_per_page);
    for (size_t sel_head = 0; sel_head < selection_heads; sel_head++) {
        // The index key heads of the selection head, and so their query heads: contiguous ranges.
        const size_t key_head_begin = div_up(sel_head * index_keys, selection_heads);
        const size_t key_head_end = div_up((sel_head + 1) * index_keys, selection_heads);
        size_t k = 0;
        if (n_complete > 0) {
            // the scores of the complete blocks, summed over the index key heads of the selection head: the blocks
            // of a head lie in runs of consecutive rows, one per page
            std::fill_n(buf.score_f32, static_cast<size_t>(n_complete), 0.0F);
            for (size_t key_head = key_head_begin; key_head < key_head_end; key_head++) {
                for (size_t blk = 0; blk < static_cast<size_t>(n_complete);) {
                    const auto [page, slot] = block_address(p, plan, static_cast<int64_t>(blk));
                    const size_t run = std::min(static_cast<size_t>(n_complete) - blk, slots - slot);
                    backend.template score_run<T>(buf.q_normed + key_head * heads_per_key * D_idx,
                                                  p.summary_cache.template ptr<T>(page, key_head, slot),
                                                  run,
                                                  heads_per_key,
                                                  D_idx,
                                                  buf.score_f32 + blk,
                                                  buf.stage,
                                                  buf.row_buf);
                    blk += run;
                }
            }
            for (int64_t blk = 0; blk < n_complete; blk++) {
                buf.scores[blk] = {buf.score_f32[blk] * scale, static_cast<int32_t>(blk)};
            }
            k = select_topk(buf.scores, buf.chosen, static_cast<size_t>(n_complete), c.block_topk, false);
        }

        std::copy_n(buf.chosen, k, p.sel_indices.template ptr<int32_t>(token, sel_head, 0));
        *p.sel_count.template ptr<int32_t>(token, sel_head) = static_cast<int32_t>(k);
    }
}

// The computation of the node, T is the precision of the summary blocks. A backend is what differs between the
// implementations (see ScalarBackend in qsa_indexer_common.hpp).
template <typename T, typename Backend>
void run_indexer_typed(const Backend& backend,
                       const MemoryArgs& memory,
                       const Config& config,
                       const ExecutorContext::CPtr& context) {
    const Ports ports(memory);
    const size_t tokens = num_tokens(ports);
    const SeqStats stats = seq_stats(ports, config);
    const Scratch scratch(ports, config, stats, context, backend.scorer());
    const auto cpu_parallel = context->getCpuParallel();
    const CpuParallel& parallel = *cpu_parallel;

    std::fill_n(ports.sel_indices.template ptr<int32_t>(), tokens * config.num_selection_heads * config.block_topk, -1);
    std::fill_n(ports.sel_count.template ptr<int32_t>(), tokens * config.num_selection_heads, 0);

    // Phase 1: the raw keys of the tokens (and the rope rows of the blocks they start) go to the caches; disjoint
    // slots.
    parallel.parallel_for(tokens, [&](size_t token) {
        if (scratch.token_seq[token] >= 0) {
            store_token_rows(ports, config, scratch, token);
        }
    });
    // Phase 2: the blocks this call completes are independent of each other (they read the raw rows of this and of
    // earlier calls, written by phase 1 or before).
    if (stats.max_new_blocks > 0) {
        parallel.parallel_for3d(num_sequences(ports),
                                index_key_heads(ports, config),
                                static_cast<size_t>(stats.max_new_blocks),
                                [&](size_t seq, size_t key_head, size_t j) {
                                    if (static_cast<int64_t>(j) < seq_plan(ports, config, seq).new_blocks) {
                                        pool_block(backend, ports, config, scratch, seq, key_head, j);
                                    }
                                });
    }
    // Phase 3: a token scores the complete blocks up to the one it completes, all written by now.
#ifdef OPENVINO_ARCH_X86_64
    if (scratch.scorer) {
        const PagedSource<T> source(ports, config, scratch);
        scratch.scorer->run(scratch.scorer_plan, source, scratch.scorer_shared, scratch.threads_base(), parallel);
        return;
    }
#endif
    parallel.parallel_for(tokens, [&](size_t token) {
        if (scratch.token_seq[token] >= 0) {
            select_blocks<T>(backend, ports, config, scratch, token);
        }
    });
}

template <typename Backend>
void run_indexer(const Backend& backend,
                 const MemoryArgs& memory,
                 const Config& config,
                 const ExecutorContext::CPtr& context) {
    // The precision of the summary blocks: the one the block scoring works in.
    const auto precision = memory.at(paged_qsa_src_arg(In::BLOCK_INDEX_KEYS_CACHE))->getDesc().getPrecision();
    if (precision == ov::element::f32) {
        run_indexer_typed<float>(backend, memory, config, context);
    } else if (precision == ov::element::f16) {
        run_indexer_typed<ov::float16>(backend, memory, config, context);
    } else if (precision == ov::element::bf16) {
        run_indexer_typed<ov::bfloat16>(backend, memory, config, context);
    } else {
        OPENVINO_THROW("PagedQSAIndexer: unsupported summary precision ", precision);
    }
}

}  // namespace ov::intel_cpu::node::qsa::paged
