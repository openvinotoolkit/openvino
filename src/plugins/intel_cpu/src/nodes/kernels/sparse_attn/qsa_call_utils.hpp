// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

// What the kernels of the QSAIndexer and the PagedQSAIndexer know about a call: the plan of every row (a batch entry
// of the dense op, a sequence of the paged one) that follows from the lengths in the call. Included by cross-compiled
// sources only (the generic kernels and the brgemm scorer), so it is compiled per instruction set in the namespace
// XARCH and nothing of it is visible to the nodes.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "nodes/kernels/scaled_attn/common.hpp"
#include "nodes/kernels/sparse_attn/qsa_executor.hpp"
#include "openvino/core/except.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH::qsa_kernel {

using ov::Extensions::Cpu::qsa::Config;
using ov::Extensions::Cpu::qsa::DenseCall;
using ov::Extensions::Cpu::qsa::PagedCall;
using ov::intel_cpu::PlainTensor;

// ---- dense op ----

// What one row of the batch contributes to a call. Every quantity is derived from the row's own key_length and the
// window length L of the call alone -- no cross-call running counter is kept.
struct RowPlan {
    int64_t num_padding;  // left-padded positions of this call's window, skipped everywhere
    int64_t current;      // real tokens of this call
    int64_t tail;         // rows already sitting in the incomplete-block state
    int64_t past_blocks;  // complete blocks of the earlier calls
    int64_t new_blocks;   // blocks this call completes

    // Complete blocks visible to the token `v` of this call (`v` counts the real tokens, left padding excluded).
    [[nodiscard]] int64_t blocks_visible(int64_t v, int64_t ratio) const {
        return past_blocks + (tail + v + 1) / ratio;
    }
};

inline RowPlan make_row_plan(int64_t key_length, int64_t window, int64_t ratio) {
    const int64_t past = std::max<int64_t>(0, key_length - window);
    RowPlan plan{};
    plan.current = key_length - past;  // == min(window, key_length)
    plan.num_padding = window - plan.current;
    plan.tail = past % ratio;
    plan.past_blocks = past / ratio;
    plan.new_blocks = (plan.tail + plan.current) / ratio;
    return plan;
}

// The blocks of row b: key_length is [B] or [1] (one value shared by all the rows).
inline RowPlan row_plan(const DenseCall& call, const Config& c, size_t b) {
    const auto key_length = call.key_length.ptr<int32_t>()[call.key_length.size(0) == 1 ? 0 : b];
    return make_row_plan(key_length, static_cast<int64_t>(call.q_idx.size(1)), static_cast<int64_t>(c.key_block_size));
}

// The RoPE tables may be shared by all the batches ([1, L, Dr]): batch 0 is read for every row then.
inline size_t rope_batch(const PlainTensor& table, size_t b) {
    return table.size(0) == 1 ? 0 : b;
}

// What the longest row of the call needs: the largest window, the most blocks a row has after the call, the most
// blocks it completes.
struct RowStats {
    int64_t max_current = 0;
    int64_t max_row_blocks = 0;
    int64_t max_new_blocks = 0;
};

inline RowStats row_stats(const DenseCall& call, const Config& c) {
    RowStats stats;
    const auto blocks_out = call.summary_cache.size(2);
    for (size_t b = 0; b < call.q_idx.size(0); b++) {
        const auto plan = row_plan(call, c, b);
        OPENVINO_ASSERT(static_cast<size_t>(plan.past_blocks + plan.new_blocks) <= blocks_out,
                        "QSAIndexer: ",
                        plan.past_blocks + plan.new_blocks,
                        " blocks do not fit the summary_cache output of ",
                        blocks_out,
                        " blocks");
        stats.max_current = std::max(stats.max_current, plan.current);
        stats.max_row_blocks = std::max(stats.max_row_blocks, plan.past_blocks + plan.new_blocks);
        stats.max_new_blocks = std::max(stats.max_new_blocks, plan.new_blocks);
    }
    return stats;
}

// ---- paged op ----

inline size_t num_tokens(const PagedCall& call) {
    return call.q_idx.size(0);
}
inline size_t num_sequences(const PagedCall& call) {
    OPENVINO_ASSERT(call.subsequence_begins.size(0) >= 1,
                    "[CPU] paged_qsa_indexer: subsequence_begins must be non-empty");
    return call.subsequence_begins.size(0) - 1;
}
inline size_t index_query_heads(const PagedCall& call, const Config& c) {
    return call.q_idx.size(1) / c.head_dim;
}
inline size_t index_key_heads(const PagedCall& call, const Config& c) {
    return call.k_idx.size(1) / c.head_dim;
}

// What a call knows about one sequence: its tokens, its page-table row and the index blocks it completes:
// past_blocks = past_len / key_block_size were complete before, new_blocks = (past_len + q_len) / key_block_size -
// past_blocks are completed by it.
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

inline SeqPlan seq_plan(const PagedCall& call, const Config& c, size_t seq) {
    const auto* subsequence_begins = call.subsequence_begins.ptr<int32_t>();
    const auto* block_indices_begins = call.block_indices_begins.ptr<int32_t>();
    const int64_t block_begin = block_indices_begins[seq];
    const int64_t seq_blocks = std::max<int64_t>(block_indices_begins[seq + 1] - block_begin, 0);
    SeqPlan plan{};
    plan.key_block_size = static_cast<int64_t>(c.key_block_size);
    plan.page_size = static_cast<int64_t>(call.raw_k_cache.size(2));
    plan.slots_per_page = plan.page_size / plan.key_block_size;
    plan.token_begin = subsequence_begins[seq];
    plan.q_len = std::max<int64_t>(subsequence_begins[seq + 1] - plan.token_begin, 0);
    plan.past_len = call.past_lens.ptr<int32_t>()[seq];
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

inline SeqStats seq_stats(const PagedCall& call, const Config& c) {
    SeqStats stats;
    const auto* block_indices_begins = call.block_indices_begins.ptr<int32_t>();
    for (size_t seq = 0; seq < num_sequences(call); seq++) {
        OPENVINO_ASSERT(block_indices_begins[seq + 1] > block_indices_begins[seq],
                        "[CPU] paged_qsa_indexer: each sequence must have at least one cache block");
        const auto plan = seq_plan(call, c, seq);
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
// the raw key cache through the shared page table.
inline CacheAddress token_address(const PagedCall& call, const SeqPlan& plan, int64_t logical_pos) {
    const int64_t page_table_idx = plan.block_begin + logical_pos / plan.page_size;
    const auto page = static_cast<size_t>(*call.block_indices.ptr<int32_t>(static_cast<size_t>(page_table_idx)));
    return {page, static_cast<size_t>(logical_pos % plan.page_size)};
}

// Maps a (sequence-local, 0-based) completed-block index to the (page, slot) address into the summary cache /
// the block rope cache.
inline CacheAddress block_address(const PagedCall& call, const SeqPlan& plan, int64_t block_idx) {
    const int64_t page_table_idx = plan.block_begin + (block_idx * plan.key_block_size) / plan.page_size;
    const auto page = static_cast<size_t>(*call.block_indices.ptr<int32_t>(static_cast<size_t>(page_table_idx)));
    return {page, static_cast<size_t>(block_idx % plan.slots_per_page)};
}

// A row of a port as f32: the row itself when the data is f32 (no copy), otherwise converted into `scratch`. All the
// floating-point ports have the precision T (the nodes make sure of it), so no port needs a run-time lookup of its own.
template <typename T, typename... Idx>
const float* row_view(const PlainTensor& t, size_t len, float* scratch, Idx... idx) {
    if constexpr (std::is_same_v<T, float>) {
        return t.ptr<float>(idx...);
    } else {
        cvt_copy(scratch, t.ptr<T>(idx...), 1, len, 0, 0);
        return scratch;
    }
}

}  // namespace ov::Extensions::Cpu::XARCH::qsa_kernel
