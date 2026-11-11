// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// PagedQSAIndexer computation: the per-token math of the QSAIndexer (see qsa_indexer_kernel.hpp) over caches addressed
// through the PagedAttention page table. A call appends the new raw keys, pools the blocks they complete, and keeps
// each token's block_topk best blocks. This header holds what the node and the kernel share.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_parallel.hpp"
#include "nodes/kernels/sparse_attn/qsa_indexer_common.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"

namespace ov::intel_cpu::node::qsa {

class QsaBrgemm;  // x64 only, see qsa_brgemm.hpp

}  // namespace ov::intel_cpu::node::qsa

namespace ov::intel_cpu::node::qsa::paged {

using Config = ov::op::internal::PagedQSAIndexer::Config;

/// @brief Fixed output order of ov::op::internal::PagedQSAIndexer (not part of the op's public API).
struct OutputIdx {
    static constexpr size_t SEL_INDICES = 0;  ///< [tokens, h_sel, block_topk]
    static constexpr size_t SEL_COUNT = 1;    ///< [tokens, h_sel]

    static constexpr size_t COUNT = 2;
};

// The ports of the node, each taken by its own name.
struct PagedPorts {
    using In = ov::op::internal::PagedQSAIndexer::InputIdx;

    PagedPorts(const std::vector<MemoryPtr>& inputs, const std::vector<MemoryPtr>& outputs)
        : q_idx(inputs.at(In::INDEX_QUERY)),
          k_idx(inputs.at(In::RAW_INDEX_KEYS)),
          rope_cos(inputs.at(In::ROPE_COS)),
          rope_sin(inputs.at(In::ROPE_SIN)),
          q_norm_weight(inputs.at(In::Q_NORM_WEIGHT)),
          k_norm_weight(inputs.at(In::K_NORM_WEIGHT)),
          past_lens(inputs.at(In::PAST_LENS)),
          subsequence_begins(inputs.at(In::SUBSEQUENCE_BEGINS)),
          block_indices(inputs.at(In::BLOCK_INDICES)),
          block_indices_begins(inputs.at(In::BLOCK_INDICES_BEGINS)),
          raw_k_cache(inputs.at(In::INDEX_KEY_CACHE)),
          indexer_rope_cache(inputs.at(In::INDEX_BLOCK_ROPE_CACHE)),
          summary_cache(inputs.at(In::BLOCK_INDEX_KEYS_CACHE)),
          sel_indices(outputs.at(OutputIdx::SEL_INDICES)),
          sel_count(outputs.at(OutputIdx::SEL_COUNT)) {}

    PlainTensor q_idx, k_idx, rope_cos, rope_sin, q_norm_weight, k_norm_weight;
    PlainTensor past_lens, subsequence_begins, block_indices, block_indices_begins;
    PlainTensor raw_k_cache, indexer_rope_cache, summary_cache;
    PlainTensor sel_indices, sel_count;
};

// The sizes of a call that follow from the shapes of the ports (the others are in Config).
inline size_t num_tokens(const PagedPorts& p) {
    return p.q_idx.size(0);
}
inline size_t num_sequences(const PagedPorts& p) {
    OPENVINO_ASSERT(p.subsequence_begins.size(0) >= 1, "[CPU] paged_qsa_indexer: subsequence_begins must be non-empty");
    return p.subsequence_begins.size(0) - 1;
}
inline size_t index_query_heads(const PagedPorts& p, const Config& c) {
    return p.q_idx.size(1) / c.indexer_head_dim;
}
inline size_t index_key_heads(const PagedPorts& p, const Config& c) {
    return p.k_idx.size(1) / c.indexer_head_dim;
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

inline SeqPlan seq_plan(const PagedPorts& p, const Config& c, size_t seq) {
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

inline SeqStats seq_stats(const PagedPorts& p, const Config& c) {
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
inline CacheAddress token_address(const PagedPorts& p, const SeqPlan& plan, int64_t logical_pos) {
    const int64_t page_table_idx = plan.block_begin + logical_pos / plan.page_size;
    const auto page = static_cast<size_t>(*p.block_indices.ptr<int32_t>(static_cast<size_t>(page_table_idx)));
    return {page, static_cast<size_t>(logical_pos % plan.page_size)};
}

// Maps a (sequence-local, 0-based) completed-block index to the (page, slot) address into BLOCK_INDEX_KEYS_CACHE /
// INDEX_BLOCK_ROPE_CACHE.
inline CacheAddress block_address(const PagedPorts& p, const SeqPlan& plan, int64_t block_idx) {
    const int64_t page_table_idx = plan.block_begin + (block_idx * plan.key_block_size) / plan.page_size;
    const auto page = static_cast<size_t>(*p.block_indices.ptr<int32_t>(static_cast<size_t>(page_table_idx)));
    return {page, static_cast<size_t>(block_idx % plan.slots_per_page)};
}

// All floating-point ports share one precision, f32, f16 or bf16 (the op validates it). The kernel converts the rows
// it touches, so the compute stays f32 without a whole-tensor Reorder of caches that grow every step.

// The buffers of one worker thread's slice of the scratch; the phases run one after the other, so they share them.
struct ThreadLayout {
    size_t q_normed_off = 0;  // floats: [H_iq, D_idx] normalized + rotated index queries
    size_t row_buf_off;       // floats: [D_idx] a converted row
    size_t summary_buf_off;   // floats: [D_idx] a pooled summary row
    size_t pooled_off;        // floats: [D_idx] the mean of the raw keys of a block
    size_t cos_sin_off;       // floats: [2 * rotary_dim] the interleaved cos / sin row of a block
    size_t cos_tmp_off;       // floats: [rotary_dim] a cos row
    size_t sin_tmp_off;       // floats: [rotary_dim] a sin row
    size_t stage_off;         // floats: [kStageRows, D_idx] summary rows converted to f32 (the scoring)
    size_t score_f32_off;     // floats: [blocks] the scores of the blocks of a token and selection head
    size_t scores_off;        // bytes: [blocks] (score * scale, block), what the top-k selects from
    size_t chosen_off;        // bytes: [block_topk]
    size_t bytes;

    static constexpr size_t kAlign = 64;
    static constexpr size_t kStageRows = 4;  // the rows the scoring converts at a time (qsa::kScoreRows)

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
          chosen_off(rnd_up(scores_off + max_blocks * sizeof(std::pair<float, int32_t>), kAlign)),
          bytes(rnd_up(chosen_off + block_topk * sizeof(int32_t), kAlign)) {}
};

// The scratch of one call, one allocation (a second createScratchPadMem() could free the first's memory):
//   [ token -> sequence array, f32 norm weights | brgemm scorer's shared part | one slice per worker thread ]
// The scorer lays out its own slices over the same space, after the kernel's phases.
struct PagedScratchLayout {
    ThreadLayout thread;
    size_t norm_weights_offset;
    size_t scorer_shared_offset;
    size_t threads_offset;
    size_t thread_bytes;
    size_t total_bytes;

    // `scorer_shared_bytes` / `scorer_thread_bytes`: what the brgemm scorer needs (0: there is none).
    PagedScratchLayout(const PagedPorts& p,
                       const Config& c,
                       int64_t max_blocks,
                       size_t scorer_shared_bytes,
                       size_t scorer_thread_bytes,
                       size_t num_threads)
        : thread(index_query_heads(p, c),
                 c.indexer_head_dim,
                 c.indexer_rotary_dim,
                 static_cast<size_t>(max_blocks),
                 c.block_topk),
          norm_weights_offset(rnd_up(num_tokens(p) * sizeof(int32_t), ThreadLayout::kAlign)),
          scorer_shared_offset(
              rnd_up(norm_weights_offset + 2 * c.indexer_head_dim * sizeof(float), ThreadLayout::kAlign)),
          threads_offset(rnd_up(scorer_shared_offset + scorer_shared_bytes, ThreadLayout::kAlign)),
          thread_bytes(std::max(thread.bytes, scorer_thread_bytes)),
          total_bytes(threads_offset + num_threads * thread_bytes) {}
};

}  // namespace ov::intel_cpu::node::qsa::paged

namespace ov::Extensions::Cpu::XARCH {

// The single cross-compiled entry of the PagedQSAIndexer node: AVX-512, AVX2 or plain C++ by the CPU. `scratch` holds
// layout.total_bytes. `brgemm` scores the blocks on brgemm (the layout was planned for it); when it is null the kernel
// scores them itself.
void paged_qsa_indexer_compute(const ov::intel_cpu::node::qsa::paged::PagedPorts& ports,
                               const ov::op::internal::PagedQSAIndexer::Config& config,
                               const ov::intel_cpu::node::qsa::paged::PagedScratchLayout& layout,
                               uint8_t* scratch,
                               const ov::intel_cpu::node::qsa::QsaBrgemm* brgemm,
                               const ov::intel_cpu::CpuParallelPtr& cpu_parallel);

}  // namespace ov::Extensions::Cpu::XARCH
