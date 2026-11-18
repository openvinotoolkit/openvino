// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// The computation of the QSAIndexer node. One call, per row b of the batch:
//   1. appends the window of L tokens (q_idx [B, L, H_iq, D], k_idx [B, L, H_ik, D]) to the key state: every `ratio`
//      keys complete a block, and the index key of the block (pooled keys -> RMSNorm -> partial RoPE) is stored in
//      summary_cache [B, H_ik, NB, D];
//   2. scores, for every token of the window, the complete blocks it can see (q . index key, summed over the query
//      heads of a selection head) and keeps the block_topk best block ids: sel_indices [B, H_sel, L, block_topk],
//      sel_count [B, H_sel, L].
// The kernel (qsa_indexer_kernel.cpp) is cross-compiled: one entry for every instruction set. This header holds what
// the node and the kernel share: the ports, the plan of a call and the layout of its scratch.

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
#include "transformations/cpu_opset/common/op/qsa_indexer.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"

namespace ov::intel_cpu::node::qsa {

class QsaBrgemm;  // x64 only, see qsa_brgemm.hpp

using Config = ov::intel_cpu::QSAIndexerNode::Config;

// The ports of the node, each taken by its own name.
struct DensePorts {
    using In = ov::intel_cpu::QSAIndexerNode::InputIdx;
    using Out = ov::intel_cpu::QSAIndexerNode::OutputIdx;

    DensePorts(const std::vector<MemoryPtr>& inputs, const std::vector<MemoryPtr>& outputs)
        : q_idx(inputs.at(In::Q_IDX)),
          k_idx(inputs.at(In::K_IDX)),
          rope_cos(inputs.at(In::ROPE_COS)),
          rope_sin(inputs.at(In::ROPE_SIN)),
          key_length(inputs.at(In::KEY_LENGTH)),
          q_norm_weight(inputs.at(In::Q_NORM_WEIGHT)),
          k_norm_weight(inputs.at(In::K_NORM_WEIGHT)),
          // The state inputs are read-only; the updated state lives in the matching outputs.
          past_raw_k_in(inputs.at(In::PAST_INCOMPLETE_RAW_K)),
          past_rope_in(inputs.at(In::PAST_INCOMPLETE_ROPE)),
          summary_cache_in(inputs.at(In::SUMMARY_CACHE)),
          summary_cache(outputs.at(Out::SUMMARY_CACHE)),
          past_raw_k(outputs.at(Out::PAST_INCOMPLETE_RAW_K)),
          past_rope(outputs.at(Out::PAST_INCOMPLETE_ROPE)),
          sel_indices(outputs.at(Out::SEL_INDICES)),
          sel_count(outputs.at(Out::SEL_COUNT)) {}

    PlainTensor q_idx, k_idx, rope_cos, rope_sin, key_length, q_norm_weight, k_norm_weight;
    PlainTensor past_raw_k_in, past_rope_in, summary_cache_in;
    PlainTensor summary_cache, past_raw_k, past_rope, sel_indices, sel_count;
};

// The blocks of row b: key_length is [B] or [1] (one value shared by all the rows).
inline RowPlan row_plan(const DensePorts& p, const Config& c, size_t b) {
    const auto key_length = p.key_length.ptr<int32_t>()[p.key_length.size(0) == 1 ? 0 : b];
    return make_row_plan(key_length, static_cast<int64_t>(p.q_idx.size(1)), static_cast<int64_t>(c.compress_ratio));
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

inline RowStats row_stats(const DensePorts& p, const Config& c) {
    RowStats stats;
    const auto blocks_out = p.summary_cache.size(2);
    for (size_t b = 0; b < p.q_idx.size(0); b++) {
        const auto plan = row_plan(p, c, b);
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

// The buffers of one worker thread's slice of the scratch. The phases run one after the other, so they share the
// floats; the (score, block) pairs and the chosen ids are for the token scorer.
struct ThreadLayout {
    size_t row_buf_off;       // floats: [D_idx] a converted row (after the [H_iq, D_idx] normalized queries)
    size_t rope_cos_sin_off;  // floats: [2 * rotary_dim] cos / sin of the first token of a block (of the token)
    size_t stage_off;         // floats: [kStageRows, D_idx] summary rows converted to f32 (the scoring)
    size_t score_f32_off;     // floats: [blocks] the scores of the blocks of a token and selection head
    size_t scores_off;        // bytes: [blocks] (score * scale, block), what the top-k selects from
    size_t chosen_off;        // bytes: [block_topk]
    size_t bytes;

    static constexpr size_t kAlign = 64;
    static constexpr size_t kStageRows = 4;  // the rows the scoring converts at a time (qsa::kScoreRows)

    ThreadLayout(size_t H_iq, size_t D_idx, size_t rotary_dim, size_t max_blocks, size_t block_topk)
        : row_buf_off(H_iq * D_idx),
          rope_cos_sin_off(row_buf_off + D_idx),
          stage_off(rope_cos_sin_off + 2 * rotary_dim),
          score_f32_off(stage_off + kStageRows * D_idx),
          scores_off(rnd_up((score_f32_off + max_blocks) * sizeof(float), kAlign)),
          chosen_off(rnd_up(scores_off + max_blocks * sizeof(std::pair<float, int32_t>), kAlign)),
          bytes(rnd_up(chosen_off + block_topk * sizeof(int32_t), kAlign)) {}
};

// The scratch of one call, one allocation (a second createScratchPadMem() could free the first's memory):
//   [ f32 q / k norm weights | what the brgemm scorer shares between the threads | one slice per worker thread ]
// The scorer lays out its own slices over the same space, after the phases of the kernel have used theirs.
struct DenseScratchLayout {
    ThreadLayout thread;
    size_t scorer_shared_offset;
    size_t threads_offset;
    size_t thread_bytes;
    size_t total_bytes;

    // `scorer_shared_bytes` / `scorer_thread_bytes`: what the brgemm scorer needs (0: there is none).
    DenseScratchLayout(const DensePorts& p,
                       const Config& c,
                       size_t scorer_shared_bytes,
                       size_t scorer_thread_bytes,
                       size_t num_threads)
        : thread(p.q_idx.size(2), p.q_idx.size(3), c.indexer_rotary_dim, p.summary_cache.size(2), c.block_topk),
          scorer_shared_offset(rnd_up(2 * p.q_idx.size(3) * sizeof(float), ThreadLayout::kAlign)),
          threads_offset(rnd_up(scorer_shared_offset + scorer_shared_bytes, ThreadLayout::kAlign)),
          thread_bytes(std::max(thread.bytes, scorer_thread_bytes)),
          total_bytes(threads_offset + num_threads * thread_bytes) {}
};

}  // namespace ov::intel_cpu::node::qsa

namespace ov::Extensions::Cpu::XARCH {

// The single cross-compiled entry of the QSAIndexer node: AVX-512, AVX2 or plain C++ by the CPU. `scratch` holds
// layout.total_bytes. `brgemm` scores the blocks on brgemm (the layout was planned for it); when it is null the kernel
// scores them itself.
void qsa_indexer_compute(const ov::intel_cpu::node::qsa::DensePorts& ports,
                         const ov::intel_cpu::QSAIndexerNode::Config& config,
                         const ov::intel_cpu::node::qsa::DenseScratchLayout& layout,
                         uint8_t* scratch,
                         const ov::intel_cpu::node::qsa::QsaBrgemm* brgemm,
                         const ov::intel_cpu::CpuParallelPtr& cpu_parallel);

}  // namespace ov::Extensions::Cpu::XARCH
