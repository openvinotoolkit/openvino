// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>

#include "nodes/kernels/scaled_attn/common.hpp"
#include "nodes/kernels/scaled_attn/softmax_kernel.hpp"

namespace ov::Extensions::Cpu {

/// @brief Query heads that share both K/V rows and the selected blocks, so a work item can process them together:
/// q_head_pack_size = gcd(q_heads_per_k_head, q_heads_per_sel_head) with q_heads_per_k_head = q_head_num / k_head_num
/// and q_heads_per_sel_head = q_head_num / sel_head_num (query head h reads K head h / q_heads_per_k_head and uses
/// selection head h / q_heads_per_sel_head). sel_head_num of 1 or k_head_num gives the whole K-head group, q_head_num
/// gives 1.
inline size_t max_q_head_pack_size(size_t q_head_num, size_t k_head_num, size_t sel_head_num) {
    return std::gcd(q_head_num / k_head_num, q_head_num / sel_head_num);
}

/// @brief Number of rows of [kpos0, kpos0 + n) a query may attend: the rows from `first_valid` (the start of the real
/// history of a left-padded sequence) up to `valid_end` (exclusive).
inline int64_t visible_rows(int64_t kpos0, int64_t n, int64_t first_valid, int64_t valid_end) {
    return kpos0 < first_valid ? 0 : std::max<int64_t>(std::min(n, valid_end - kpos0), 0);
}

/// @brief First position of the causal tail of the query at `token_pos`: its incomplete last block of key_block_size
/// rows, counted from `pad` (the physical start of the history). The tail is [tail_start, token_pos], empty at a block
/// boundary. sel_indices / sel_count only carry complete blocks, so every causal consumer attends the tail itself.
inline int64_t causal_tail_start(int64_t token_pos, int64_t pad, int64_t key_block_size) {
    return pad + ((token_pos - pad + 1) / key_block_size) * key_block_size;
}

}  // namespace ov::Extensions::Cpu

namespace ov::Extensions::Cpu::XARCH {

/// @brief Starts the attention of a head pack: zero accumulators, empty online softmax.
inline void init_pack_state(float* acc, float* running_max, float* running_sum, size_t pack_size, size_t v_head_size) {
    std::fill_n(acc, pack_size * v_head_size, 0.0F);
    std::fill_n(running_max, pack_size, -std::numeric_limits<float>::infinity());
    std::fill_n(running_sum, pack_size, 0.0F);
}

/// @brief Online-softmax step of one head over scores[0, rows): rescales the running state (maximum, sum and the
/// accumulator acc[v_head_size]) to the new maximum and turns the scores into weights in place.
inline void online_softmax_step(float* scores,
                                size_t rows,
                                float scale,
                                float& running_max,
                                float& running_sum,
                                float* acc,
                                size_t v_head_size) {
    float scaled_max = std::numeric_limits<float>::lowest();
    scale_add2_reduce_max<false, false, false, false, float>(scores,
                                                             scale,
                                                             nullptr,
                                                             nullptr,
                                                             nullptr,
                                                             false,
                                                             rows,
                                                             0.0F,
                                                             scaled_max,
                                                             nullptr,
                                                             0);
    const float new_max = std::max(running_max, scaled_max);
    if (new_max > running_max) {
        // corr == 0 for the first run (running_max == -inf), where acc and running_sum are still zero
        const float corr = std::exp(running_max - new_max);
        running_sum *= corr;
        for (size_t d = 0; d < v_head_size; d++) {
            acc[d] *= corr;
        }
        running_max = new_max;
    }
    float sum = 0.0F;
    exp_reduce_sum(scores, new_max, rows, sum);
    running_sum += sum;
}

/// @brief Ends the attention of a head pack: divides every head's accumulator by its softmax sum (0 when nothing was
/// visible) and stores it as T to out_row(head), the row of v_head_size elements of the output.
template <typename OutRow>
void store_normalized_pack(float* acc,
                           const float* running_sum,
                           size_t pack_size,
                           size_t v_head_size,
                           const OutRow& out_row) {
    for (size_t head = 0; head < pack_size; head++) {
        float* head_acc = acc + head * v_head_size;
        const float inv_sum = running_sum[head] > 0.0F ? 1.0F / running_sum[head] : 0.0F;
        for (size_t d = 0; d < v_head_size; d++) {
            head_acc[d] *= inv_sum;
        }
        cvt_copy(out_row(head), head_acc, 1, v_head_size, 0, 0);
    }
}

}  // namespace ov::Extensions::Cpu::XARCH

// The part of the AMX tile attention (scaled_attn/attn_brgemm_tiles.hpp) that knows about sparsity: the selected blocks
// of an item and how they become runs of rows for gather_run(). Only the cross-compiled kernels use it.
#if defined(OPENVINO_ARCH_X86_64) && defined(HAVE_AVX512F)

#    include "nodes/kernels/scaled_attn/attn_brgemm_tiles.hpp"

namespace ov::Extensions::Cpu::XARCH {

/// @brief The selection of an item: the selected blocks (ids of blocks of key_block_size rows, in any order) and the
/// causal tail (the incomplete last block of the query, `tail_rows` rows from `tail_start`).
struct BrgemmSelection {
    const int32_t* block_ids;
    size_t nblocks;
    int64_t first_pos;  ///< position of block 0
    int64_t key_block_size;
    int64_t tail_start;
    int64_t tail_rows;
};

/// @brief Attention of the pack's heads over the item's whole selection with AMX matrix products: the rows are gathered
/// into tiles (the layout specific part is the row source) and every full tile, then the last partial one, goes through
/// attention_over_tile(), whose online softmax carries the state across the tiles. It leaves the unnormalized attention
/// in pack.acc and the softmax denominators in pack.running_sum (all 0 when nothing is visible); the caller normalizes
/// and stores the output.
template <typename T, typename RowSource>
void attention_over_selection_tiles(const RowSource& src,
                                    const BrgemmPack& pack,
                                    const BrgemmBuffers<T>& b,
                                    const BrgemmSelection& sel) {
    TileState tile;
    for (size_t i = 0; i < sel.nblocks; i++) {
        gather_run<T>(src, pack, b, tile, sel.first_pos + sel.block_ids[i] * sel.key_block_size, sel.key_block_size);
    }
    gather_run<T>(src, pack, b, tile, sel.tail_start, sel.tail_rows);
    flush_tile<T>(pack, b, tile);
}

}  // namespace ov::Extensions::Cpu::XARCH

#endif  // OPENVINO_ARCH_X86_64 && HAVE_AVX512F
