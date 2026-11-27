// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

// The plan of a call, the row arithmetic and the block selection of the QSAIndexer and PagedQSAIndexer kernels, on the
// simd layer (nodes/kernels/simd). It is included by cross-compiled sources only, so the functions are compiled once
// per instruction set (AVX-512, AVX2, plain C++) in the namespace XARCH; nothing of it is visible to the nodes.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include "nodes/kernels/scaled_attn/common.hpp"
#include "nodes/kernels/simd/simd_loop.hpp"
#include "nodes/kernels/sparse_attn/qsa_executor.hpp"
#include "nodes/kernels/sparse_attn/sparse_attn_common.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH::qsa_kernel {

using ov::Extensions::Cpu::XARCH::ScratchPartition;

// The vector type of a loop body: the full-width one in the main loop, the scalar one in the tail.
template <typename Lanes>
using vec_of = simd::f32_t<Lanes::isa_tag::value>;

// ---- rows of the summary / key / query tensors: f32, f16 or bf16 in memory, f32 in the registers ----

// Read-only f32 view of a row of `n` values: the row itself for f32, otherwise the row converted into `buf`.
template <typename T>
const float* row_as_f32(const T* src, size_t n, float* buf) {
    if constexpr (std::is_same_v<T, float>) {
        return src;
    } else {
        cvt_copy(buf, src, 1, n, 0, 0);
        return buf;
    }
}

// dst[0, n) = src[0, n) rounded to T.
template <typename T>
void store_row(const float* src, T* dst, size_t n) {
    cvt_copy(dst, src, 1, n, 0, 0);
}

// ---- row arithmetic ----

// sum of x[i]^2
inline float sum_squares(const float* x, size_t n) {
    return simd::simd_loop_reduce<2>(
        static_cast<int>(n),
        [&](int j, simd::f32& acc) {
            const auto v = simd::load<simd::f32>(x + j);
            acc = fmadd(v, v, acc);
        },
        [&](int j, float& tail) {
            tail += x[j] * x[j];
        });
}

// dst[i] += src[i]
inline void add_inplace(float* dst, const float* src, size_t n) {
    simd::simd_loop(static_cast<int>(n), [&](int j, auto lanes) {
        using V = vec_of<decltype(lanes)>;
        store(simd::load<V>(dst + j) + simd::load<V>(src + j), dst + j);
    });
}

// x[i] *= factor
inline void scale_inplace(float* x, float factor, size_t n) {
    simd::simd_loop(static_cast<int>(n), [&](int j, auto lanes) {
        using V = vec_of<decltype(lanes)>;
        store(simd::load<V>(x + j) * V(factor), x + j);
    });
}

// One row of D = `dim` channels in, one row out (src [dim], weight [dim], cos / sin [rotary_dim], dst [dim]):
//   1. RMSNorm   y = src / sqrt(mean(src^2) + eps) * weight
//   2. RoPE      the split-half (rotate-half) RoPE of HF Transformers over the first `rotary_dim` channels of y:
//                y_rot = y * cos + rotate_half(y) * sin, with rotate_half(y) = concat(-y[half:], y[:half]);
//                the remaining dim - rotary_dim channels are kept as they are.
// cos / sin (the rope_cos / rope_sin inputs of the QSA ops) are already duplicated across the two halves. `dst` may
// be `src`.
// The steps of a query head are optional: `do_norm` / `do_rope` are false while the RMSNorm / RoPE of the query are
// still done by the graph in front of the op, and the row then only passes through that step. The flags are fixed per
// node, so the branches are loop-invariant and well predicted (a template argument per flag measured about 1 ns per
// row faster but made the kernel objects 2-3.5x larger). The rows of the keys always take both steps.
inline void rms_rope(const float* src,
                     float* dst,
                     const float* weight,
                     size_t dim,
                     size_t rotary_dim,
                     float eps,
                     const float* cos,
                     const float* sin,
                     bool do_norm = true,
                     bool do_rope = true) {
    if (do_norm) {
        const float inv_rms = 1.0F / std::sqrt(sum_squares(src, dim) / static_cast<float>(dim) + eps);
        simd::simd_loop(static_cast<int>(dim), [&](int j, auto lanes) {
            using V = vec_of<decltype(lanes)>;
            store(simd::load<V>(src + j) * V(inv_rms) * simd::load<V>(weight + j), dst + j);
        });
    } else if (dst != src) {
        std::copy(src, src + dim, dst);
    }
    if (do_rope) {
        // the pair of channels (j, j + half) is rotated by the angle of the channel j
        const auto half = static_cast<int>(rotary_dim / 2);
        simd::simd_loop(half, [&](int j, auto lanes) {
            using V = vec_of<decltype(lanes)>;
            const V x0 = simd::load<V>(dst + j);
            const V x1 = simd::load<V>(dst + j + half);
            store(x0 * simd::load<V>(cos + j) - x1 * simd::load<V>(sin + j), dst + j);
            store(x1 * simd::load<V>(cos + j + half) + x0 * simd::load<V>(sin + j + half), dst + j + half);
        });
    }
}

// ---- block scores ----

// The rows the scoring handles together: each loaded chunk of a query head is used by all of them, and their
// accumulators are independent chains (the FMA latency would otherwise bound the loop).
inline constexpr size_t kScoreRows = 4;

// scores[i] += sum over the G query heads h of max(0, q[h] . row[i]) for R rows of D f32 channels (`row[r]` is the
// row r), q [G, D] f32.
template <size_t R>
void score_rows(const float* q, const float* const* row, size_t G, size_t D, float* scores) {
    constexpr auto W = static_cast<size_t>(simd::f32::width);
    float sum[R] = {};
    for (size_t h = 0; h < G; h++) {
        const float* const qh = q + h * D;
        simd::f32 acc[R][2];
        size_t d = 0;
        for (; d + 2 * W <= D; d += 2 * W) {
            const auto q0 = simd::load<simd::f32>(qh + d);
            const auto q1 = simd::load<simd::f32>(qh + d + W);
            for (size_t r = 0; r < R; r++) {
                acc[r][0] = fmadd(q0, simd::load<simd::f32>(row[r] + d), acc[r][0]);
                acc[r][1] = fmadd(q1, simd::load<simd::f32>(row[r] + d + W), acc[r][1]);
            }
        }
        for (; d + W <= D; d += W) {
            const auto q0 = simd::load<simd::f32>(qh + d);
            for (size_t r = 0; r < R; r++) {
                acc[r][0] = fmadd(q0, simd::load<simd::f32>(row[r] + d), acc[r][0]);
            }
        }
        float tail[R] = {};
        for (; d < D; d++) {
            for (size_t r = 0; r < R; r++) {
                tail[r] += qh[d] * row[r][d];
            }
        }
        for (size_t r = 0; r < R; r++) {
            sum[r] += std::max(0.0F, reduce(acc[r][0] + acc[r][1]) + tail[r]);
        }
    }
    for (size_t r = 0; r < R; r++) {
        scores[r] += sum[r];
    }
}

// scores[i] += sum over the G query heads h of max(0, q[h] . rows[i]), for the n summary rows ([n, D] of the type T)
// of one index key head and its G query heads q ([G, D] f32). The rows are converted to f32 once (they are shared by
// the heads) into `stage`, kScoreRows * D floats (not touched for f32 rows).
template <typename T>
void score_run(const float* q, const T* rows, size_t n, size_t G, size_t D, float* scores, float* stage) {
    size_t i = 0;
    for (; i + kScoreRows <= n; i += kScoreRows) {
        const float* row[kScoreRows];
        for (size_t r = 0; r < kScoreRows; r++) {
            row[r] = row_as_f32(rows + (i + r) * D, D, stage + r * D);
        }
        score_rows<kScoreRows>(q, row, G, D, scores + i);
    }
    for (; i < n; i++) {
        const float* const row[1] = {row_as_f32(rows + i * D, D, stage)};
        score_rows<1>(q, row, G, D, scores + i);
    }
}

// ---- block selection ----

// The (score, block id) of a complete block.
using BlockScore = std::pair<float, int32_t>;

// A NaN / Inf in the query, keys or norm weights poisons a whole score row. The scores are sanitized before they are
// ordered: finite, within +-kScoreLimit (no overflow in the row range or the bucketing of the brgemm scorer), NaN as
// the lowest score. The ordering is then a strict weak one, and every scoring path picks the same blocks.
constexpr float kScoreLimit = std::numeric_limits<float>::max() / 2;

inline float sanitize_score(float score) {
    // the negated comparison also catches NaN
    return score > -kScoreLimit ? std::min(score, kScoreLimit) : -kScoreLimit;
}

// Writes the ids of the k = min(block_topk, num_blocks) blocks of the highest scores[0, num_blocks) to `chosen` and
// returns k. Equal scores: the lower block id wins. The blocks are in ascending id order. `scores` is reordered.
inline size_t select_topk(BlockScore* scores, int32_t* chosen, size_t num_blocks, size_t block_topk) {
    const size_t k = std::min(block_topk, num_blocks);
    for (size_t i = 0; i < num_blocks; i++) {
        scores[i].first = sanitize_score(scores[i].first);
    }
    std::partial_sort(scores, scores + k, scores + num_blocks, [](const BlockScore& lhs, const BlockScore& rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first > rhs.first;
        }
        return lhs.second < rhs.second;
    });
    for (size_t i = 0; i < k; i++) {
        chosen[i] = scores[i].second;
    }
    std::sort(chosen, chosen + k);
    return k;
}

// ======== the plan of a call ========
// What the kernels know about a call: the plan of every row (a batch entry of the dense op, a sequence of the paged
// one) that follows from the lengths in the call.

using ov::Extensions::Cpu::qsa::Config;
using ov::Extensions::Cpu::qsa::IndexerArgs;
using ov::Extensions::Cpu::qsa::PagedIndexerArgs;
using ov::intel_cpu::PlainTensor;

// ---- dense op ----

// What one entry of the batch contributes to a call. Every quantity is derived from the entry's own key_length and the
// window length L of the call alone -- no cross-call running counter is kept.
struct BatchEntryPlan {
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

// The plan of batch entry b: key_length is [B] or [1] (one value shared by all the entries); the window is the L of the
// call.
inline BatchEntryPlan batch_entry_plan(const IndexerArgs& call, const Config& c, size_t b) {
    const int64_t key_length = call.key_length.ptr<int32_t>()[call.key_length.size(0) == 1 ? 0 : b];
    const auto window = static_cast<int64_t>(call.q_idx.size(1));
    const auto ratio = static_cast<int64_t>(c.key_block_size);
    const int64_t past = std::max<int64_t>(0, key_length - window);
    BatchEntryPlan plan{};
    plan.current = key_length - past;  // == min(window, key_length)
    plan.num_padding = window - plan.current;
    plan.tail = past % ratio;
    plan.past_blocks = past / ratio;
    plan.new_blocks = (plan.tail + plan.current) / ratio;
    return plan;
}

// The RoPE tables may be shared by all the batches ([1, L, Dr]): batch 0 is read for every entry then.
inline size_t rope_batch(const PlainTensor& table, size_t b) {
    return table.size(0) == 1 ? 0 : b;
}

// ---- paged op ----

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

inline SeqPlan seq_plan(const PagedIndexerArgs& call, const Config& c, size_t seq) {
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

// A row of a paged cache: the physical page (an entry of block_indices) and the slot inside it.
struct CacheAddress {
    size_t page;
    size_t slot;
};

// Maps a logical token position (relative to the sequence's own 0..S history) to the (page, slot) address into
// the raw key cache through the shared page table.
inline CacheAddress token_address(const PagedIndexerArgs& call, const SeqPlan& plan, int64_t logical_pos) {
    const int64_t page_table_idx = plan.block_begin + logical_pos / plan.page_size;
    const auto page = static_cast<size_t>(*call.block_indices.ptr<int32_t>(static_cast<size_t>(page_table_idx)));
    return {page, static_cast<size_t>(logical_pos % plan.page_size)};
}

// Maps a (sequence-local, 0-based) completed-block index to the (page, slot) address into the summary cache /
// the block rope cache.
inline CacheAddress block_address(const PagedIndexerArgs& call, const SeqPlan& plan, int64_t block_idx) {
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
