// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

// Block scoring and top-k selection of the QSAIndexer and the PagedQSAIndexer on brgemm (x64). Included by the
// cross-compiled kernels only, so it is compiled per instruction set in the namespace XARCH, like the rest of them; the
// brgemm kernels themselves (BrgemmKernel) are compiled once, the configuration and the scratch plan are in
// qsa_brgemm.hpp.
//
// A "row" is one sequence of the call (a batch entry of the dense node, a sequence of the paged one); the physical
// layouts differ and are hidden behind `Source` (see below). Notation, per row: L tokens of the call, H_iq index query
// heads, H_ik index key heads ("K heads"), G = H_iq / H_ik, H_sel selection heads, D head dim, Dr rotary dim, NB
// complete key blocks.
// Logical inputs: raw query [L, H_iq, D], RoPE tables, q_norm_weight, summary_cache [H_ik, NB, D] (one index key per
// block). Outputs: the chosen block ids and their count per token and selection head.
//
// Per query token (the function that does it in brackets):
//   1. RMSNorm of every query head                                                      (prepare_query_slices)
//   2. RoPE of its first Dr channels: the rotated query q_rot[h], [D]                   (prepare_query_slices)
//   3. dot[h, blk] = q_rot[h] . summary_cache[h / G, blk]                               (score_key_chunks)
//   4. score[sel_head, blk] = scale * sum over the heads of the selection head of max(0, dot[h, blk])
//                                                                                       (score_key_chunks)
//   5. top-k: the block_topk best blocks among those visible to the token               (select_token)
//
// Steps 1-2 run once per query slice (the tokens of one brgemm call) and give its query tile [query_rows, D],
// query_rows = G x tokens of the slice. Steps 3-4 run per key chunk (32 blocks): one brgemm per slice,
//   C[query_rows, 32] = max(0, query_tile x blocks[32, D]^T),
// with the blocks packed once per chunk and K head; ReLU is a post-op and the sum over the heads is plain code. A
// single token (decode) swaps the operands, C[32, G], so the blocks are read in place.
// A work item is `tile_tokens` tokens x a range of key chunks; prefill items select right away, decode items split the
// chunks among threads and select in a separate pass.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "cpu_parallel.hpp"
#include "nodes/kernels/sparse_attn/qsa_brgemm.hpp"
#include "nodes/kernels/sparse_attn/qsa_kernel_utils.hpp"
#include "nodes/kernels/x64/brgemm_kernel.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"

namespace ov::Extensions::Cpu::XARCH::qsa {

using ov::intel_cpu::AmxTileState;
using ov::intel_cpu::BrgemmKernel;
using ov::intel_cpu::CpuParallel;
using ov::intel_cpu::div_up;
using ov::intel_cpu::rnd_up;
using ov::intel_cpu::node::qsa::kScoreBlockRows;
using ov::intel_cpu::node::qsa::kSelectBuckets;
using ov::intel_cpu::node::qsa::QsaBrgemm;

// Step 5: writes the (ascending) ids of the k blocks of the highest scores[0, n) to `chosen`, n > k > 0 (scores: the
// [NB] row of one token and selection head). Equal scores: the lower block id wins. The k-th largest score is found by
// a bucket selection: the scores are spread over kBuckets buckets of equal width between the smallest and the largest
// one, a histogram finds the bucket that holds the k-th score, and the few scores of that bucket are partially sorted
// (a skewed distribution only makes that bucket bigger). `candidates` (n + 1 floats) and `histogram` (kSelectBuckets
// values) are scratch; the passes are branch-free as which scores qualify is as good as random.
inline void select_top_blocks(const float* scores,
                              size_t n,
                              size_t k,
                              float* candidates,
                              uint32_t* histogram,
                              int32_t* chosen) {
    // smallest and largest score
    vfloat vlow = vset1(scores[0]);
    vfloat vhigh = vlow;
    size_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
        const vfloat v = vload(scores + i);
        vlow = vmin(vlow, v);
        vhigh = vmax(vhigh, v);
    }
    float low = vreduce_min(vlow);
    float high = vreduce_max(vhigh);
    for (; i < n; i++) {
        low = std::min(low, scores[i]);
        high = std::max(high, scores[i]);
    }

    constexpr auto max_bucket = static_cast<int32_t>(kSelectBuckets) - 1;
    const float to_bucket = high > low ? static_cast<float>(kSelectBuckets) / (high - low) : 0.0F;
    // The bucket of a score, the same arithmetic as the vector one (vbucket) so that both passes agree on it.
    const auto bucket_of = [&](float score) {
        return std::min(max_bucket, static_cast<int32_t>((score - low) * to_bucket));
    };
    const vfloat vlow_all = vset1(low);
    const vfloat vto_bucket = vset1(to_bucket);

    // histogram: the buckets of a vector of scores are computed at once, the counters are bumped one by one
    std::fill_n(histogram, kSelectBuckets, 0U);
    alignas(64) int32_t buckets[kLanes];
    for (i = 0; i + kLanes <= n; i += kLanes) {
        vstore_buckets(vload(scores + i), vlow_all, vto_bucket, max_bucket, buckets);
        for (const int32_t bucket : buckets) {
            histogram[bucket]++;
        }
    }
    for (; i < n; i++) {
        histogram[bucket_of(scores[i])]++;
    }
    // the bucket of the k-th largest score, and how many scores are above it
    size_t above = 0;
    int bucket = max_bucket;
    while (above + histogram[bucket] < k) {
        above += histogram[bucket--];
    }
    // the scores of that bucket
    size_t candidate_count = 0;
    for (i = 0; i + kLanes <= n; i += kLanes) {
        candidate_count += vcompress_store(candidates + candidate_count,
                                           vload(scores + i),
                                           vbucket_mask(vload(scores + i), vlow_all, vto_bucket, max_bucket, bucket));
    }
    for (; i < n; i++) {
        candidates[candidate_count] = scores[i];
        candidate_count += bucket_of(scores[i]) == bucket ? 1 : 0;
    }
    const size_t rank = k - above;  // 1-based rank of the k-th score among the scores of its bucket
    std::nth_element(candidates, candidates + (rank - 1), candidates + candidate_count, std::greater<>());
    const float threshold = candidates[rank - 1];
    size_t greater = above;
    for (i = 0; i < candidate_count; i++) {
        greater += candidates[i] > threshold ? 1 : 0;
    }

    // Blocks above the threshold, then as many blocks of the threshold score as are missing, the lowest ids first.
    // Collected ascending in the (no longer needed) candidate buffer, as the stores past the end of the chosen
    // blocks are overwritten by the next one. A vector of scores without a tie at the threshold (the usual case)
    // takes the blocks above it at once; one with ties goes block by block.
    auto* const collected = reinterpret_cast<int32_t*>(candidates);
    size_t equal_left = k - greater;
    size_t written = 0;
    const vfloat vthreshold = vset1(threshold);
    for (i = 0; i + kLanes <= n; i += kLanes) {
        const vfloat v = vload(scores + i);
        const uint32_t above_mask = vcmp_gt(v, vthreshold);
        const uint32_t equal_mask = vcmp_eq(v, vthreshold);
        if (equal_mask != 0 && equal_left > 0) {
            for (size_t lane = 0; lane < kLanes; lane++) {
                const bool take_equal = ((equal_mask >> lane) & 1U) != 0 && equal_left > 0;
                equal_left -= take_equal ? 1 : 0;
                collected[written] = static_cast<int32_t>(i + lane);
                written += (((above_mask >> lane) & 1U) != 0 || take_equal) ? 1 : 0;
            }
        } else {
            written += vcompress_ids(collected + written, static_cast<int32_t>(i), above_mask);
        }
    }
    for (; i < n; i++) {
        const bool take_equal = scores[i] == threshold && equal_left > 0;
        equal_left -= take_equal ? 1 : 0;
        collected[written] = static_cast<int32_t>(i);
        written += (scores[i] > threshold || take_equal) ? 1 : 0;
    }
    std::copy_n(collected, k, chosen);
}

// Orders `count` chosen block ids by descending score, equal scores by ascending block id: the order of
// SparseAttentionIndexer-17. `scores` is the [NB] row of the token and selection head.
inline void order_blocks_by_score(const float* scores, int32_t* blocks, size_t count) {
    std::sort(blocks, blocks + count, [scores](int32_t a, int32_t b) {
        return scores[a] != scores[b] ? scores[a] > scores[b] : a < b;
    });
}

// The computation of a call on the scorer of the node (QsaBrgemm); `Source` is described below.
//
// `Source` is the layout of the call: it gives the runner the tokens of the rows, the query and RoPE rows, the blocks
// and the place of the results, so the runner does not know whether they are dense tensors or paged caches. A row is a
// sequence of the batch, `v` a token of its call counted from the first real token (`b`: row, `h`: query head,
// `key_head`: K head, `sel_head`: selection head). `Source` provides (all const, called from several threads):
//   using Element = T                       the precision of the summary blocks (f32, f16 or bf16), also the precision
//                                           the query tile is rounded to
//   int64_t tokens(b)                       the real tokens of the call of the row
//   int64_t blocks_visible(b, v)            the complete blocks visible to the token v
//   int64_t blocks(b)                       the complete blocks of the row after the call (>= those visible to any
//                                           token)
//   const float* query_row(b, v, h, buf)    q[h] [D] as f32: a view, or converted into `buf` [D]
//   const float* rope_cos(b, v, buf)        cos [Dr] as f32 of the token (a view, or converted into `buf` [Dr])
//   const float* rope_sin(b, v, buf)        the same for sin
//   const float* q_norm_weight()            the gain of the query RMSNorm, f32 [D]
//   const T* block_rows(b, key_head, first, count, stage)
//                                           `count` (1..32) consecutive complete blocks of the K head from the block
//                                           `first` as rows [count, D]: a pointer into the cache when they are
//                                           contiguous there, otherwise gathered into `stage` ([32, D]); a staged
//                                           result has zero rows from `count` on, a direct one only `count` rows
//   int32_t* selection_indices(b, v, sel_head)    where the block_topk chosen block ids of the token go
//   int32_t* selection_count(b, v, sel_head)      where their number goes
class BrgemmRunner {
public:
    using Config = QsaBrgemm::Config;
    using Plan = QsaBrgemm::Plan;

    explicit BrgemmRunner(const QsaBrgemm& brgemm)
        : m_brgemm(brgemm),
          m_config(brgemm.config()),
          m_q_k_kernel(brgemm.q_k_kernel()),
          m_k_q_kernel(brgemm.k_q_kernel()),
          m_heads_per_key(brgemm.heads_per_key()),
          m_tokens_per_call(brgemm.tokens_per_call()),
          m_tile_tokens(brgemm.tile_tokens()),
          m_packed_query_bytes(brgemm.packed_query_bytes()),
          m_offsets(brgemm.offsets()),
          m_select_offset(brgemm.select_offset()) {}

    // Selects the blocks of every token of the call. `shared` / `threads` are the scratch of plan.shared_bytes and of
    // num_threads slices of plan.thread_bytes.
    template <typename Source>
    void run(const Plan& plan, const Source& source, void* shared, void* threads, const CpuParallel& parallel) const {
        OPENVINO_ASSERT(ov::element::from<typename Source::Element>() == m_config.precision,
                        "QSA block scorer: the summary blocks are not of the precision of the scorer");
        const Context<Source> ctx{source, plan, static_cast<float*>(shared), static_cast<uint8_t*>(threads), parallel};
        if (!m_brgemm.needs_scoring(plan)) {
            // every token sees at most block_topk blocks: all of them are chosen
            parallel.parallel_for2d(plan.batch, static_cast<size_t>(plan.max_current), [&](size_t b, size_t v) {
                if (static_cast<int64_t>(v) < source.tokens(b)) {
                    select_token(ctx, b, static_cast<int64_t>(v), nullptr, nullptr);
                }
            });
            return;
        }

        const size_t groups_total = div_up(static_cast<size_t>(plan.max_row_blocks), kScoreBlockRows);
        const size_t tile_alloc_tokens = std::min(m_tile_tokens, static_cast<size_t>(plan.max_current));
        for (size_t first_tile = 0; first_tile < plan.tiles_total; first_tile += plan.tiles_per_round) {
            const size_t tiles = std::min(plan.tiles_per_round, plan.tiles_total - first_tile);
            // Few work items (decode, short windows): split the block range of an item among several threads.
            const size_t splits = std::clamp<size_t>(div_up(2 * m_config.num_threads, tiles * plan.batch),
                                                     1,
                                                     std::max<size_t>(1, groups_total / 4));

            parallel.parallel_for3d(tiles, plan.batch, splits, [&](size_t tile, size_t b, size_t split) {
                score_item(ctx, first_tile + tile, tile, b, split, splits);
            });
            if (splits == 1) {
                continue;  // the items selected already
            }
            parallel.parallel_for3d(tiles, plan.batch, tile_alloc_tokens, [&](size_t tile, size_t b, size_t t) {
                const auto v = static_cast<int64_t>((first_tile + tile) * m_tile_tokens + t);
                if (v < source.tokens(b)) {
                    select_token(ctx,
                                 b,
                                 v,
                                 tile_scores(ctx, tile, b) + t * m_config.selection_heads * plan.score_stride);
                }
            });
        }
    }

private:
    template <typename Source>
    struct Context {
        const Source& source;
        const Plan& plan;
        float* score_matrices;  // [tile of the round, row][tile token, selection head, block (padded stride)]
        uint8_t* threads;
        const CpuParallel& parallel;
    };

    // The buffers of a worker thread (see init_thread_layout()); `query_tile` and `key_chunk_stage` hold elements of
    // the precision of the summary blocks.
    struct ThreadScratch {
        float* q_row;     // f32 row of the query head, [D]
        float* normed;    // the row normalized and rotated, [D]
        float* rope_cos;  // f32 [Dr]
        float* rope_sin;
        uint8_t* query_tile;     // [q_slice][K head][query_rows][D]
        uint8_t* packed_blocks;  // the B matrix of 32 blocks (several tokens)
        uint8_t* packed_query;   // [K head] B matrices of the query tile (one token)
        float* gemm_scores;      // brgemm output: [query_rows, 32 blocks], or [32 blocks, query_rows] for one token
        uint8_t* workspace;
        uint8_t* scratch_a;
        uint8_t* key_chunk_stage;  // zero-padded copy of the last chunk of blocks of the cache
        float* select_candidates;
        uint32_t* select_histogram;
    };

    // A query slice, a part of the tokens of an item: the tokens that go through one brgemm call (an item has up to
    // kQuerySlicesPerItem of them).
    struct QuerySlice {
        size_t real_tokens;  // tokens of the item in this query slice
        size_t token_slots;  // token slots of the brgemm call: a single token goes through the k.q kernel, else all
                             // slots
        size_t query_rows;   // rows of the query tile of the query slice: token slots x heads_per_key
        int64_t blocks;      // complete blocks visible to its last token
        BrgemmKernel* kernel;
        uint8_t* query;        // its query tile, [K head][query_rows][D]
        bool is_single_token;  // one token (decode): the k.q kernel, the blocks are the A matrix (see get_score_kernel)
    };

    template <typename Source>
    [[nodiscard]] float* tile_scores(const Context<Source>& ctx, size_t tile, size_t b) const {
        return ctx.score_matrices + (tile * ctx.plan.batch + b) * (ctx.plan.score_tile_bytes / sizeof(float));
    }

    template <typename Source>
    [[nodiscard]] ThreadScratch thread_scratch(const Context<Source>& ctx) const {
        auto* const base = ctx.threads + static_cast<size_t>(parallel_get_thread_num()) * ctx.plan.thread_bytes;
        ThreadScratch s{};
        s.q_row = reinterpret_cast<float*>(base + m_offsets.rows);
        s.normed = s.q_row + m_config.dim;
        s.rope_cos = s.normed + m_config.dim;
        s.rope_sin = s.rope_cos + m_config.rotary_dim;
        s.query_tile = base + m_offsets.query_tile;
        s.packed_blocks = base + m_offsets.packed_blocks;
        s.packed_query = base + m_offsets.packed_query;
        s.gemm_scores = reinterpret_cast<float*>(base + m_offsets.gemm_scores);
        s.workspace = base + m_offsets.workspace;
        s.scratch_a = base + m_offsets.scratch_a;
        s.key_chunk_stage = base + m_offsets.key_chunk_stage;
        s.select_candidates = reinterpret_cast<float*>(base + m_select_offset);
        s.select_histogram = reinterpret_cast<uint32_t*>(s.select_candidates + ctx.plan.padded_blocks + 1);
        return s;
    }

    // Step 5 for the token `v` of row `b`: the selection of its selection heads from `token_scores`, its [H_sel, NB]
    // score rows (stride plan.score_stride). A token that sees at most block_topk blocks takes all of them: its scores
    // are not needed.
    template <typename Source>
    void select_token(const Context<Source>& ctx, size_t b, int64_t v, const float* token_scores) const {
        const auto s = thread_scratch(ctx);
        select_token(ctx, b, v, token_scores, &s);
    }

    template <typename Source>
    void select_token(const Context<Source>& ctx,
                      size_t b,
                      int64_t v,
                      const float* token_scores,
                      const ThreadScratch* s) const {
        const auto visible = static_cast<size_t>(ctx.source.blocks_visible(b, v));
        const size_t k = std::min(m_config.block_topk, visible);
        for (size_t sel_head = 0; sel_head < m_config.selection_heads; sel_head++) {
            auto* const out_row = ctx.source.selection_indices(b, v, sel_head);
            if (visible <= m_config.block_topk) {
                for (size_t i = 0; i < k; i++) {
                    out_row[i] = static_cast<int32_t>(i);
                }
            } else {
                select_top_blocks(token_scores + sel_head * ctx.plan.score_stride,
                                  visible,
                                  k,
                                  s->select_candidates,
                                  s->select_histogram,
                                  out_row);
            }
            if (m_config.order_by_score && k > 0) {
                order_blocks_by_score(token_scores + sel_head * ctx.plan.score_stride, out_row, k);
            }
            *ctx.source.selection_count(b, v, sel_head) = static_cast<int32_t>(k);
        }
    }

    // One work item: the tokens [tile * tile_tokens, ...) of row `b`, the `split`-th range of their key chunks.
    //   scores: [tile tokens, H_sel, NB] f32 (the scores of the tokens of the tile, step 4)
    template <typename Source>
    void score_item(const Context<Source>& ctx, size_t tile, size_t round_tile, size_t b, size_t split, size_t splits)
        const {
        const auto first_token = static_cast<int64_t>(tile * m_tile_tokens);
        const int64_t row_tokens = ctx.source.tokens(b);
        if (first_token >= row_tokens) {
            return;
        }
        const int64_t valid = std::min<int64_t>(static_cast<int64_t>(m_tile_tokens), row_tokens - first_token);
        const int64_t blocks = ctx.source.blocks_visible(b, first_token + valid - 1);
        float* const scores = tile_scores(ctx, round_tile, b);
        const auto thread = thread_scratch(ctx);
        // an unsplit item selects itself: its scores are in the cache of this thread
        auto select_tokens = [&]() {
            for (int64_t t = 0; t < valid; t++) {
                select_token(ctx,
                             b,
                             first_token + t,
                             scores + static_cast<size_t>(t) * m_config.selection_heads * ctx.plan.score_stride,
                             &thread);
            }
        };
        if (!m_config.order_by_score && blocks <= static_cast<int64_t>(m_config.block_topk)) {
            // every token of the tile takes all its blocks, and their order is the ascending one: no scores needed
            if (splits == 1) {
                select_tokens();
            }
            return;
        }
        const size_t key_chunks = div_up(static_cast<size_t>(blocks), kScoreBlockRows);
        const size_t key_chunks_per_split = div_up(key_chunks, splits);
        const size_t key_chunk_begin = split * key_chunks_per_split;
        const size_t key_chunk_end = std::min(key_chunks, key_chunk_begin + key_chunks_per_split);
        if (key_chunk_begin >= key_chunk_end) {
            return;
        }

        QuerySlice q_slices[kQuerySlicesPerItem];
        const size_t q_slice_count = div_up(static_cast<size_t>(valid), m_tokens_per_call);
        prepare_query_slices(ctx, thread, b, first_token, valid, q_slices);
        score_key_chunks(ctx, thread, b, key_chunk_begin, key_chunk_end, q_slices, q_slice_count, scores);
        if (splits == 1) {
            select_tokens();
        }
    }

    // Steps 1 and 2, the query side: RMSNorm + RoPE of the H_iq heads of every token of the item, as the A matrix of
    // the brgemm call that scores its token slice, per K head:
    //   query_tile[key_head]  [query_rows, D] = the q_rot rows of the G heads of the K head for the token slots of the
    //                         brgemm call; the row of (head h, token t) is (h % G) * token_slots + t
    // The rows of the missing tokens of an incomplete slice are zero. The tile of a single token is also packed, as the
    // B matrix of its brgemm call.
    template <typename Source>
    void prepare_query_slices(const Context<Source>& ctx,
                              const ThreadScratch& thread,
                              size_t b,
                              int64_t first_token,
                              int64_t valid,
                              QuerySlice* q_slices) const {
        using T = typename Source::Element;
        const Source& source = ctx.source;
        const size_t dim = m_config.dim;
        const size_t heads_per_key = m_heads_per_key;
        const size_t q_slice_count = div_up(static_cast<size_t>(valid), m_tokens_per_call);
        for (size_t q_slice_idx = 0; q_slice_idx < q_slice_count; q_slice_idx++) {
            auto& q_slice = q_slices[q_slice_idx];
            q_slice.query = thread.query_tile + q_slice_idx * m_config.index_key_heads * m_tokens_per_call *
                                                    heads_per_key * dim * sizeof(T);
            auto* const query_tile = reinterpret_cast<T*>(q_slice.query);
            q_slice.real_tokens =
                std::min(m_tokens_per_call, static_cast<size_t>(valid) - q_slice_idx * m_tokens_per_call);
            // a single token goes through the k.q kernel, anything else through the q.k one
            q_slice.is_single_token = q_slice.real_tokens == 1;
            q_slice.token_slots = q_slice.is_single_token ? 1 : m_tokens_per_call;
            q_slice.query_rows = q_slice.token_slots * heads_per_key;
            q_slice.kernel = q_slice.is_single_token ? m_k_q_kernel.get() : m_q_k_kernel.get();
            q_slice.blocks = source.blocks_visible(
                b,
                first_token + static_cast<int64_t>(q_slice_idx * m_tokens_per_call + q_slice.real_tokens) - 1);
            if (q_slice.real_tokens < q_slice.token_slots) {
                std::fill_n(query_tile, m_config.index_key_heads * q_slice.query_rows * dim, T{});  // missing tokens
            }
            for (size_t token = 0; token < q_slice.real_tokens; token++) {
                const int64_t v = first_token + static_cast<int64_t>(q_slice_idx * m_tokens_per_call + token);
                const float* const cos_row = source.rope_cos(b, v, thread.rope_cos);
                const float* const sin_row = source.rope_sin(b, v, thread.rope_sin);
                for (size_t h = 0; h < m_config.index_query_heads; h++) {
                    // q[token, h, :]  [D], as f32
                    const float* const q_row = source.query_row(b, v, h, thread.q_row);
                    // step 1 (RMSNorm of the [D] row: x / sqrt(mean(x^2) + eps) * q_norm_weight) and step 2 (RoPE of
                    // its first Dr channels with cos / sin of the token, [Dr] each) -> q_rot[h]  [D]
                    rms_rope(q_row,
                             thread.normed,
                             source.q_norm_weight(),
                             dim,
                             m_config.rotary_dim,
                             m_config.eps,
                             cos_row,
                             sin_row);
                    // q_rot[h] is the row (h % G) * tokens + token of the tile of the K head h / G, rounded to T
                    const size_t row = (h % heads_per_key) * q_slice.token_slots + token;
                    store_row(thread.normed, query_tile + ((h / heads_per_key) * q_slice.query_rows + row) * dim, dim);
                }
            }
            if (q_slice.is_single_token) {
                for (size_t key_head = 0; key_head < m_config.index_key_heads; key_head++) {
                    q_slice.kernel->copy_buffer_b(query_tile + key_head * q_slice.query_rows * dim,
                                                  thread.packed_query + key_head * m_packed_query_bytes);
                }
            }
        }
    }

    // Steps 3 and 4, the block side: 32 blocks at a time against every query slice, the block scores summed over the
    // heads of a selection head into `scores` ([tile token, selection head, block]).
    template <typename Source>
    void score_key_chunks(const Context<Source>& ctx,
                          const ThreadScratch& thread,
                          size_t b,
                          size_t key_chunk_begin,
                          size_t key_chunk_end,
                          const QuerySlice* q_slices,
                          size_t q_slice_count,
                          float* scores) const {
        using T = typename Source::Element;
        const auto key_total_blocks = static_cast<size_t>(ctx.source.blocks(b));
        // Every slice has tokens_per_call tokens but the last, which may have fewer: the item has multi-token slices
        // (q.k, they use the packed blocks) iff its first slice is one.
        const bool has_multi_token_slice = !q_slices[0].is_single_token;
        AmxTileState amx_tiles;
        for (size_t key_chunk = key_chunk_begin; key_chunk < key_chunk_end; key_chunk++) {
            const size_t key_first_block = key_chunk * kScoreBlockRows;
            // the chunks of an item end where its last token's blocks end, so the last slice sees every one of them
            const size_t key_chunk_blocks = std::min(kScoreBlockRows, key_total_blocks - key_first_block);
            for (size_t key_head = 0; key_head < m_config.index_key_heads; key_head++) {
                const size_t sel_head = (key_head * m_config.selection_heads) / m_config.index_key_heads;
                // the first K head of a selection head writes its scores, the others add
                const bool first_key_head_of_sel =
                    key_head ==
                    (sel_head * m_config.index_key_heads + m_config.selection_heads - 1) / m_config.selection_heads;
                // the 32 blocks of the chunk; the last chunk of the row may have fewer, and a source whose cache is
                // paged may have to gather them
                const T* const blocks = ctx.source.block_rows(b,
                                                              key_head,
                                                              key_first_block,
                                                              key_chunk_blocks,
                                                              reinterpret_cast<T*>(thread.key_chunk_stage));
                if (has_multi_token_slice) {
                    // the blocks are the B matrix of the multi-token slices: packed once for all of them
                    m_q_k_kernel->copy_buffer_b(const_cast<T*>(blocks), thread.packed_blocks);
                }
                for (size_t q_slice_idx = 0; q_slice_idx < q_slice_count; q_slice_idx++) {
                    const auto& q_slice = q_slices[q_slice_idx];
                    if (static_cast<int64_t>(key_first_block) >= q_slice.blocks) {
                        continue;  // no token of the slice sees these blocks
                    }
                    // the scores of the first token of the slice in `scores`
                    float* const score_out = scores +
                                             (q_slice_idx * m_tokens_per_call * m_config.selection_heads + sel_head) *
                                                 ctx.plan.score_stride +
                                             key_first_block;
                    if (q_slice.is_single_token) {
                        score_k_q_slice(thread, q_slice, blocks, key_head, first_key_head_of_sel, score_out, amx_tiles);
                    } else {
                        score_q_k_slice<T>(thread,
                                           q_slice,
                                           key_head,
                                           first_key_head_of_sel,
                                           score_out,
                                           ctx.plan.score_stride,
                                           amx_tiles);
                    }
                }
            }
        }
    }

    // Steps 3 and 4 of a single token (k.q): gemm_scores[i, head] = max(0, blocks[i] . q_rot[head]), [32, D] x [D, G]
    // -> [32, G]; the blocks are read in place. The score of the block i is the sum of its G consecutive values (the
    // ReLU is a post-op of the brgemm).
    template <typename T>
    void score_k_q_slice(const ThreadScratch& thread,
                         const QuerySlice& q_slice,
                         const T* blocks,
                         size_t key_head,
                         bool first_key_head_of_sel,
                         float* score_out,
                         AmxTileState& amx_tiles) const {
        q_slice.kernel->executeGemm(false,
                                    const_cast<T*>(blocks),
                                    thread.packed_query + key_head * m_packed_query_bytes,
                                    thread.gemm_scores,
                                    nullptr,
                                    nullptr,
                                    thread.workspace,
                                    thread.scratch_a,
                                    amx_tiles);
        for (size_t i = 0; i < kScoreBlockRows; i++) {
            const float* const row = thread.gemm_scores + i * m_heads_per_key;
            float sum = 0.0F;
            for (size_t g = 0; g < m_heads_per_key; g++) {
                sum += row[g];
            }
            score_out[i] = first_key_head_of_sel ? sum * m_config.scale : score_out[i] + sum * m_config.scale;
        }
    }

    // Steps 3 and 4 of several tokens (q.k), the blocks packed in thread.packed_blocks:
    // gemm_scores[row, i] = max(0, q_rot[row] . blocks[i]), [query_rows, D] x [D, 32] -> [query_rows, 32]; the rows
    // come in blocks of 32 and a tail.
    template <typename T>
    void score_q_k_slice(const ThreadScratch& thread,
                         const QuerySlice& q_slice,
                         size_t key_head,
                         bool first_key_head_of_sel,
                         float* score_out,
                         size_t score_stride,
                         AmxTileState& amx_tiles) const {
        const size_t dim = m_config.dim;
        const auto* const query = reinterpret_cast<const T*>(q_slice.query) + key_head * q_slice.query_rows * dim;
        const size_t full_row_tiles = q_slice.query_rows / kScoreBlockRows;
        for (size_t row_tile = 0; row_tile < full_row_tiles; row_tile++) {
            q_slice.kernel->executeGemm(false,
                                        const_cast<T*>(query) + row_tile * kScoreBlockRows * dim,
                                        thread.packed_blocks,
                                        thread.gemm_scores + row_tile * kScoreBlockRows * kScoreBlockRows,
                                        nullptr,
                                        nullptr,
                                        thread.workspace,
                                        thread.scratch_a,
                                        amx_tiles);
        }
        if (q_slice.query_rows % kScoreBlockRows != 0) {
            q_slice.kernel->executeGemm(true,
                                        const_cast<T*>(query) + full_row_tiles * kScoreBlockRows * dim,
                                        thread.packed_blocks,
                                        thread.gemm_scores + full_row_tiles * kScoreBlockRows * kScoreBlockRows,
                                        nullptr,
                                        nullptr,
                                        thread.workspace,
                                        thread.scratch_a,
                                        amx_tiles);
        }
        // step 4: for the token t, sum[i] = sum over the G heads of the K head of gemm_scores[head, t, i] (the rows
        // of one head are consecutive tokens); then score[token t, selection head, block] = scale * sum[i] (+ the
        // other K heads of the selection head). A row of the brgemm output is 32 consecutive scores of a token, as in
        // `scores`: the rows are added and stored, no transpose. The stores of a partial last chunk reach the padding
        // of the score row.
        for (size_t t = 0; t < q_slice.real_tokens; t++) {
            float sum[kScoreBlockRows];
            std::copy_n(thread.gemm_scores + t * kScoreBlockRows, kScoreBlockRows, sum);
            for (size_t g = 1; g < m_heads_per_key; g++) {
                const float* const row = thread.gemm_scores + (g * q_slice.token_slots + t) * kScoreBlockRows;
                for (size_t i = 0; i < kScoreBlockRows; i++) {
                    sum[i] += row[i];
                }
            }
            float* const token_out = score_out + t * m_config.selection_heads * score_stride;
            if (first_key_head_of_sel) {
                for (size_t i = 0; i < kScoreBlockRows; i++) {
                    token_out[i] = sum[i] * m_config.scale;
                }
            } else {
                for (size_t i = 0; i < kScoreBlockRows; i++) {
                    token_out[i] += sum[i] * m_config.scale;
                }
            }
        }
    }

    static constexpr size_t kAlign = QsaBrgemm::kAlign;
    static constexpr size_t kQuerySlicesPerItem = QsaBrgemm::kQuerySlicesPerItem;

    const QsaBrgemm& m_brgemm;
    const Config& m_config;
    const std::shared_ptr<BrgemmKernel>& m_q_k_kernel;
    const std::shared_ptr<BrgemmKernel>& m_k_q_kernel;
    size_t m_heads_per_key;
    size_t m_tokens_per_call;
    size_t m_tile_tokens;
    size_t m_packed_query_bytes;  // the B matrix of the query tile of a K head (one token)
    QsaBrgemm::ThreadOffsets m_offsets;
    size_t m_select_offset;  // the selection scratch follows the fixed part of a thread slice
};

}  // namespace ov::Extensions::Cpu::XARCH::qsa
