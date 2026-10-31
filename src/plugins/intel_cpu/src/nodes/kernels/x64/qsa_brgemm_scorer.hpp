// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

// Block scoring and top-k selection of the QSAIndexer and the PagedQSAIndexer on brgemm (x64).
//
// Notation: B batch, L tokens of the call, H_iq index query heads, H_ik index key heads ("K heads": one index key per
// block each), G = H_iq / H_ik query heads per K head, H_sel selection heads, D head dim, Dr rotary dim,
// NB complete key blocks of a row, Bk key tokens per block.
//
// Inputs:
//   q_idx          [B, L, H_iq, D]    raw index query, before RMSNorm and RoPE
//   rope_cos / sin [1 | B, L, Dr]     RoPE tables of the tokens of the call
//   q_norm_weight  [D]                RMSNorm gain of the query
//   summary_cache  [B, H_ik, NB, D]   one index key per complete block: the mean of its Bk raw keys, already RMS-normed
//                                     and rotated (the node does that before it calls the scorer)
// Output:
//   sel_indices    [B, H_sel, L, block_topk]   chosen block ids, ascending, padded with -1
//   sel_count      [B, H_sel, L]               how many were chosen
//
// For every query token (a "row" below is a vector of D channels) the steps are:
//   1. RMSNorm   q'[h]    = q[h] / sqrt(mean(q[h]^2) + eps) * q_norm_weight          q [H_iq, D] -> q' [H_iq, D]
//   2. RoPE      q''[h]   = rotate the first Dr channels of q'[h] by the token's position:
//                           x * cos + rotate_half(x) * sin; the other D - Dr channels are kept
//   3. q . k     dot[h, blk] = q''[h] . summary_cache[kv(h), blk]                    [H_iq, D] x [D, NB] -> [H_iq, NB]
//                with kv(h) = h / G, the K head of the query head h
//   4. score     score[sh, blk] = scale * sum over the heads h of the selection head sh of max(0, dot[h, blk])
//                                                                                    -> [H_sel, NB]
//   5. top-k     the block_topk blocks of the highest score among the blocks visible to the token
//                (the complete blocks of the earlier calls and those completed up to the token)
//
// Step 3 is one brgemm call per group of query tokens and 32 blocks:
//     C[32 blocks, cols] = max(0, blocks[32, D] x query_tile[cols, D]^T)         (the q . k of the 32 blocks + ReLU)
// with one column per (head of the K head group, token): cols = G x tokens of the call. The rows of the summary cache
// are the A matrix and are read in place; only the small query tile (the q'' rows of step 2) is packed as the B
// matrix. So, unlike packing the cache, a decode step never makes an extra pass over it. ReLU is a post-op of the
// brgemm kernel. The sum over the heads and the scale (step 4) follow in plain code.
//
// Work is split in items of `tile_tokens` tokens x a range of block groups:
//   - many tokens (prefill): an item covers all the block groups of its tokens and selects right away, while its
//     scores are in the cache of the thread;
//   - few tokens (decode, short windows): the block range of an item is split among several threads, the selection
//     then runs as a separate pass.
// The score matrices of the tokens in flight are bounded: the tiles are processed in rounds.

#include <algorithm>
#include <common/utils.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <utility>

#include "cache/multi_cache.h"
#include "cpu_parallel.hpp"
#include "nodes/common/cpu_convert.h"
#include "nodes/kernels/x64/brgemm_kernel.hpp"
#include "nodes/qsa_indexer_common.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"

namespace ov::intel_cpu::node::qsa {

// The blocks one brgemm call scores: the M of the kernel (BrgemmKernel::get_mblk_size()).
inline constexpr size_t kScoreBlockRows = 32;
// Columns of the query tile of a call: (tokens x heads of a K head group) up to this many.
inline constexpr size_t kScoreMaxCols = 128;

// Tokens one brgemm call scores together for K head groups of `group_size` heads.
inline size_t score_tokens_per_call(size_t group_size) {
    return std::max<size_t>(1, kScoreMaxCols / group_size);
}

// The brgemm kernel of the scorer is a JIT product that is expensive to build and depends only on the shape of its
// columns, the head dim and the precision: the layers of a model share them through the params cache of the graph
// context (like the SDPA and PagedAttention kernels), and a node fetches them once when it is created, not per call.
struct ScoreKernelKey {
    size_t cols = 0;  // columns of the query tile: tokens x heads of a K head group
    size_t dim = 0;   // head dim, the K of the product
    ov::element::Type precision;

    [[nodiscard]] size_t hash() const {
        size_t seed = 0;
        seed = dnnl::impl::hash_combine(seed, cols);
        seed = dnnl::impl::hash_combine(seed, dim);
        seed = dnnl::impl::hash_combine(seed, precision.hash());
        return seed;
    }
    bool operator==(const ScoreKernelKey& rhs) const {
        return cols == rhs.cols && dim == rhs.dim && precision == rhs.precision;
    }
};

// Whether this CPU runs the scorer for `precision` and head dim `dim`.
inline bool score_kernel_supported(const ov::element::Type& precision, size_t dim) {
    using namespace dnnl::impl::cpu::x64;
    if (dim % 2 != 0) {
        return false;
    }
    if (precision == ov::element::bf16) {
        return mayiuse(avx512_core_bf16);
    }
    if (precision == ov::element::f16) {
        return mayiuse(avx512_core_fp16);
    }
    return precision == ov::element::f32 && mayiuse(avx512_core);
}

// The number of query heads per K head when the scoring of a node with these (static) head counts, head dim and
// precision runs on brgemm on this CPU, 0 when it does not (the node then scores in plain code). The selection heads
// have to be covered by the K heads.
inline size_t brgemm_score_group_size(const ov::element::Type& precision,
                                      size_t query_heads,
                                      size_t key_heads,
                                      size_t selection_heads,
                                      size_t dim) {
    if (key_heads == 0 || query_heads % key_heads != 0 || selection_heads > key_heads ||
        !score_kernel_supported(precision, dim)) {
        return 0;
    }
    return query_heads / key_heads;
}

// The implementation type a node that scores on brgemm reports (execution graph, profiling): the matrix extension of
// the kernels of `precision` on this CPU.
inline impl_desc_type score_impl_type(const ov::element::Type& precision) {
    using namespace dnnl::impl::cpu::x64;
    const bool amx = (precision == ov::element::bf16 && mayiuse(avx512_core_amx)) ||
                     (precision == ov::element::f16 && mayiuse(avx512_core_amx_fp16));
    return amx ? impl_desc_type::brgemm_avx512_amx : impl_desc_type::brgemm_avx512;
}

// The kernel of C[32 blocks, cols] = max(0, blocks[32, D] x query_tile[cols, D]^T) (A: the blocks, B: the transposed
// packed tile), from the cache; nullptr if it can not be built.
inline std::shared_ptr<BrgemmKernel> get_score_kernel(const MultiCachePtr& cache,
                                                      const ov::element::Type& precision,
                                                      size_t cols,
                                                      size_t dim) {
    if (!score_kernel_supported(precision, dim)) {
        return nullptr;
    }
    const auto builder = [](const ScoreKernelKey& key) -> std::shared_ptr<BrgemmKernel> {
        return std::make_shared<BrgemmKernel>(kScoreBlockRows,
                                              key.cols,
                                              key.dim,
                                              key.dim,
                                              key.dim,
                                              key.cols,
                                              /*b_transposed=*/true,
                                              key.precision,
                                              /*b_accumulate=*/false,
                                              /*fuse_relu=*/true);
    };
    try {
        return cache->getOrCreate(ScoreKernelKey{cols, dim, precision}, builder).first;
    } catch (const ov::Exception&) {
        return nullptr;
    }
}

// Step 5: writes the (ascending) ids of the k blocks of the highest scores[0, n) to `chosen`, n > k > 0 (scores: the
// [NB] row of one token and selection head). Equal scores: the lower block id wins. The k-th largest score is found by
// a bucket selection: the scores are spread over kBuckets buckets of equal width between the smallest and the largest
// one, a histogram finds the bucket that holds the k-th score, and the few scores of that bucket are partially sorted
// (a skewed distribution only makes that bucket bigger). `candidates` (n + 1 floats) and `histogram` (kSelectBuckets
// values) are scratch; the passes are branch-free as which scores qualify is as good as random.
inline constexpr size_t kSelectBuckets = 256;

inline void select_top_blocks(const float* scores,
                              size_t n,
                              size_t k,
                              float* candidates,
                              uint32_t* histogram,
                              int32_t* chosen) {
    // smallest and largest score, in independent lanes so the compiler can vectorize
    constexpr size_t lanes = 8;
    float lows[lanes];
    float highs[lanes];
    std::fill_n(lows, lanes, scores[0]);
    std::fill_n(highs, lanes, scores[0]);
    size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        for (size_t lane = 0; lane < lanes; lane++) {
            lows[lane] = std::min(lows[lane], scores[i + lane]);
            highs[lane] = std::max(highs[lane], scores[i + lane]);
        }
    }
    float low = *std::min_element(lows, lows + lanes);
    float high = *std::max_element(highs, highs + lanes);
    for (; i < n; i++) {
        low = std::min(low, scores[i]);
        high = std::max(high, scores[i]);
    }

    const float to_bucket = high > low ? static_cast<float>(kSelectBuckets) / (high - low) : 0.0F;
    const auto bucket_of = [&](float score) {
        return std::min(static_cast<int>(kSelectBuckets) - 1, static_cast<int>((score - low) * to_bucket));
    };
    std::fill_n(histogram, kSelectBuckets, 0U);
    for (i = 0; i < n; i++) {
        histogram[bucket_of(scores[i])]++;
    }
    // the bucket of the k-th largest score, and how many scores are above it
    size_t above = 0;
    int bucket = static_cast<int>(kSelectBuckets) - 1;
    while (above + histogram[bucket] < k) {
        above += histogram[bucket--];
    }
    size_t candidate_count = 0;
    for (i = 0; i < n; i++) {
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
    // blocks are overwritten by the next one.
    auto* const collected = reinterpret_cast<int32_t*>(candidates);
    size_t equal_left = k - greater;
    size_t written = 0;
    for (i = 0; i < n; i++) {
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

// The scorer of the QSAIndexer and the PagedQSAIndexer. It holds what the layers with the same configuration share (the
// head counts, the kernels, the fixed part of the scratch layout) and is built once, when a node is created: through
// the params cache of the graph context like the SDPA executor, see get(). What depends on a call (its lengths, its
// data, its scratch) is not in it: plan() derives the sizes of a call from a few numbers, and run() takes the plan, the
// layout of the call (`Source`) and the scratch.
//
// `Source` is the layout of the call: it gives the scorer the tokens of the rows, the query and RoPE rows, the blocks
// and the place of the results, so the scorer does not know whether they are dense tensors or paged caches. A row is a
// sequence of the batch, `v` a token of its call counted from the first real token (`b`: row, `h`: query head, `hk`: K
// head, `sh`: selection head). `Source` provides (all const, called from several threads):
//   using Element = T                       the precision of the summary blocks (f32, f16 or bf16), also the precision
//                                           the query tile is rounded to
//   int64_t tokens(b)                       the real tokens of the call of the row
//   int64_t blocks_visible(b, v)            the complete blocks visible to the token v
//   int64_t blocks(b)                       the complete blocks of the row after the call (>= those visible to any
//   token) const float* query_row(b, v, h, buf)    q[h] [D] as f32: a view, or converted into `buf` [D] const float*
//   rope_cos(b, v, buf)        cos [Dr] as f32 of the token (a view, or converted into `buf` [Dr]) const float*
//   rope_sin(b, v, buf) const float* q_norm_weight()            the gain of the query RMSNorm, f32 [D] const T*
//   block_rows(b, hk, first, count, stage)
//                                           `count` (1..32) consecutive complete blocks of the K head from the block
//                                           `first` as rows [count, D]: a pointer into the cache when they are
//                                           contiguous there, otherwise gathered into `stage` ([32, D]); a staged
//                                           result has zero rows from `count` on, a direct one only `count` rows
//   int32_t* selection_indices(b, v, sh)    where the block_topk chosen block ids of the token go
//   int32_t* selection_count(b, v, sh)      where their number goes
class BrgemmBlockScorer {
public:
    // Everything that does not change from call to call; also the key of the scorer in the params cache.
    struct Config {
        size_t index_query_heads = 0;
        size_t index_key_heads = 0;
        size_t selection_heads = 0;
        size_t dim = 0;         // channels of an index head
        size_t rotary_dim = 0;  // channels of the query rotated by RoPE
        size_t block_topk = 0;
        size_t num_threads = 0;
        float eps = 0.0F;
        float scale = 0.0F;
        ov::element::Type precision;  // of the summary blocks
        // The chosen blocks of a row by descending score, equal scores by ascending block id (the order of
        // SparseAttentionIndexer-17), instead of ascending block ids. Needs the scores of every token, so the tokens
        // that take all their blocks are scored too.
        bool order_by_score = false;

        [[nodiscard]] size_t hash() const {
            size_t seed = 0;
            for (const size_t value :
                 {index_query_heads, index_key_heads, selection_heads, dim, rotary_dim, block_topk, num_threads}) {
                seed = dnnl::impl::hash_combine(seed, value);
            }
            seed = dnnl::impl::hash_combine(seed, eps);
            seed = dnnl::impl::hash_combine(seed, scale);
            seed = dnnl::impl::hash_combine(seed, order_by_score);
            return dnnl::impl::hash_combine(seed, precision.hash());
        }
        bool operator==(const Config& rhs) const {
            return index_query_heads == rhs.index_query_heads && index_key_heads == rhs.index_key_heads &&
                   selection_heads == rhs.selection_heads && dim == rhs.dim && rotary_dim == rhs.rotary_dim &&
                   block_topk == rhs.block_topk && num_threads == rhs.num_threads && eps == rhs.eps &&
                   scale == rhs.scale && precision == rhs.precision && order_by_score == rhs.order_by_score;
        }
    };

    // What a call adds to the configuration: the sizes that follow from its lengths. A handful of numbers computed per
    // call by plan(), the scratch (of `shared_bytes` and `thread_bytes` per worker thread) is the caller's.
    struct Plan {
        size_t batch = 0;
        int64_t max_current = 0;      // the most tokens of a row in the call
        int64_t max_row_blocks = 0;   // the most complete blocks of a row after the call
        size_t padded_blocks = 0;     // the blocks of a score row, a multiple of the blocks of a brgemm call
        size_t score_stride = 0;      // floats between the score rows of two tokens
        size_t score_tile_bytes = 0;  // the score matrix of one tile of tokens of one row
        size_t tiles_total = 0;       // tiles of tokens of the longest row
        size_t tiles_per_round = 0;   // the tiles in flight, bounded by the memory of their score matrices
        size_t shared_bytes = 0;      // scratch shared by the threads: the score matrices of the tiles of a round
        size_t thread_bytes = 0;      // scratch of one worker thread
    };

    // The scorer of `config` from the params cache (built on first use, shared by the layers of the same
    // configuration), nullptr if this CPU or the configuration has no brgemm scoring.
    static std::shared_ptr<const BrgemmBlockScorer> get(const MultiCachePtr& cache, const Config& config) {
        if (brgemm_score_group_size(config.precision,
                                    config.index_query_heads,
                                    config.index_key_heads,
                                    config.selection_heads,
                                    config.dim) == 0) {
            return nullptr;
        }
        const auto builder = [&cache](const Config& key) -> std::shared_ptr<const BrgemmBlockScorer> {
            const size_t group = key.index_query_heads / key.index_key_heads;
            const size_t tokens_per_call = score_tokens_per_call(group);
            // `full` scores a full group of tokens (kernel columns: tokens_per_call * group size), `narrow` a single
            // token (the group size): the decode step is one token wide.
            auto full = get_score_kernel(cache, key.precision, tokens_per_call * group, key.dim);
            auto narrow = tokens_per_call > 1 ? get_score_kernel(cache, key.precision, group, key.dim) : full;
            if (!full || !narrow) {
                return nullptr;
            }
            return std::make_shared<const BrgemmBlockScorer>(key, std::move(full), std::move(narrow));
        };
        try {
            return cache->getOrCreate(config, builder).first;
        } catch (const ov::Exception&) {
            return nullptr;
        }
    }

    BrgemmBlockScorer(const Config& config, std::shared_ptr<BrgemmKernel> full, std::shared_ptr<BrgemmKernel> narrow)
        : m_config(config),
          m_full(std::move(full)),
          m_narrow(std::move(narrow)),
          m_group_size(config.index_query_heads / config.index_key_heads),
          m_tokens_per_call(score_tokens_per_call(m_group_size)),
          m_tile_tokens(m_tokens_per_call * kSubTiles),
          m_packed_query_bytes(rnd_up(m_full->get_scratch_b_size(), kAlign)) {
        init_thread_layout();
    }

    // The sizes of a call of `batch` rows, the longest of `max_current` tokens and with at most `max_row_blocks`
    // complete blocks in a row.
    [[nodiscard]] Plan plan(size_t batch, int64_t max_current, int64_t max_row_blocks) const {
        Plan p;
        p.batch = batch;
        p.max_current = max_current;
        p.max_row_blocks = max_row_blocks;
        p.padded_blocks = rnd_up(static_cast<size_t>(max_row_blocks), kScoreBlockRows);
        // A cache line more than the blocks: the score rows of the tokens a call writes together do not share an L1
        // set.
        p.score_stride = p.padded_blocks + kAlign / sizeof(float);
        p.tiles_total = div_up(static_cast<size_t>(max_current), m_tile_tokens);
        const size_t alloc_tokens = std::min(m_tile_tokens, static_cast<size_t>(max_current));
        p.score_tile_bytes = rnd_up(alloc_tokens * m_config.selection_heads * p.score_stride * sizeof(float), kAlign);
        if (needs_scoring(p) && p.score_tile_bytes != 0) {
            p.tiles_per_round = std::clamp<size_t>(kScoreBudgetBytes / (batch * p.score_tile_bytes), 1, p.tiles_total);
        }
        p.shared_bytes = p.tiles_per_round * batch * p.score_tile_bytes;
        p.thread_bytes =
            m_select_offset + rnd_up((p.padded_blocks + 1) * sizeof(float) + kSelectBuckets * sizeof(uint32_t), kAlign);
        return p;
    }

    // Selects the blocks of every token of the call. `shared` / `threads` are the scratch of plan.shared_bytes and of
    // num_threads slices of plan.thread_bytes.
    template <typename Source>
    void run(const Plan& plan, const Source& source, void* shared, void* threads, const CpuParallel& parallel) const {
        OPENVINO_ASSERT(ov::element::from<typename Source::Element>() == m_config.precision,
                        "QSA block scorer: the summary blocks are not of the precision of the scorer");
        const Context<Source> ctx{source, plan, static_cast<float*>(shared), static_cast<uint8_t*>(threads), parallel};
        if (!needs_scoring(plan)) {
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
    static constexpr size_t kAlign = 64;
    // Groups of tokens (one brgemm call each) an item prepares and scores.
    static constexpr size_t kSubTiles = 4;
    // Bound of the score matrices of the tokens in flight.
    static constexpr size_t kScoreBudgetBytes = 32UL << 20;

    template <typename Source>
    struct Context {
        const Source& source;
        const Plan& plan;
        float* score_matrices;  // [tile of the round, row][tile token, selection head, block (padded stride)]
        uint8_t* threads;
        const CpuParallel& parallel;
    };

    // The buffers of a worker thread (see init_thread_layout()); `query_tile` and `group_stage` hold elements of the
    // precision of the summary blocks.
    struct ThreadScratch {
        float* q_row;     // f32 row of the query head, [D]
        float* normed;    // the row normalized and rotated, [D]
        float* rope_cos;  // f32 [Dr]
        float* rope_sin;
        uint8_t* query_tile;    // [K head][cols][D]
        uint8_t* packed_query;  // [sub-tile][K head] B matrices
        float* call_scores;     // brgemm output [32 blocks, cols]
        uint8_t* workspace;
        uint8_t* scratch_a;
        uint8_t* group_stage;  // zero-padded copy of the last group of blocks of the cache
        float* select_candidates;
        uint32_t* select_histogram;
    };

    // The tokens of an item that go through one brgemm call.
    struct SubTile {
        size_t valid;    // real tokens
        size_t tokens;   // tokens of the call (a single token goes through the narrow kernel)
        size_t cols;     // columns of the call: tokens x group size
        int64_t blocks;  // complete blocks visible to its last token
        BrgemmKernel* kernel;
    };

    // Scores are only needed when a token has more blocks than it can choose, or, to order the blocks by score, when it
    // has any.
    [[nodiscard]] bool needs_scoring(const Plan& plan) const {
        return static_cast<size_t>(plan.max_row_blocks) > (m_config.order_by_score ? 0 : m_config.block_topk);
    }

    template <typename Source>
    [[nodiscard]] float* tile_scores(const Context<Source>& ctx, size_t tile, size_t b) const {
        return ctx.score_matrices + (tile * ctx.plan.batch + b) * (ctx.plan.score_tile_bytes / sizeof(float));
    }

    // Layout of a thread slice: f32 rows | query tile | packed query tiles | brgemm output | workspace | A scratch |
    // zero-padded group | selection scratch (its size follows from the call, see plan()).
    void init_thread_layout() {
        const size_t dim = m_config.dim;
        const size_t element = m_config.precision.size();
        const size_t cols = m_tokens_per_call * m_group_size;
        size_t offset = 0;
        auto reserve = [&offset](size_t bytes) {
            const size_t start = offset;
            offset += rnd_up(bytes, kAlign);
            return start;
        };
        m_offsets.rows = reserve((2 * dim + 2 * m_config.rotary_dim) * sizeof(float));
        m_offsets.query_tile = reserve(m_config.index_key_heads * cols * dim * element);
        m_offsets.packed_query = reserve(kSubTiles * m_config.index_key_heads * m_packed_query_bytes);
        m_offsets.call_scores = reserve(kScoreBlockRows * cols * sizeof(float));
        m_offsets.workspace = reserve(BrgemmKernel::get_wsp_size());
        m_offsets.scratch_a = reserve(std::max(m_full->get_scratch_a_size(), m_narrow->get_scratch_a_size()));
        m_offsets.group_stage = reserve(kScoreBlockRows * dim * element);
        m_select_offset = offset;
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
        s.packed_query = base + m_offsets.packed_query;
        s.call_scores = reinterpret_cast<float*>(base + m_offsets.call_scores);
        s.workspace = base + m_offsets.workspace;
        s.scratch_a = base + m_offsets.scratch_a;
        s.group_stage = base + m_offsets.group_stage;
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
        for (size_t sh = 0; sh < m_config.selection_heads; sh++) {
            auto* const out_row = ctx.source.selection_indices(b, v, sh);
            if (visible <= m_config.block_topk) {
                for (size_t i = 0; i < k; i++) {
                    out_row[i] = static_cast<int32_t>(i);
                }
            } else {
                select_top_blocks(token_scores + sh * ctx.plan.score_stride,
                                  visible,
                                  k,
                                  s->select_candidates,
                                  s->select_histogram,
                                  out_row);
            }
            if (m_config.order_by_score && k > 0) {
                order_blocks_by_score(token_scores + sh * ctx.plan.score_stride, out_row, k);
            }
            *ctx.source.selection_count(b, v, sh) = static_cast<int32_t>(k);
        }
    }

    // One work item: the tokens [tile * tile_tokens, ...) of row `b`, the `split`-th range of their block groups.
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
        const size_t groups = div_up(static_cast<size_t>(blocks), kScoreBlockRows);
        const size_t groups_per_split = div_up(groups, splits);
        const size_t group_begin = split * groups_per_split;
        const size_t group_end = std::min(groups, group_begin + groups_per_split);
        if (group_begin >= group_end) {
            return;
        }

        SubTile sub_tiles[kSubTiles];
        const size_t sub_count = div_up(static_cast<size_t>(valid), m_tokens_per_call);
        prepare_query_tiles(ctx, thread, b, first_token, valid, sub_tiles);
        score_block_groups(ctx, thread, b, group_begin, group_end, sub_tiles, sub_count, scores);
        if (splits == 1) {
            select_tokens();
        }
    }

    // Steps 1 and 2, the query side: RMSNorm + RoPE of the H_iq heads of every token of the item, packed as the B
    // matrix of the call of its sub-tile, per K head:
    //   query_tile[hk]   [cols, D]  = the q'' rows of the G heads of the K head hk for the `tokens` tokens of the call;
    //                    the column of (head h, token t) is (h % G) * tokens + t, so the heads of a group come first
    //   packed_query     the same tile packed for brgemm (VNNI layout), [D, cols] as the B matrix
    // The rows of the missing tokens of an incomplete sub-tile are zero.
    template <typename Source>
    void prepare_query_tiles(const Context<Source>& ctx,
                             const ThreadScratch& thread,
                             size_t b,
                             int64_t first_token,
                             int64_t valid,
                             SubTile* sub_tiles) const {
        using T = typename Source::Element;
        const Source& source = ctx.source;
        auto* const query_tile = reinterpret_cast<T*>(thread.query_tile);
        const size_t dim = m_config.dim;
        const size_t group = m_group_size;
        const size_t sub_count = div_up(static_cast<size_t>(valid), m_tokens_per_call);
        for (size_t s = 0; s < sub_count; s++) {
            auto& sub = sub_tiles[s];
            sub.valid = std::min(m_tokens_per_call, static_cast<size_t>(valid) - s * m_tokens_per_call);
            // a single token goes through the narrow kernel, anything else through the full one
            const bool narrow = sub.valid == 1 && m_tokens_per_call > 1;
            sub.tokens = narrow ? 1 : m_tokens_per_call;
            sub.cols = sub.tokens * group;
            sub.kernel = narrow ? m_narrow.get() : m_full.get();
            sub.blocks =
                source.blocks_visible(b, first_token + static_cast<int64_t>(s * m_tokens_per_call + sub.valid) - 1);
            if (sub.valid < sub.tokens) {
                std::fill_n(query_tile, m_config.index_key_heads * sub.cols * dim, T{});  // missing tokens
            }
            for (size_t token = 0; token < sub.valid; token++) {
                const int64_t v = first_token + static_cast<int64_t>(s * m_tokens_per_call + token);
                const float* const cos_row = source.rope_cos(b, v, thread.rope_cos);
                const float* const sin_row = source.rope_sin(b, v, thread.rope_sin);
                for (size_t h = 0; h < m_config.index_query_heads; h++) {
                    // q[token, h, :]  [D], as f32
                    const float* const q_row = source.query_row(b, v, h, thread.q_row);
                    // step 1 (RMSNorm of the [D] row: x / sqrt(mean(x^2) + eps) * q_norm_weight) and step 2 (RoPE of
                    // its first Dr channels with cos / sin of the token, [Dr] each) -> q''[h]  [D]
                    rmsnorm_partial_rope(q_row,
                                         thread.normed,
                                         source.q_norm_weight(),
                                         dim,
                                         m_config.rotary_dim,
                                         m_config.eps,
                                         cos_row,
                                         sin_row);
                    // q''[h] is the column (h % G) * tokens + token of the tile of the K head h / G, rounded to T
                    const size_t column = (h % group) * sub.tokens + token;
                    row_from_f32(thread.normed, query_tile + ((h / group) * sub.cols + column) * dim, dim);
                }
            }
            for (size_t hk = 0; hk < m_config.index_key_heads; hk++) {
                sub.kernel->copy_buffer_b(
                    query_tile + hk * sub.cols * dim,
                    thread.packed_query + (s * m_config.index_key_heads + hk) * m_packed_query_bytes);
            }
        }
    }

    // Steps 3 and 4, the block side: 32 blocks at a time against every sub-tile, the block scores summed over the heads
    // of a selection head into `scores` ([tile token, selection head, block]).
    template <typename Source>
    void score_block_groups(const Context<Source>& ctx,
                            const ThreadScratch& thread,
                            size_t b,
                            size_t group_begin,
                            size_t group_end,
                            const SubTile* sub_tiles,
                            size_t sub_count,
                            float* scores) const {
        using T = typename Source::Element;
        const size_t group = m_group_size;
        const auto row_blocks = static_cast<size_t>(ctx.source.blocks(b));
        AmxTileState tiles;
        for (size_t block_group = group_begin; block_group < group_end; block_group++) {
            const size_t first_block = block_group * kScoreBlockRows;
            for (size_t hk = 0; hk < m_config.index_key_heads; hk++) {
                const size_t sh = (hk * m_config.selection_heads) / m_config.index_key_heads;
                // the first K head of a selection head writes its scores, the others add
                const bool first_of_selection =
                    hk == (sh * m_config.index_key_heads + m_config.selection_heads - 1) / m_config.selection_heads;
                // the 32 blocks of the group as the A matrix; the last group of the row may have fewer, and a source
                // whose cache is paged may have to gather them
                const size_t count = std::min(kScoreBlockRows, row_blocks - first_block);
                const T* const blocks =
                    ctx.source.block_rows(b, hk, first_block, count, reinterpret_cast<T*>(thread.group_stage));
                for (size_t s = 0; s < sub_count; s++) {
                    const auto& sub = sub_tiles[s];
                    if (static_cast<int64_t>(first_block) >= sub.blocks) {
                        continue;  // no token of the sub-tile sees these blocks
                    }
                    // step 3 (q . k): call_scores[i, col] = max(0, blocks[i] . q''[col]), [32, D] x [D, cols] -> [32,
                    // cols]
                    sub.kernel->executeGemm(
                        false,
                        const_cast<T*>(blocks),
                        thread.packed_query + (s * m_config.index_key_heads + hk) * m_packed_query_bytes,
                        thread.call_scores,
                        nullptr,
                        nullptr,
                        thread.workspace,
                        thread.scratch_a,
                        tiles);
                    // step 4: for the block i of the call, sum[t] = sum over the G heads of the K head of
                    // call_scores[i, head, t]  (the columns of one head are consecutive tokens); then
                    // score[token t, selection head, block] = scale * sum[t] (+ the other K heads of the selection
                    // head). The scores of a token are rows of `scores`, the call produced them as columns, so this
                    // is also a transpose: one value per token, in the row of each.
                    for (size_t i = 0; i < kScoreBlockRows; i++) {
                        const float* const row = thread.call_scores + i * sub.cols;
                        float sum[kScoreMaxCols];
                        std::fill_n(sum, sub.valid, 0.0F);
                        for (size_t g = 0; g < group; g++) {
                            for (size_t t = 0; t < sub.valid; t++) {
                                sum[t] += row[g * sub.tokens + t];  // (ReLU is a post-op of the brgemm)
                            }
                        }
                        for (size_t t = 0; t < sub.valid; t++) {
                            const size_t at =
                                ((s * m_tokens_per_call + t) * m_config.selection_heads + sh) * ctx.plan.score_stride +
                                first_block + i;
                            scores[at] =
                                first_of_selection ? sum[t] * m_config.scale : scores[at] + sum[t] * m_config.scale;
                        }
                    }
                }
            }
        }
    }

    struct ThreadOffsets {
        size_t rows, query_tile, packed_query, call_scores, workspace, scratch_a, group_stage;
    };

    Config m_config;
    std::shared_ptr<BrgemmKernel> m_full;
    std::shared_ptr<BrgemmKernel> m_narrow;
    size_t m_group_size;
    size_t m_tokens_per_call;
    size_t m_tile_tokens;
    size_t m_packed_query_bytes;
    ThreadOffsets m_offsets{};
    size_t m_select_offset = 0;  // the selection scratch follows the fixed part of a thread slice
};

}  // namespace ov::intel_cpu::node::qsa
