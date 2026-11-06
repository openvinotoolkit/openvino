// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

// Block scoring and top-k selection of the QSAIndexer and the PagedQSAIndexer on brgemm (x64).
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
#include "nodes/kernels/sparse_attn/qsa_indexer_common.hpp"
#include "nodes/kernels/sparse_attn/qsa_jit_kernels.hpp"
#include "nodes/kernels/x64/brgemm_kernel.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"

namespace ov::intel_cpu::node::qsa {

// The blocks one brgemm call scores: the M of the kernel (BrgemmKernel::get_mblk_size()).
inline constexpr size_t kScoreBlockRows = 32;
// Rows of the query tile of a call: (tokens x heads of a K head) up to this many.
inline constexpr size_t kScoreMaxCols = 128;

// Tokens one brgemm call scores together for K heads of `heads_per_key` query heads.
inline size_t score_tokens_per_call(size_t heads_per_key) {
    return std::max<size_t>(1, kScoreMaxCols / heads_per_key);
}

// The brgemm kernel of the scorer is a JIT product that is expensive to build and depends only on the shape of its
// columns, the head dim and the precision: the layers of a model share them through the params cache of the graph
// context (like the SDPA and PagedAttention kernels), and a node fetches them once when it is created, not per call.
struct ScoreKernelKey {
    size_t query_rows = 0;  // rows of the query tile: tokens x heads of a K head
    size_t dim = 0;         // head dim, the K of the product
    ov::element::Type precision;
    bool is_k_q_gemm = false;  // C = K.Q^T, the blocks are the A matrix (one token); else C = Q.K^T, the query tile is

    [[nodiscard]] size_t hash() const {
        size_t seed = 0;
        seed = dnnl::impl::hash_combine(seed, query_rows);
        seed = dnnl::impl::hash_combine(seed, dim);
        seed = dnnl::impl::hash_combine(seed, precision.hash());
        return dnnl::impl::hash_combine(seed, is_k_q_gemm);
    }
    bool operator==(const ScoreKernelKey& rhs) const {
        return query_rows == rhs.query_rows && dim == rhs.dim && precision == rhs.precision &&
               is_k_q_gemm == rhs.is_k_q_gemm;
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

// The kernel of the q . k of 32 blocks and `query_rows` query rows, with the ReLU, from the cache; nullptr if it can
// not be built. Two forms, by how many tokens a call scores:
//   is_k_q_gemm == false (q.k, several tokens)  C[query_rows, 32] = max(0, query_tile[query_rows, D] x blocks[32, D]^T)
//       A: the query tile, B: the transposed packed blocks. A row of C is 32 scores of one token.
//   is_k_q_gemm == true (k.q, one token)        C[32, query_rows] = max(0, blocks[32, D] x query_tile[query_rows, D]^T)
//       A: the blocks, read in place; B: the transposed packed query tile. A decode step does not make an extra pass
//       over the cache (packing it would double its memory traffic, the cache is what a decode step is bound by).
inline std::shared_ptr<BrgemmKernel> get_score_kernel(const MultiCachePtr& cache,
                                                      const ov::element::Type& precision,
                                                      size_t query_rows,
                                                      size_t dim,
                                                      bool is_k_q_gemm) {
    if (!score_kernel_supported(precision, dim)) {
        return nullptr;
    }
    const auto builder = [](const ScoreKernelKey& key) -> std::shared_ptr<BrgemmKernel> {
        if (key.is_k_q_gemm) {
            return std::make_shared<BrgemmKernel>(kScoreBlockRows,
                                                  key.query_rows,
                                                  key.dim,
                                                  key.dim,
                                                  key.dim,
                                                  key.query_rows,
                                                  /*b_transposed=*/true,
                                                  key.precision,
                                                  /*b_accumulate=*/false,
                                                  /*fuse_relu=*/true);
        }
        return std::make_shared<BrgemmKernel>(key.query_rows,
                                              kScoreBlockRows,
                                              key.dim,
                                              key.dim,
                                              key.dim,
                                              kScoreBlockRows,
                                              /*b_transposed=*/true,
                                              key.precision,
                                              /*b_accumulate=*/false,
                                              /*fuse_relu=*/true);
    };
    try {
        return cache->getOrCreate(ScoreKernelKey{query_rows, dim, precision, is_k_q_gemm}, builder).first;
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
// sequence of the batch, `v` a token of its call counted from the first real token (`b`: row, `h`: query head,
// `key_head`: K head, `sel_head`: selection head). `Source` provides (all const, called from several threads):
//   using Element = T                       the precision of the summary blocks (f32, f16 or bf16), also the precision
//                                           the query tile is rounded to
//   int64_t tokens(b)                       the real tokens of the call of the row
//   int64_t blocks_visible(b, v)            the complete blocks visible to the token v
//   int64_t blocks(b)                       the complete blocks of the row after the call (>= those visible to any
//   token) const float* query_row(b, v, h, buf)    q[h] [D] as f32: a view, or converted into `buf` [D] const float*
//   rope_cos(b, v, buf)        cos [Dr] as f32 of the token (a view, or converted into `buf` [Dr]) const float*
//   rope_sin(b, v, buf) const float* q_norm_weight()            the gain of the query RMSNorm, f32 [D] const T*
//   block_rows(b, key_head, first, count, stage)
//                                           `count` (1..32) consecutive complete blocks of the K head from the block
//                                           `first` as rows [count, D]: a pointer into the cache when they are
//                                           contiguous there, otherwise gathered into `stage` ([32, D]); a staged
//                                           result has zero rows from `count` on, a direct one only `count` rows
//   int32_t* selection_indices(b, v, sel_head)    where the block_topk chosen block ids of the token go
//   int32_t* selection_count(b, v, sel_head)      where their number goes
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
            const size_t heads_per_key = key.index_query_heads / key.index_key_heads;
            const size_t tokens_per_call = score_tokens_per_call(heads_per_key);
            // `q_k_kernel` scores a full set of tokens (query rows: tokens_per_call * heads_per_key), `k_q_kernel` a
            // single token (heads_per_key rows): the decode step is one token wide.
            auto q_k_kernel = get_score_kernel(cache, key.precision, tokens_per_call * heads_per_key, key.dim, false);
            auto k_q_kernel = get_score_kernel(cache, key.precision, heads_per_key, key.dim, true);
            // the RMSNorm + RoPE of the query rows (steps 1 and 2) on the JIT kernels of the RMSNorm and RoPE nodes
            auto rms_rope = JitRmsRope::get(cache, RmsRopeKey{key.dim, key.rotary_dim, key.eps});
            if (!q_k_kernel || !k_q_kernel || !rms_rope) {
                return nullptr;
            }
            return std::make_shared<const BrgemmBlockScorer>(key,
                                                             std::move(q_k_kernel),
                                                             std::move(k_q_kernel),
                                                             std::move(rms_rope));
        };
        try {
            return cache->getOrCreate(config, builder).first;
        } catch (const ov::Exception&) {
            return nullptr;
        }
    }

    BrgemmBlockScorer(const Config& config,
                      std::shared_ptr<BrgemmKernel> q_k_kernel,
                      std::shared_ptr<BrgemmKernel> k_q_kernel,
                      std::shared_ptr<const JitRmsRope> rms_rope)
        : m_config(config),
          m_q_k_kernel(std::move(q_k_kernel)),
          m_k_q_kernel(std::move(k_q_kernel)),
          m_rms_rope(std::move(rms_rope)),
          m_heads_per_key(config.index_query_heads / config.index_key_heads),
          m_tokens_per_call(score_tokens_per_call(m_heads_per_key)),
          m_tile_tokens(m_tokens_per_call * kQuerySlicesPerItem),
          m_packed_blocks_bytes(rnd_up(m_q_k_kernel->get_scratch_b_size(), kAlign)),
          m_packed_query_bytes(rnd_up(m_k_q_kernel->get_scratch_b_size(), kAlign)) {
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
    // Query slices (the tokens of one brgemm call each) an item prepares and scores.
    static constexpr size_t kQuerySlicesPerItem = 4;
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

    // Scores are only needed when a token has more blocks than it can choose, or, to order the blocks by score, when it
    // has any.
    [[nodiscard]] bool needs_scoring(const Plan& plan) const {
        return static_cast<size_t>(plan.max_row_blocks) > (m_config.order_by_score ? 0 : m_config.block_topk);
    }

    template <typename Source>
    [[nodiscard]] float* tile_scores(const Context<Source>& ctx, size_t tile, size_t b) const {
        return ctx.score_matrices + (tile * ctx.plan.batch + b) * (ctx.plan.score_tile_bytes / sizeof(float));
    }

    // Layout of a thread slice: f32 rows | query tiles | packed blocks | packed query | brgemm output | workspace |
    // A scratch |
    // zero-padded chunk | selection scratch (its size follows from the call, see plan()).
    void init_thread_layout() {
        const size_t dim = m_config.dim;
        const size_t element = m_config.precision.size();
        const size_t query_rows = m_tokens_per_call * m_heads_per_key;
        size_t offset = 0;
        auto reserve = [&offset](size_t bytes) {
            const size_t start = offset;
            offset += rnd_up(bytes, kAlign);
            return start;
        };
        m_offsets.rows = reserve((2 * dim + 2 * m_config.rotary_dim) * sizeof(float));
        m_offsets.query_tile = reserve(kQuerySlicesPerItem * m_config.index_key_heads * query_rows * dim * element);
        m_offsets.packed_blocks = reserve(m_packed_blocks_bytes);
        m_offsets.packed_query = reserve(m_config.index_key_heads * m_packed_query_bytes);
        m_offsets.gemm_scores = reserve(query_rows * kScoreBlockRows * sizeof(float));
        m_offsets.workspace = reserve(BrgemmKernel::get_wsp_size());
        m_offsets.scratch_a = reserve(std::max(m_q_k_kernel->get_scratch_a_size(), m_k_q_kernel->get_scratch_a_size()));
        m_offsets.key_chunk_stage = reserve(kScoreBlockRows * dim * element);
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
                    (*m_rms_rope)(q_row, thread.normed, source.q_norm_weight(), cos_row, sin_row);
                    // q_rot[h] is the row (h % G) * tokens + token of the tile of the K head h / G, rounded to T
                    const size_t row = (h % heads_per_key) * q_slice.token_slots + token;
                    row_from_f32(thread.normed,
                                 query_tile + ((h / heads_per_key) * q_slice.query_rows + row) * dim,
                                 dim);
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

    struct ThreadOffsets {
        size_t rows, query_tile, packed_blocks, packed_query, gemm_scores, workspace, scratch_a, key_chunk_stage;
    };

    Config m_config;
    std::shared_ptr<BrgemmKernel> m_q_k_kernel;
    std::shared_ptr<BrgemmKernel> m_k_q_kernel;
    std::shared_ptr<const JitRmsRope> m_rms_rope;
    size_t m_heads_per_key;
    size_t m_tokens_per_call;
    size_t m_tile_tokens;
    size_t m_packed_blocks_bytes;  // the B matrix of 32 blocks (several tokens)
    size_t m_packed_query_bytes;   // the B matrix of the query tile of a K head (one token)
    ThreadOffsets m_offsets{};
    size_t m_select_offset = 0;  // the selection scratch follows the fixed part of a thread slice
};

}  // namespace ov::intel_cpu::node::qsa
