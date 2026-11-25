// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Block scoring and top-k selection of the QSAIndexer and PagedQSAIndexer on brgemm. It is included by the
// cross-compiled executors only, so it is compiled once per instruction set; AVX-512 builds have the scorer, the
// others get a stub whose create() returns nullptr.
//
// Per row (one sequence of the call; `Source` hides the dense / paged layout): L tokens, H_iq index query heads, H_ik
// key heads ("K heads"), G = H_iq / H_ik, H_sel selection heads, D head dim, Dr rotary dim, NB complete blocks.
//
// Per query token:
//   1-2. RMSNorm + RoPE of every query head                                  (prepare_query_slices)
//   3-4. score[sel_head, blk] = scale * sum over the selection head's heads of max(0, q_rot[h] . summary[h / G, blk])
//        (one brgemm per query slice and 32-block key chunk)                 (score_key_chunks)
//   5.   top-k of the visible blocks                                         (select_token)
//
// A work item is `tile_tokens` tokens x a range of key chunks; decode items split the chunks among threads.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "cache/multi_cache.h"
#include "cpu_parallel.hpp"
#include "nodes/kernels/sparse_attn/qsa_executor.hpp"
#include "nodes/kernels/sparse_attn/qsa_kernel_utils.hpp"
#include "openvino/core/type/element_type.hpp"

#if defined(HAVE_AVX512F)

#    include <immintrin.h>

#    include <bitset>
#    include <common/utils.hpp>
#    include <cpu/x64/cpu_isa_traits.hpp>
#    include <functional>
#    include <utility>

#    include "nodes/kernels/sparse_attn/qsa_executor_cache.hpp"
#    include "nodes/kernels/x64/brgemm_kernel.hpp"
#    include "openvino/core/except.hpp"
#    include "openvino/core/parallel.hpp"
#    include "openvino/runtime/system_conf.hpp"
#    include "utils/general_utils.h"

#endif  // HAVE_AVX512F

namespace ov::Extensions::Cpu::XARCH::brgemm_scorer_impl {

using ov::Extensions::Cpu::qsa::Config;
using ov::Extensions::Cpu::qsa::IndexerArgs;
using ov::Extensions::Cpu::qsa::kScoreBlockRows;
using ov::Extensions::Cpu::qsa::kScoreMaxCols;
using ov::Extensions::Cpu::qsa::PagedIndexerArgs;
using ov::Extensions::Cpu::qsa::score_tokens_per_call;
using ov::intel_cpu::CpuParallel;
using ov::intel_cpu::CpuParallelPtr;
using ov::intel_cpu::div_up;
using ov::intel_cpu::rnd_up;
using qsa_kernel::batch_entry_plan;
using qsa_kernel::block_address;
using qsa_kernel::kScoreLimit;
using qsa_kernel::rope_batch;
using qsa_kernel::row_view;
using qsa_kernel::sanitize_score;
using qsa_kernel::seq_plan;
using qsa_kernel::SeqPlan;

// The layout of a QSAIndexer call (see BrgemmScorer): rows are batch rows; q_idx [B, L, H_iq, D], RoPE tables
// [1 | B, L, Dr], summary_cache [B, H_ik, NB, D] (with this call's new blocks), selection [B, H_sel, L, block_topk]
// and [B, H_sel, L].

// A row's window is left-padded: token v is at position num_padding + v.
template <typename T>
class DenseSource {
public:
    using Element = T;
    using Args = IndexerArgs;

    static size_t batch(const Args& args) {
        return args.q_idx.size(0);
    }

    DenseSource(const IndexerArgs& call, const Config& config, const float* q_norm_weight)
        : m_call(call),
          m_c(config),
          m_q_norm_weight(q_norm_weight) {}

    [[nodiscard]] int64_t tokens(size_t b) const {
        return batch_entry_plan(m_call, m_c, b).current;
    }
    [[nodiscard]] int64_t blocks_visible(size_t b, int64_t v) const {
        return batch_entry_plan(m_call, m_c, b).blocks_visible(v, static_cast<int64_t>(m_c.key_block_size));
    }
    [[nodiscard]] int64_t blocks(size_t b) const {
        const auto plan = batch_entry_plan(m_call, m_c, b);
        return plan.past_blocks + plan.new_blocks;
    }
    [[nodiscard]] const float* query_row(size_t b, int64_t v, size_t h, float* buf) const {
        return qsa_kernel::row_as_f32(m_call.q_idx.ptr<T>(b, position(b, v), h), m_c.head_dim, buf);
    }
    [[nodiscard]] const float* rope_cos(size_t b, int64_t v, float* buf) const {
        return qsa_kernel::row_as_f32(m_call.rope_cos.ptr<T>(rope_batch(m_call.rope_cos, b), position(b, v)),
                                      m_c.rotary_dim,
                                      buf);
    }
    [[nodiscard]] const float* rope_sin(size_t b, int64_t v, float* buf) const {
        return qsa_kernel::row_as_f32(m_call.rope_sin.ptr<T>(rope_batch(m_call.rope_sin, b), position(b, v)),
                                      m_c.rotary_dim,
                                      buf);
    }
    [[nodiscard]] const float* q_norm_weight() const {
        return m_q_norm_weight;
    }
    // The blocks of a K head are contiguous in the cache: 32 of them are used in place, fewer (the end of the row) are
    // copied into the zero-padded stage.
    [[nodiscard]] const T* block_rows(size_t b, size_t hk, size_t first, size_t count, T* stage) const {
        const T* rows = m_call.summary_cache.ptr<T>(b, hk, first);
        if (count == kScoreBlockRows) {
            return rows;
        }
        const size_t dim = m_c.head_dim;
        std::copy_n(rows, count * dim, stage);
        std::fill_n(stage + count * dim, (kScoreBlockRows - count) * dim, T{});
        return stage;
    }
    [[nodiscard]] int32_t* selection_indices(size_t b, int64_t v, size_t sh) const {
        return m_call.sel_indices.ptr<int32_t>(b, sh, position(b, v), 0);
    }
    [[nodiscard]] int32_t* selection_count(size_t b, int64_t v, size_t sh) const {
        return m_call.sel_count.ptr<int32_t>(b, sh, position(b, v));
    }

private:
    [[nodiscard]] size_t position(size_t b, int64_t v) const {
        return static_cast<size_t>(batch_entry_plan(m_call, m_c, b).num_padding + v);
    }

    const IndexerArgs& m_call;
    const Config& m_c;
    const float* m_q_norm_weight;
};

// The paged layout of a call as the scoring sees it (see BrgemmScorer): rows are sequences, token v of a sequence is
// flat token plan.token_begin + v, and summary blocks are addressed through the page table. `T` is the summary cache
// precision.
template <typename T>
class PagedSource {
public:
    using Element = T;
    using Args = PagedIndexerArgs;

    static size_t batch(const Args& args) {
        return args.subsequence_begins.size(0) - 1;
    }

    PagedSource(const PagedIndexerArgs& call, const Config& config, const float* q_norm_weight)
        : m_call(call),
          m_c(config),
          m_q_norm_w(q_norm_weight) {}

    [[nodiscard]] int64_t tokens(size_t seq) const {
        return plan(seq).q_len;
    }
    // the complete blocks up to the one the token completes
    [[nodiscard]] int64_t blocks_visible(size_t seq, int64_t v) const {
        return plan(seq).blocks_visible(v);
    }
    [[nodiscard]] int64_t blocks(size_t seq) const {
        return plan(seq).total_blocks();
    }
    [[nodiscard]] const float* query_row(size_t seq, int64_t v, size_t h, float* buf) const {
        return row_view<T>(m_call.q_idx, m_c.head_dim, buf, token(seq, v), h * m_c.head_dim);
    }
    [[nodiscard]] const float* rope_cos(size_t seq, int64_t v, float* buf) const {
        return row_view<T>(m_call.rope_cos, m_c.rotary_dim, buf, token(seq, v));
    }
    [[nodiscard]] const float* rope_sin(size_t seq, int64_t v, float* buf) const {
        return row_view<T>(m_call.rope_sin, m_c.rotary_dim, buf, token(seq, v));
    }
    [[nodiscard]] const float* q_norm_weight() const {
        return m_q_norm_w;
    }
    // 32 blocks that lie in one page are used in place; blocks of several pages (a small page, or the end of the
    // sequence) are gathered, run by run, into the zero-padded stage.
    [[nodiscard]] const T* block_rows(size_t seq, size_t key_head, size_t first, size_t count, T* stage) const {
        const auto seq_plan = plan(seq);
        const auto slots = static_cast<size_t>(seq_plan.slots_per_page);
        const auto [first_page, first_slot] = block_address(m_call, seq_plan, static_cast<int64_t>(first));
        if (count == kScoreBlockRows && first_slot + count <= slots) {
            return m_call.summary_cache.ptr<T>(first_page, key_head, first_slot);
        }
        const size_t dim = m_c.head_dim;
        size_t done = 0;
        while (done < count) {
            const auto [page, slot] = block_address(m_call, seq_plan, static_cast<int64_t>(first + done));
            const size_t run = std::min(count - done, slots - slot);
            std::copy_n(m_call.summary_cache.ptr<T>(page, key_head, slot), run * dim, stage + done * dim);
            done += run;
        }
        std::fill_n(stage + count * dim, (kScoreBlockRows - count) * dim, T{});
        return stage;
    }
    [[nodiscard]] int32_t* selection_indices(size_t seq, int64_t v, size_t sel_head) const {
        return m_call.sel_indices.ptr<int32_t>(token(seq, v), sel_head, 0);
    }
    [[nodiscard]] int32_t* selection_count(size_t seq, int64_t v, size_t sel_head) const {
        return m_call.sel_count.ptr<int32_t>(token(seq, v), sel_head);
    }

private:
    [[nodiscard]] SeqPlan plan(size_t seq) const {
        return seq_plan(m_call, m_c, seq);
    }
    [[nodiscard]] size_t token(size_t seq, int64_t v) const {
        return static_cast<size_t>(plan(seq).token_begin + v);
    }

    const PagedIndexerArgs& m_call;
    const Config& m_c;
    const float* m_q_norm_w;
};

#if defined(HAVE_AVX512F)

using ov::intel_cpu::AmxTileState;
using ov::intel_cpu::BrgemmKernel;

inline constexpr size_t kSelectBuckets = 256;

// Step 5 sanitizes the scores (see qsa_kernel::sanitize_score; the executor's select_topk does the same).
// A row with a smaller range goes to one bucket: its scores are (nearly) equal anyway, and 1 / range stays finite.
constexpr float kMinScoreRange = 1e-30F;

inline __m512 sanitize_scores(__m512 v) {
    // _mm512_max_ps returns its second operand when either one is NaN
    return _mm512_min_ps(_mm512_max_ps(v, _mm512_set1_ps(-kScoreLimit)), _mm512_set1_ps(kScoreLimit));
}

// Step 5: writes the ascending ids of the k highest scores[0, n) to `chosen` (n > k > 0); ties go to the lower id.
// A histogram over equal-width buckets finds the k-th score's bucket; only it is sorted. `scores` is
// sanitized in place; `candidates`, `histogram` are scratch.
inline void select_top_blocks(float* scores,
                              size_t n,
                              size_t k,
                              float* candidates,
                              uint32_t* histogram,
                              int32_t* chosen) {
    constexpr size_t kLanes = 16;
    // sanitize the scores, and find the smallest and the largest one
    __m512 vlow = _mm512_set1_ps(kScoreLimit);
    __m512 vhigh = _mm512_set1_ps(-kScoreLimit);
    size_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
        const __m512 v = sanitize_scores(_mm512_loadu_ps(scores + i));
        _mm512_storeu_ps(scores + i, v);
        vlow = _mm512_min_ps(vlow, v);
        vhigh = _mm512_max_ps(vhigh, v);
    }
    float low = _mm512_reduce_min_ps(vlow);
    float high = _mm512_reduce_max_ps(vhigh);
    for (; i < n; i++) {
        scores[i] = sanitize_score(scores[i]);
        low = std::min(low, scores[i]);
        high = std::max(high, scores[i]);
    }

    constexpr auto max_bucket = static_cast<int32_t>(kSelectBuckets) - 1;
    const float to_bucket = high - low > kMinScoreRange ? static_cast<float>(kSelectBuckets) / (high - low) : 0.0F;
    // Same arithmetic as the vector path, so both passes agree. Scores are sanitized, so the bucket is in
    // [0, kSelectBuckets]; the clamps guard against an out-of-bounds histogram access.
    const auto bucket_of = [&](float score) {
        return std::clamp(static_cast<int32_t>((score - low) * to_bucket), 0, max_bucket);
    };
    const __m512 vlow_all = _mm512_set1_ps(low);
    const __m512 vto_bucket = _mm512_set1_ps(to_bucket);
    const auto buckets_of = [&](__m512 v) {
        const __m512i bucket = _mm512_cvttps_epi32(_mm512_mul_ps(_mm512_sub_ps(v, vlow_all), vto_bucket));
        return _mm512_max_epi32(_mm512_min_epi32(bucket, _mm512_set1_epi32(max_bucket)), _mm512_setzero_si512());
    };

    // histogram: the buckets of a vector of scores are computed at once, the counters are bumped one by one
    std::fill_n(histogram, kSelectBuckets, 0U);
    alignas(64) int32_t buckets[kLanes];
    for (i = 0; i + kLanes <= n; i += kLanes) {
        _mm512_storeu_si512(buckets, buckets_of(_mm512_loadu_ps(scores + i)));
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
    const __m512i wanted = _mm512_set1_epi32(bucket);
    for (i = 0; i + kLanes <= n; i += kLanes) {
        const __m512 v = _mm512_loadu_ps(scores + i);
        const __mmask16 mask = _mm512_cmpeq_epi32_mask(buckets_of(v), wanted);
        _mm512_mask_compressstoreu_ps(candidates + candidate_count, mask, v);
        candidate_count += std::bitset<32>(mask).count();
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

    // Blocks above the threshold, then the lowest-id blocks at the threshold score until k. Collected ascending in the
    // spare candidate buffer. A vector without a tie at the threshold (usual) takes its blocks at once, one with ties
    // block by block.
    auto* const collected = reinterpret_cast<int32_t*>(candidates);
    size_t equal_left = k - greater;
    size_t written = 0;
    const __m512 vthreshold = _mm512_set1_ps(threshold);
    const __m512i lane_ids = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    for (i = 0; i + kLanes <= n; i += kLanes) {
        const __m512 v = _mm512_loadu_ps(scores + i);
        const __mmask16 above_mask = _mm512_cmp_ps_mask(v, vthreshold, _CMP_GT_OQ);
        const __mmask16 equal_mask = _mm512_cmp_ps_mask(v, vthreshold, _CMP_EQ_OQ);
        if (equal_mask != 0 && equal_left > 0) {
            for (size_t lane = 0; lane < kLanes; lane++) {
                const bool take_equal = ((equal_mask >> lane) & 1U) != 0 && equal_left > 0;
                equal_left -= take_equal ? 1 : 0;
                collected[written] = static_cast<int32_t>(i + lane);
                written += (((above_mask >> lane) & 1U) != 0 || take_equal) ? 1 : 0;
            }
        } else {
            const __m512i ids = _mm512_add_epi32(_mm512_set1_epi32(static_cast<int32_t>(i)), lane_ids);
            _mm512_mask_compressstoreu_epi32(collected + written, above_mask, ids);
            written += std::bitset<32>(above_mask).count();
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

// Block scoring of a QSA node on brgemm. `Source` is the layout of the call (DenseSource, PagedSource).
template <typename Source>
class BrgemmScorer {
public:
    using Args = typename Source::Args;

    // The scorer of `config` on the brgemm kernels of `cache`, nullptr if there are none: the CPU, the precision or the
    // head counts of the configuration have no brgemm scoring (see getScoreKernels() in qsa_executor_cache.cpp).
    static std::unique_ptr<BrgemmScorer> create(const Config& config, const ov::intel_cpu::MultiCachePtr& cache) {
        const auto kernels = ov::Extensions::Cpu::qsa::getScoreKernels(cache, config);
        if (!kernels.q_k || !kernels.k_q) {
            return nullptr;
        }
        return std::unique_ptr<BrgemmScorer>(new BrgemmScorer(config, kernels.q_k, kernels.k_q));
    }

    // The implementation type the scorer reports (execution graph, profiling): the matrix extension of the kernels of
    // the precision on this CPU.
    [[nodiscard]] ov::intel_cpu::impl_desc_type impl_type() const {
        using namespace dnnl::impl::cpu::x64;
        const auto precision = m_config.precision;
        const bool amx = (precision == ov::element::bf16 && mayiuse(avx512_core_amx)) ||
                         (precision == ov::element::f16 && mayiuse(avx512_core_amx_fp16));
        return amx ? ov::intel_cpu::impl_desc_type::brgemm_avx512_amx : ov::intel_cpu::impl_desc_type::brgemm_avx512;
    }

    // The scratch of a call: [ f32 query norm weights | what the threads share | one slice per worker thread ].
    [[nodiscard]] size_t scratch_bytes(const Args& args) const {
        return ScratchLayout(*this, args).total_bytes;
    }

    // Selects the blocks of every token of the call into args.sel_indices / sel_count. `memory` is the scratch.
    void score(const Args& args, uint8_t* memory, const CpuParallelPtr& cpu_parallel) const {
        using T = typename Source::Element;
        OPENVINO_ASSERT(ov::element::from<T>() == m_config.precision,
                        "QSA block scorer: the summary blocks are not of the precision of the scorer");
        const ScratchLayout layout(*this, args);
        const float* const q_norm_w = qsa_kernel::row_as_f32(args.q_norm_weight.template ptr<T>(),
                                                             m_config.head_dim,
                                                             reinterpret_cast<float*>(memory));
        const Source source(args, m_config, q_norm_w);
        run(layout.plan, source, memory + layout.shared_offset, memory + layout.threads_offset, *cpu_parallel);
    }

private:
    // What a call adds to the configuration: the sizes that follow from its lengths. A handful of numbers computed per
    // call by make_plan(), the scratch (of `shared_bytes` and `thread_bytes` per worker thread) is the caller's.
    struct Plan {
        size_t batch = 0;
        int64_t max_q_len = 0;        // the most tokens of a batch entry / sequence in the call
        int64_t max_blocks = 0;       // the most complete blocks of a batch entry / sequence after the call
        size_t padded_blocks = 0;     // the blocks of a score row, a multiple of the blocks of a brgemm call
        size_t score_stride = 0;      // floats between the score rows of two tokens
        size_t score_tile_bytes = 0;  // the score matrix of one tile of tokens of one row
        size_t tiles_total = 0;       // tiles of tokens of the longest row
        size_t rows_per_round = 0;    // the rows in flight, bounded by the memory of their score matrices
        size_t tiles_per_round = 0;   // the tiles in flight, bounded by the memory of their score matrices
        size_t shared_bytes = 0;      // scratch shared by the threads: the score matrices of the tiles of a round
        size_t thread_bytes = 0;      // scratch of one worker thread
    };

    // The scratch layout of a call.
    struct ScratchLayout {
        Plan plan;
        size_t shared_offset;
        size_t threads_offset;
        size_t total_bytes;

        ScratchLayout(const BrgemmScorer& scorer, const Args& args)
            : plan(scorer.make_plan(Source::batch(args), args.max_q_len, args.max_blocks)),
              shared_offset(rnd_up(scorer.m_config.head_dim * sizeof(float), kAlign)),
              threads_offset(rnd_up(shared_offset + plan.shared_bytes, kAlign)),
              total_bytes(threads_offset + scorer.m_config.num_threads * plan.thread_bytes) {}
    };

    static constexpr size_t kAlign = 64;
    // Query slices (the tokens of one brgemm call each) an item prepares and scores.
    static constexpr size_t kQuerySlicesPerItem = 4;
    // Bound of the score matrices of the tokens in flight.
    static constexpr size_t kScoreBudgetBytes = 32UL << 20;

    BrgemmScorer(const Config& config,
                 std::shared_ptr<BrgemmKernel> q_k_kernel,
                 std::shared_ptr<BrgemmKernel> k_q_kernel)
        : m_config(config),
          m_q_k_kernel(std::move(q_k_kernel)),
          m_k_q_kernel(std::move(k_q_kernel)),
          m_heads_per_key(config.index_query_heads / config.index_key_heads),
          m_tokens_per_call(score_tokens_per_call(m_heads_per_key)),
          m_tile_tokens(m_tokens_per_call * kQuerySlicesPerItem),
          m_packed_query_bytes(rnd_up(m_k_q_kernel->get_scratch_b_size(), kAlign)),
          m_packed_blocks_bytes(rnd_up(m_q_k_kernel->get_scratch_b_size(), kAlign)) {}

    // The sizes of a call of `batch` rows, the longest of `max_q_len` tokens and with at most `max_blocks`
    // complete blocks in a batch entry / sequence.
    [[nodiscard]] Plan make_plan(size_t batch, int64_t max_q_len, int64_t max_blocks) const {
        Plan p;
        p.batch = batch;
        p.max_q_len = max_q_len;
        p.max_blocks = max_blocks;
        p.padded_blocks = rnd_up(static_cast<size_t>(max_blocks), kScoreBlockRows);
        // A cache line more than the blocks: the score rows of the tokens a call writes together do not share an L1
        // set.
        p.score_stride = p.padded_blocks + kAlign / sizeof(float);
        p.tiles_total = div_up(static_cast<size_t>(max_q_len), m_tile_tokens);
        const size_t alloc_tokens = std::min(m_tile_tokens, static_cast<size_t>(max_q_len));
        p.score_tile_bytes =
            rnd_up(alloc_tokens * m_config.num_selection_heads * p.score_stride * sizeof(float), kAlign);
        if (needs_scoring(p) && p.score_tile_bytes != 0) {
            // The budget holds that many (tile, row) score matrices: the rows of the batch first, then the tiles. A
            // batch whose first tile alone exceeds it is split in rounds of rows, so the scratch does not grow with
            // the batch (a score matrix is as long as the longest row). At least one (tile, row) is in flight.
            const size_t matrices = kScoreBudgetBytes / p.score_tile_bytes;
            p.rows_per_round = std::clamp<size_t>(matrices, 1, batch);
            p.tiles_per_round = std::clamp<size_t>(matrices / p.rows_per_round, 1, p.tiles_total);
        }
        p.shared_bytes = p.tiles_per_round * p.rows_per_round * p.score_tile_bytes;
        p.thread_bytes = partition_thread(nullptr, p.padded_blocks).bytes;
        return p;
    }

    // Scores are only needed when a token has more blocks than it can choose.
    [[nodiscard]] bool needs_scoring(const Plan& plan) const {
        return static_cast<size_t>(plan.max_blocks) > m_config.block_topk;
    }

    // Selects the blocks of every token of the call. `shared` / `threads` are the scratch of plan.shared_bytes and of
    // num_threads slices of plan.thread_bytes.
    void run(const Plan& plan, const Source& source, void* shared, void* threads, const CpuParallel& parallel) const {
        OPENVINO_ASSERT(ov::element::from<typename Source::Element>() == m_config.precision,
                        "QSA block scorer: the summary blocks are not of the precision of the scorer");
        const Context ctx{source, plan, static_cast<float*>(shared), static_cast<uint8_t*>(threads), parallel};
        if (!needs_scoring(plan)) {
            // every token sees at most block_topk blocks: all of them are chosen
            parallel.parallel_for2d(plan.batch, static_cast<size_t>(plan.max_q_len), [&](size_t b, size_t v) {
                if (static_cast<int64_t>(v) < source.tokens(b)) {
                    select_token(ctx, b, static_cast<int64_t>(v), nullptr, nullptr);
                }
            });
            return;
        }

        const size_t groups_total = div_up(static_cast<size_t>(plan.max_blocks), kScoreBlockRows);
        const size_t tile_alloc_tokens = std::min(m_tile_tokens, static_cast<size_t>(plan.max_q_len));
        for (size_t first_tile = 0; first_tile < plan.tiles_total; first_tile += plan.tiles_per_round) {
            const size_t tiles = std::min(plan.tiles_per_round, plan.tiles_total - first_tile);
            for (size_t first_row = 0; first_row < plan.batch; first_row += plan.rows_per_round) {
                const size_t rows = std::min(plan.rows_per_round, plan.batch - first_row);
                // Few work items (decode, short windows): split the block range of an item among several threads.
                const size_t splits = std::clamp<size_t>(div_up(2 * m_config.num_threads, tiles * rows),
                                                         1,
                                                         std::max<size_t>(1, groups_total / 4));

                parallel.parallel_for3d(tiles, rows, splits, [&](size_t tile, size_t row, size_t split) {
                    score_item(ctx, first_tile + tile, tile, first_row + row, split, splits);
                });
                if (splits == 1) {
                    continue;  // the items selected already
                }
                parallel.parallel_for3d(tiles, rows, tile_alloc_tokens, [&](size_t tile, size_t row, size_t t) {
                    const size_t b = first_row + row;
                    const auto v = static_cast<int64_t>((first_tile + tile) * m_tile_tokens + t);
                    if (v < source.tokens(b)) {
                        select_token(ctx,
                                     b,
                                     v,
                                     tile_scores(ctx, tile, b) + t * m_config.num_selection_heads * plan.score_stride);
                    }
                });
            }
        }
    }

    struct Context {
        const Source& source;
        const Plan& plan;
        float*
            score_matrices;  // [tile of the round, row of the round][tile token, selection head, block (padded stride)]
        uint8_t* threads;
        const CpuParallel& parallel;
    };

    // The buffers of a worker thread's slice of the scratch (see partition_thread()); `query_tile` and
    // `key_chunk_stage` hold elements of the precision of the summary blocks.
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
        size_t bytes;  // the size of the slice
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
        bool is_single_token;  // one token (decode): the k.q kernel, the blocks are the A matrix (see getScoreKernel)
    };

    [[nodiscard]] float* tile_scores(const Context& ctx, size_t tile, size_t b) const {
        // The rounds of rows start at multiples of rows_per_round: `b % rows_per_round` is the row of the round.
        return ctx.score_matrices + (tile * ctx.plan.rows_per_round + b % ctx.plan.rows_per_round) *
                                        (ctx.plan.score_tile_bytes / sizeof(float));
    }

    // The buffers of the slice that starts at `base`; with a null base only `bytes` is meaningful. The selection
    // scratch follows from the call (`padded_blocks`).
    [[nodiscard]] ThreadScratch partition_thread(uint8_t* base, size_t padded_blocks) const {
        const size_t dim = m_config.head_dim;
        const size_t element = m_config.precision.size();
        const size_t query_rows = m_tokens_per_call * m_heads_per_key;
        qsa_kernel::ScratchPartition slice(base);
        ThreadScratch s{};
        s.q_row = slice.take<float>(dim);
        s.normed = slice.take<float>(dim);
        s.rope_cos = slice.take<float>(m_config.rotary_dim);
        s.rope_sin = slice.take<float>(m_config.rotary_dim);
        s.query_tile = slice.take<uint8_t>(kQuerySlicesPerItem * m_config.index_key_heads * query_rows * dim * element);
        s.packed_blocks = slice.take<uint8_t>(m_packed_blocks_bytes);
        s.packed_query = slice.take<uint8_t>(m_config.index_key_heads * m_packed_query_bytes);
        s.gemm_scores = slice.take<float>(query_rows * kScoreBlockRows);
        s.workspace = slice.take<uint8_t>(BrgemmKernel::get_wsp_size());
        s.scratch_a =
            slice.take<uint8_t>(std::max(m_q_k_kernel->get_scratch_a_size(), m_k_q_kernel->get_scratch_a_size()));
        s.key_chunk_stage = slice.take<uint8_t>(kScoreBlockRows * dim * element);
        s.select_candidates = slice.take<float>(padded_blocks + 1);
        s.select_histogram = slice.take<uint32_t>(kSelectBuckets);
        s.bytes = slice.bytes();
        return s;
    }

    [[nodiscard]] ThreadScratch thread_scratch(const Context& ctx) const {
        return partition_thread(ctx.threads + static_cast<size_t>(parallel_get_thread_num()) * ctx.plan.thread_bytes,
                                ctx.plan.padded_blocks);
    }

    // Step 5 for the token `v` of row `b`: the selection of its selection heads from `token_scores`, its [H_sel, NB]
    // score rows (stride plan.score_stride). A token that sees at most block_topk blocks takes all of them: its scores
    // are not needed.
    void select_token(const Context& ctx, size_t b, int64_t v, float* token_scores) const {
        const auto s = thread_scratch(ctx);
        select_token(ctx, b, v, token_scores, &s);
    }

    void select_token(const Context& ctx, size_t b, int64_t v, float* token_scores, const ThreadScratch* s) const {
        const auto visible = static_cast<size_t>(ctx.source.blocks_visible(b, v));
        const size_t k = std::min(m_config.block_topk, visible);
        for (size_t sel_head = 0; sel_head < m_config.num_selection_heads; sel_head++) {
            auto* const out_row = ctx.source.selection_indices(b, v, sel_head);
            float* const row_scores =
                token_scores == nullptr ? nullptr : token_scores + sel_head * ctx.plan.score_stride;
            if (visible <= m_config.block_topk) {
                for (size_t i = 0; i < k; i++) {
                    out_row[i] = static_cast<int32_t>(i);
                }
            } else {
                select_top_blocks(row_scores, visible, k, s->select_candidates, s->select_histogram, out_row);
            }
            *ctx.source.selection_count(b, v, sel_head) = static_cast<int32_t>(k);
        }
    }

    // One work item: the tokens [tile * tile_tokens, ...) of row `b`, the `split`-th range of their key chunks.
    //   scores: [tile tokens, H_sel, NB] f32 (the scores of the tokens of the tile, step 4)
    void score_item(const Context& ctx, size_t tile, size_t round_tile, size_t b, size_t split, size_t splits) const {
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
                             scores + static_cast<size_t>(t) * m_config.num_selection_heads * ctx.plan.score_stride,
                             &thread);
            }
        };
        if (blocks <= static_cast<int64_t>(m_config.block_topk)) {
            // every token of the tile takes all its blocks, in ascending order: no scores needed
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

    // Steps 1-2, the query side: RMSNorm + RoPE of every token's heads, the A matrix of the per-K-head brgemm.
    // query_tile[key_head] is [query_rows, D]; row (head h, token t) is (h % G) * token_slots + t; missing tokens
    // are zero rows. A single token's tile is also the B matrix.
    void prepare_query_slices(const Context& ctx,
                              const ThreadScratch& thread,
                              size_t b,
                              int64_t first_token,
                              int64_t valid,
                              QuerySlice* q_slices) const {
        using T = typename Source::Element;
        const Source& source = ctx.source;
        const size_t dim = m_config.head_dim;
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
                    // its first Dr channels with cos / sin of the token, [Dr] each), each only if the op does it
                    // -> q_rot[h]  [D]
                    qsa_kernel::rms_rope(q_row,
                                         thread.normed,
                                         source.q_norm_weight(),
                                         dim,
                                         m_config.rotary_dim,
                                         m_config.eps,
                                         cos_row,
                                         sin_row,
                                         m_config.query_norm,
                                         m_config.query_rope);
                    // q_rot[h] is the row (h % G) * tokens + token of the tile of the K head h / G, rounded to T
                    const size_t row = (h % heads_per_key) * q_slice.token_slots + token;
                    qsa_kernel::store_row(thread.normed,
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
    void score_key_chunks(const Context& ctx,
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
                const size_t sel_head = (key_head * m_config.num_selection_heads) / m_config.index_key_heads;
                // the first K head of a selection head writes its scores, the others add
                const bool first_key_head_of_sel =
                    key_head == (sel_head * m_config.index_key_heads + m_config.num_selection_heads - 1) /
                                    m_config.num_selection_heads;
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
                    float* const score_out =
                        scores +
                        (q_slice_idx * m_tokens_per_call * m_config.num_selection_heads + sel_head) *
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

    // Steps 3-4 of a single token (k.q): gemm_scores[i, head] = max(0, blocks[i] . q_rot[head]), [32, D] x [D, G] ->
    // [32, G], blocks read in place. A block's score is the sum of its G values (ReLU is a brgemm post-op).
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
        const size_t dim = m_config.head_dim;
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
        // Step 4: for token t, sum[i] = sum over the K head's G heads of gemm_scores[head, t, i]; then
        // score[t, sel_head, block] = scale * sum[i] (+ other K heads). Output rows are 32 consecutive scores like
        // `scores`: added and stored, no transpose.
        for (size_t t = 0; t < q_slice.real_tokens; t++) {
            float sum[kScoreBlockRows];
            std::copy_n(thread.gemm_scores + t * kScoreBlockRows, kScoreBlockRows, sum);
            for (size_t g = 1; g < m_heads_per_key; g++) {
                const float* const row = thread.gemm_scores + (g * q_slice.token_slots + t) * kScoreBlockRows;
                for (size_t i = 0; i < kScoreBlockRows; i++) {
                    sum[i] += row[i];
                }
            }
            float* const token_out = score_out + t * m_config.num_selection_heads * score_stride;
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

    Config m_config;
    std::shared_ptr<BrgemmKernel> m_q_k_kernel;
    std::shared_ptr<BrgemmKernel> m_k_q_kernel;
    size_t m_heads_per_key;
    size_t m_tokens_per_call;
    size_t m_tile_tokens;
    size_t m_packed_query_bytes;   // the B matrix of the query tile of a K head (one token)
    size_t m_packed_blocks_bytes;  // the B matrix of 32 blocks (several tokens)
};

#else  // HAVE_AVX512F

// A variant without AVX-512 has no scorer, whatever the CPU runs (a build without AVX-512 included): create() builds
// no kernels and the executors fall back to the scoring of their stages on nullptr.
template <typename Source>
class BrgemmScorer {
public:
    using Args = typename Source::Args;

    static std::unique_ptr<BrgemmScorer> create(const Config& /*config*/,
                                                const ov::intel_cpu::MultiCachePtr& /*cache*/) {
        return nullptr;
    }
    // Never called: there is no scorer to call it on.
    [[nodiscard]] ov::intel_cpu::impl_desc_type impl_type() const {
        return ov::intel_cpu::impl_desc_type::undef;
    }
    [[nodiscard]] size_t scratch_bytes(const Args& /*args*/) const {
        return 0;
    }
    void score(const Args& /*args*/, uint8_t* /*memory*/, const CpuParallelPtr& /*cpu_parallel*/) const {}
};

#endif  // HAVE_AVX512F

}  // namespace ov::Extensions::Cpu::XARCH::brgemm_scorer_impl
