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

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

#include "cpu_memory.h"
#include "cpu_parallel.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/qsa_indexer_config.hpp"
#include "nodes/kernels/sparse_attn/qsa_indexer_common.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "transformations/cpu_opset/x64/op/qsa_indexer.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"

#ifdef OPENVINO_ARCH_X86_64
#    include "nodes/kernels/sparse_attn/qsa_brgemm_scorer.hpp"
#endif

namespace ov::intel_cpu::node::qsa {

using Config = ov::intel_cpu::QSAIndexerNode::Config;

// The ports of the node, each taken by its own name.
struct Ports {
    using In = ov::intel_cpu::QSAIndexerNode::InputIdx;
    using Out = ov::intel_cpu::QSAIndexerNode::OutputIdx;

    explicit Ports(const MemoryArgs& memory)
        : q_idx(memory.at(qsa_src_arg(In::Q_IDX))),
          k_idx(memory.at(qsa_src_arg(In::K_IDX))),
          rope_cos(memory.at(qsa_src_arg(In::ROPE_COS))),
          rope_sin(memory.at(qsa_src_arg(In::ROPE_SIN))),
          key_length(memory.at(qsa_src_arg(In::KEY_LENGTH))),
          q_norm_weight(memory.at(qsa_src_arg(In::Q_NORM_WEIGHT))),
          k_norm_weight(memory.at(qsa_src_arg(In::K_NORM_WEIGHT))),
          // The state inputs are read-only; the updated state lives in the matching outputs.
          past_raw_k_in(memory.at(qsa_src_arg(In::PAST_INCOMPLETE_RAW_K))),
          past_rope_in(memory.at(qsa_src_arg(In::PAST_INCOMPLETE_ROPE))),
          summary_cache_in(memory.at(qsa_src_arg(In::SUMMARY_CACHE))),
          summary_cache(memory.at(qsa_dst_arg(Out::SUMMARY_CACHE))),
          past_raw_k(memory.at(qsa_dst_arg(Out::PAST_INCOMPLETE_RAW_K))),
          past_rope(memory.at(qsa_dst_arg(Out::PAST_INCOMPLETE_ROPE))),
          sel_indices(memory.at(qsa_dst_arg(Out::SEL_INDICES))),
          sel_count(memory.at(qsa_dst_arg(Out::SEL_COUNT))) {}

    PlainTensor q_idx, k_idx, rope_cos, rope_sin, key_length, q_norm_weight, k_norm_weight;
    PlainTensor past_raw_k_in, past_rope_in, summary_cache_in;
    PlainTensor summary_cache, past_raw_k, past_rope, sel_indices, sel_count;
};

// The blocks of row b: key_length is [B] or [1] (one value shared by all the rows).
inline RowPlan row_plan(const Ports& p, const Config& c, size_t b) {
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

inline RowStats row_stats(const Ports& p, const Config& c) {
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

// The pending window of a row is [the `tail` rows of the incomplete-block state | the real tokens of this call].
// Raw K row at pending position `pos` (a view of the tensor when it is f32, `buf` otherwise):
template <typename T>
const float* pending_row(const Ports& p, const RowPlan& plan, size_t b, size_t h, int64_t pos, float* buf) {
    const size_t D_idx = p.q_idx.size(3);
    if (pos < plan.tail) {
        return row_f32<T>(p.past_raw_k_in, D_idx, buf, b, h, pos);
    }
    return row_f32<T>(p.k_idx, D_idx, buf, b, plan.num_padding + pos - plan.tail, h);
}

// [rope_cos, rope_sin] of the first token of the block starting at pending position `pos`, in the layout of the
// past_incomplete_rope state:
template <typename T>
void get_first_token_cos_sin(const Ports& p, const Config& c, const RowPlan& plan, size_t b, int64_t pos, float* dst) {
    constexpr auto prec = ov::element::from<T>();
    const size_t rotary_dim = c.indexer_rotary_dim;
    if (pos < plan.tail) {
        cpu_convert(p.past_rope_in.template ptr<T>(b), dst, prec, ov::element::f32, 2 * rotary_dim);
    } else {
        const int64_t l = plan.num_padding + pos - plan.tail;
        cpu_convert(p.rope_cos.template ptr<T>(rope_batch(p.rope_cos, b), l), dst, prec, ov::element::f32, rotary_dim);
        cpu_convert(p.rope_sin.template ptr<T>(rope_batch(p.rope_sin, b), l),
                    dst + rotary_dim,
                    prec,
                    ov::element::f32,
                    rotary_dim);
    }
}

// Step 1 and 2 of the block's index key: the RMSNorm of the pooled keys (in place) and the RoPE of its first channels
// with the cos / sin of the first token of the block starting at pending position `pos`. `cos_sin` is scratch for
// [rope_cos | rope_sin].
template <typename T, typename Backend>
void norm_rope_block(const Backend& backend,
                     const Ports& p,
                     const Config& c,
                     const RowPlan& plan,
                     size_t b,
                     int64_t pos,
                     const float* k_norm_w,
                     float* pooled,
                     float* cos_sin) {
    get_first_token_cos_sin<T>(p, c, plan, b, pos, cos_sin);
    backend.rms_rope(pooled,
                     pooled,
                     k_norm_w,
                     p.q_idx.size(3),
                     c.indexer_rotary_dim,
                     c.eps,
                     cos_sin,
                     cos_sin + c.indexer_rotary_dim);
}

// The buffers of one worker thread's slice of the scratch. The phases run one after the other, so they share the
// floats; the (score, block) pairs and the chosen ids are for the scalar scoring.
struct ThreadBuffers {
    float* q_normed;  // [H_iq, D_idx] normalized + rotated index queries (phase 1: the first row is the pooled block)
    float* row_buf;   // [D_idx] a converted row
    float* rope_cos_sin;  // [2 * rotary_dim] cos / sin of the first token of a block (phase 2: of the token)
    float* stage;         // [4, D_idx] summary rows converted to f32 (the JIT scoring)
    float* score_f32;     // [blocks] the scores of the blocks of a token and selection head
    BlockScore* scores;   // [blocks] (score * scale, block), what the top-k selects from
    int32_t* chosen;      // [block_topk]
};

struct ThreadLayout {
    size_t row_buf_off;       // floats
    size_t rope_cos_sin_off;  // floats
    size_t stage_off;         // floats
    size_t score_f32_off;     // floats
    size_t scores_off;        // bytes
    size_t chosen_off;        // bytes
    size_t bytes;

    static constexpr size_t kAlign = 64;
    static constexpr size_t kStageRows = 4;  // the rows the JIT scoring converts at a time

    ThreadLayout(size_t H_iq, size_t D_idx, size_t rotary_dim, size_t max_blocks, size_t block_topk)
        : row_buf_off(H_iq * D_idx),
          rope_cos_sin_off(row_buf_off + D_idx),
          stage_off(rope_cos_sin_off + 2 * rotary_dim),
          score_f32_off(stage_off + kStageRows * D_idx),
          scores_off(rnd_up((score_f32_off + max_blocks) * sizeof(float), kAlign)),
          chosen_off(rnd_up(scores_off + max_blocks * sizeof(BlockScore), kAlign)),
          bytes(rnd_up(chosen_off + block_topk * sizeof(int32_t), kAlign)) {}

    [[nodiscard]] ThreadBuffers bind(uint8_t* slice) const {
        auto* const floats = reinterpret_cast<float*>(slice);
        return {floats,
                floats + row_buf_off,
                floats + rope_cos_sin_off,
                floats + stage_off,
                floats + score_f32_off,
                reinterpret_cast<BlockScore*>(slice + scores_off),
                reinterpret_cast<int32_t*>(slice + chosen_off)};
    }
};

// The scratch of one call, one allocation (a second createScratchPadMem() could free the first's memory):
//   [ f32 q / k norm weights | what the brgemm scorer shares between the threads | one slice per worker thread ]
// The scorer lays out its own slices over the same space, after phases 1 and 2 have used theirs.
template <typename T>
class IndexerScratch {
public:
    // `brgemm_scorer` is the scorer of the node (null: the scalar scoring runs); the call only plans its sizes.
    IndexerScratch(const Ports& p,
                   const Config& c,
                   const RowStats& stats,
                   const ExecutorContext::CPtr& context,
                   const BrgemmBlockScorer* brgemm_scorer)
        : scorer(brgemm_scorer != nullptr && p.summary_cache.size(2) > 0 ? brgemm_scorer : nullptr),
          layout(p.q_idx.size(2), p.q_idx.size(3), c.indexer_rotary_dim, p.summary_cache.size(2), c.block_topk) {
        size_t scorer_shared_bytes = 0;
        size_t scorer_thread_bytes = 0;
#ifdef OPENVINO_ARCH_X86_64
        if (scorer) {
            scorer_plan = scorer->plan(p.q_idx.size(0), stats.max_current, stats.max_row_blocks);
            scorer_shared_bytes = scorer_plan.shared_bytes;
            scorer_thread_bytes = scorer_plan.thread_bytes;
        }
#else
        static_cast<void>(stats);
#endif
        const size_t D_idx = p.q_idx.size(3);
        const size_t scorer_shared_offset = rnd_up(2 * D_idx * sizeof(float), ThreadLayout::kAlign);
        threads_offset = rnd_up(scorer_shared_offset + scorer_shared_bytes, ThreadLayout::kAlign);
        thread_bytes = std::max(layout.bytes, scorer_thread_bytes);
        const auto num_threads = static_cast<size_t>(CpuParallel::get_num_worker_threads());
        m_memory = context->getScratchPad()->createScratchPadMem(
            std::make_shared<CpuBlockedMemoryDesc>(ov::element::u8,
                                                   ov::intel_cpu::Shape{threads_offset + num_threads * thread_bytes}));
        base = m_memory->getDataAs<uint8_t>();

        auto* const q_norm_f32 = reinterpret_cast<float*>(base);
        q_norm_w = row_f32<T>(p.q_norm_weight, D_idx, q_norm_f32);
        k_norm_w = row_f32<T>(p.k_norm_weight, D_idx, q_norm_f32 + D_idx);
        scorer_shared = base + scorer_shared_offset;
    }

    [[nodiscard]] uint8_t* threads_base() const {
        return base + threads_offset;
    }
    // This worker thread's slice of the scratch.
    [[nodiscard]] ThreadBuffers thread() const {
        return layout.bind(threads_base() + static_cast<size_t>(parallel_get_thread_num()) * thread_bytes);
    }

    const BrgemmBlockScorer* scorer;  // null: the scalar scoring runs
#ifdef OPENVINO_ARCH_X86_64
    BrgemmBlockScorer::Plan scorer_plan;
#endif
    ThreadLayout layout;
    const float* q_norm_w = nullptr;  // f32 [D_idx] (the tensor itself when f32)
    const float* k_norm_w = nullptr;
    uint8_t* scorer_shared = nullptr;

private:
    size_t threads_offset = 0;
    size_t thread_bytes = 0;
    uint8_t* base = nullptr;
    MemoryPtr m_memory;
};

#ifdef OPENVINO_ARCH_X86_64
// The layout of a call as the brgemm scorer sees it (see BrgemmBlockScorer): the rows are the rows of the batch,
//   q_idx [B, L, H_iq, D], RoPE tables [1 | B, L, Dr], summary_cache [B, H_ik, NB, D] (with the blocks this call
//   completes), selection [B, H_sel, L, block_topk] / [B, H_sel, L].
// The window of a row is left-padded: the token v of the call is at the position num_padding + v of the window.
template <typename T>
class DenseSource {
public:
    using Element = T;

    DenseSource(const Ports& ports, const Config& config, const float* q_norm_weight)
        : m_p(ports),
          m_c(config),
          m_q_norm_weight(q_norm_weight) {}

    [[nodiscard]] int64_t tokens(size_t b) const {
        return row_plan(m_p, m_c, b).current;
    }
    [[nodiscard]] int64_t blocks_visible(size_t b, int64_t v) const {
        return row_plan(m_p, m_c, b).blocks_visible(v, static_cast<int64_t>(m_c.compress_ratio));
    }
    [[nodiscard]] int64_t blocks(size_t b) const {
        const auto plan = row_plan(m_p, m_c, b);
        return plan.past_blocks + plan.new_blocks;
    }
    [[nodiscard]] const float* query_row(size_t b, int64_t v, size_t h, float* buf) const {
        return row_f32<T>(m_p.q_idx, m_p.q_idx.size(3), buf, b, position(b, v), h);
    }
    [[nodiscard]] const float* rope_cos(size_t b, int64_t v, float* buf) const {
        return row_f32<T>(m_p.rope_cos, m_c.indexer_rotary_dim, buf, rope_batch(m_p.rope_cos, b), position(b, v));
    }
    [[nodiscard]] const float* rope_sin(size_t b, int64_t v, float* buf) const {
        return row_f32<T>(m_p.rope_sin, m_c.indexer_rotary_dim, buf, rope_batch(m_p.rope_sin, b), position(b, v));
    }
    [[nodiscard]] const float* q_norm_weight() const {
        return m_q_norm_weight;
    }
    // The blocks of a K head are contiguous in the cache: 32 of them are used in place, fewer (the end of the row) are
    // copied into the zero-padded stage.
    [[nodiscard]] const T* block_rows(size_t b, size_t hk, size_t first, size_t count, T* stage) const {
        const T* rows = m_p.summary_cache.template ptr<T>(b, hk, first);
        if (count == kScoreBlockRows) {
            return rows;
        }
        const size_t dim = m_p.q_idx.size(3);
        std::copy_n(rows, count * dim, stage);
        std::fill_n(stage + count * dim, (kScoreBlockRows - count) * dim, T{});
        return stage;
    }
    [[nodiscard]] int32_t* selection_indices(size_t b, int64_t v, size_t sh) const {
        return m_p.sel_indices.template ptr<int32_t>(b, sh, position(b, v), 0);
    }
    [[nodiscard]] int32_t* selection_count(size_t b, int64_t v, size_t sh) const {
        return m_p.sel_count.template ptr<int32_t>(b, sh, position(b, v));
    }

private:
    [[nodiscard]] size_t position(size_t b, int64_t v) const {
        return static_cast<size_t>(row_plan(m_p, m_c, b).num_padding + v);
    }

    const Ports& m_p;
    const Config& m_c;
    const float* m_q_norm_weight;
};
#endif

// ---- Phase 0: the summary_cache output [B, H_ik, blocks_out, D] starts as the blocks of the earlier calls (the input
// [B, H_ik, blocks_in, D]), followed by zeros; the selection outputs start empty (indices -1, counts 0) ----
// The copy of a long cache is split in block ranges to use the threads (decode has a single row and head).
template <typename T>
void initialize_summary_cache(const Ports& p, const CpuParallel& parallel) {
    const size_t B = p.summary_cache.size(0);
    const size_t H_ik = p.summary_cache.size(1);
    const size_t blocks_out = p.summary_cache.size(2);
    const size_t blocks_in = p.summary_cache_in.size(2);

    std::fill_n(p.sel_indices.template ptr<int32_t>(),
                p.sel_indices.size(0) * p.sel_indices.size(1) * p.sel_indices.size(2) * p.sel_indices.size(3),
                -1);
    std::fill_n(p.sel_count.template ptr<int32_t>(),
                p.sel_count.size(0) * p.sel_count.size(1) * p.sel_count.size(2),
                0);

    const size_t block_bytes = p.summary_cache.size(3) * sizeof(T);
    const size_t copy_blocks = std::min(blocks_in, blocks_out);
    constexpr size_t kPartBytes = 256UL << 10;
    const auto num_threads = static_cast<size_t>(CpuParallel::get_num_worker_threads());
    const size_t parts = std::clamp<size_t>(div_up(blocks_out * block_bytes, kPartBytes), 1, num_threads);
    const size_t blocks_per_part = div_up(blocks_out, parts);
    const auto* src_base = p.summary_cache_in.template ptr<uint8_t>();
    auto* dst_base = p.summary_cache.template ptr<uint8_t>();
    parallel.parallel_for3d(B, H_ik, parts, [&](size_t b, size_t h, size_t part) {
        const size_t begin = std::min(blocks_out, part * blocks_per_part);
        const size_t end = std::min(blocks_out, begin + blocks_per_part);
        auto* dst = dst_base + (b * H_ik + h) * blocks_out * block_bytes;
        const auto* src = src_base + (b * H_ik + h) * blocks_in * block_bytes;
        const size_t copy_end = std::min(end, copy_blocks);
        if (begin < copy_end) {
            std::memcpy(dst + begin * block_bytes, src + begin * block_bytes, (copy_end - begin) * block_bytes);
        }
        const size_t zero_begin = std::max(begin, copy_blocks);
        if (zero_begin < end) {
            std::memset(dst + zero_begin * block_bytes, 0, (end - zero_begin) * block_bytes);
        }
    });
}

// ---- Phase 1: the blocks this call completes are independent of each other. Per block (B x H_ik x new_blocks):
//   pool   the mean of its `ratio` raw keys (rows [D] of the incomplete-block state or of k_idx [B, L, H_ik, D])
//   RMS    pooled / sqrt(mean(pooled^2) + eps) * k_norm_weight   [D]
//   RoPE   the first Dr channels rotated with the cos / sin of the FIRST token of the block ([Dr] each)
// and the result is the index key of the block: summary_cache[b, h, blk, :] ----
template <typename T, typename Backend>
void append_complete_blocks(const Backend& backend,
                            const Ports& p,
                            const Config& c,
                            const RowStats& stats,
                            const IndexerScratch<T>& scratch,
                            const CpuParallel& parallel) {
    const size_t D_idx = p.q_idx.size(3);
    const auto r = static_cast<int64_t>(c.compress_ratio);
    parallel.parallel_for3d(
        p.q_idx.size(0),
        p.k_idx.size(2),
        static_cast<size_t>(stats.max_new_blocks),
        [&](size_t b, size_t h, size_t j) {
            const auto plan = row_plan(p, c, b);
            if (static_cast<int64_t>(j) >= plan.new_blocks) {
                return;
            }
            const auto buf = scratch.thread();
            float* const pooled = buf.q_normed;

            const int64_t block_start = static_cast<int64_t>(j) * r;
            std::fill_n(pooled, D_idx, 0.0F);
            for (int64_t pos = block_start; pos < block_start + r; pos++) {
                const float* const row = pending_row<T>(p, plan, b, h, pos, buf.row_buf);
                for (size_t d = 0; d < D_idx; d++) {
                    pooled[d] += row[d];
                }
            }
            const float inv_r = 1.0F / static_cast<float>(r);
            for (size_t d = 0; d < D_idx; d++) {
                pooled[d] *= inv_r;
            }
            norm_rope_block<T>(backend, p, c, plan, b, block_start, scratch.k_norm_w, pooled, buf.rope_cos_sin);
            const auto blk = static_cast<size_t>(plan.past_blocks) + j;
            cpu_convert(pooled,
                        p.summary_cache.template ptr<T>(b, h, blk),
                        ov::element::f32,
                        ov::element::from<T>(),
                        D_idx);
        });
}

// ---- Phase 1b: fully overwrite the fixed-size incomplete-block state: the raw keys [B, H_ik, ratio - 1, D] of the
// tokens after the last complete block (zero-padded) and the cos / sin [B, 2 * Dr] of the first of them ----
template <typename T>
void update_incomplete_block(const Ports& p,
                             const Config& c,
                             const IndexerScratch<T>& scratch,
                             const CpuParallel& parallel) {
    constexpr auto prec = ov::element::from<T>();
    const size_t D_idx = p.q_idx.size(3);
    const size_t H_ik = p.k_idx.size(2);
    const auto r = static_cast<int64_t>(c.compress_ratio);
    parallel.parallel_for(p.q_idx.size(0), [&](size_t b) {
        const auto plan = row_plan(p, c, b);
        const auto buf = scratch.thread();

        const int64_t rest = plan.tail + plan.current - plan.new_blocks * r;
        const int64_t rest_start = plan.new_blocks * r;
        for (int64_t pos = 0; pos < r - 1; pos++) {
            for (size_t h = 0; h < H_ik; h++) {
                T* const dst = p.past_raw_k.template ptr<T>(b, h, pos);
                if (pos < rest) {
                    const float* const src = pending_row<T>(p, plan, b, h, rest_start + pos, buf.row_buf);
                    cpu_convert(src, dst, ov::element::f32, prec, D_idx);
                } else {
                    std::fill_n(dst, D_idx, T{});
                }
            }
        }
        if (rest > 0) {
            get_first_token_cos_sin<T>(p, c, plan, b, rest_start, buf.rope_cos_sin);
        } else {
            std::fill_n(buf.rope_cos_sin, 2 * c.indexer_rotary_dim, 0.0F);
        }
        cpu_convert(buf.rope_cos_sin, p.past_rope.template ptr<T>(b), ov::element::f32, prec, 2 * c.indexer_rotary_dim);
    });
}

#ifdef OPENVINO_ARCH_X86_64
// ---- Phase 2 on brgemm: false when the node has no scorer for this call (the scalar scoring runs then) ----
template <typename T>
bool run_brgemm_scorer(const Ports& p, const Config& c, const IndexerScratch<T>& scratch, const CpuParallel& parallel) {
    if (!scratch.scorer) {
        return false;
    }
    const DenseSource<T> source(p, c, scratch.q_norm_w);
    scratch.scorer->run(scratch.scorer_plan, source, scratch.scorer_shared, scratch.threads_base(), parallel);
    return true;
}
#endif

// ---- Phase 2 per real token (the steps are described in qsa_brgemm_scorer.hpp): the query RMSNorm + RoPE, then per
// selection head the scoring of the complete blocks (what the backend provides) and the top-k ----
template <typename T, typename Backend>
void run_token_scorer(const Backend& backend,
                      const Ports& p,
                      const Config& c,
                      const IndexerScratch<T>& scratch,
                      const CpuParallel& parallel) {
    const size_t D_idx = p.q_idx.size(3);
    const size_t H_iq = p.q_idx.size(2);
    const size_t H_ik = p.k_idx.size(2);
    const size_t G = H_iq / H_ik;  // index query heads per index key head
    const size_t rotary_dim = c.indexer_rotary_dim;
    const auto r = static_cast<int64_t>(c.compress_ratio);
    const float scale = ov::intel_cpu::QSAIndexerNode::effective_scale(c);
    parallel.parallel_for2d(p.q_idx.size(0), p.q_idx.size(1), [&](size_t b, size_t l_u) {
        const auto plan = row_plan(p, c, b);
        const auto l = static_cast<int64_t>(l_u);
        if (l < plan.num_padding) {
            return;
        }
        const auto buf = scratch.thread();
        float* const rope_cos_buf = buf.rope_cos_sin;
        float* const rope_sin_buf = rope_cos_buf + rotary_dim;

        // ---- Phase 2a: per index-query head, q_idx[b, l, h, :] [D] -> RMSNorm * q_norm_weight -> RoPE of the first
        // Dr channels with cos / sin[b, l, :] [Dr] -> q_normed[h] [D] ----
        const float* const cos_row = row_f32<T>(p.rope_cos, rotary_dim, rope_cos_buf, rope_batch(p.rope_cos, b), l);
        const float* const sin_row = row_f32<T>(p.rope_sin, rotary_dim, rope_sin_buf, rope_batch(p.rope_sin, b), l);
        for (size_t h = 0; h < H_iq; h++) {
            const float* const q_row = row_f32<T>(p.q_idx, D_idx, buf.row_buf, b, l, h);
            backend.rms_rope(q_row,
                             buf.q_normed + h * D_idx,
                             scratch.q_norm_w,
                             D_idx,
                             rotary_dim,
                             c.eps,
                             cos_row,
                             sin_row);
        }

        // ---- Phase 2b: per selection head sh and complete block blk visible to the token:
        //   score = scale * sum over the query heads h of sh of max(0, q_normed[h] . summary_cache[kv(h), blk])
        // then the top-k blocks by score (ties: the lower block first) ----
        // The complete blocks visible to this token: the earlier calls' plus those finished up to this token.
        const int64_t n_complete = plan.blocks_visible(l - plan.num_padding, r);
        for (size_t sh = 0; sh < c.h_sel; sh++) {
            // The index key heads of the selection head, and so their query heads: contiguous ranges.
            const size_t hk_begin = div_up(sh * H_ik, c.h_sel);
            const size_t hk_end = div_up((sh + 1) * H_ik, c.h_sel);
            size_t k = 0;
            if (n_complete > 0) {
                // the scores of the complete blocks, summed over the index key heads of the selection head: the
                // blocks of a head are contiguous rows of the summary cache
                std::fill_n(buf.score_f32, static_cast<size_t>(n_complete), 0.0F);
                for (size_t hk = hk_begin; hk < hk_end; hk++) {
                    backend.template score_run<T>(buf.q_normed + hk * G * D_idx,
                                                  p.summary_cache.template ptr<T>(b, hk, 0),
                                                  static_cast<size_t>(n_complete),
                                                  G,
                                                  D_idx,
                                                  buf.score_f32,
                                                  buf.stage,
                                                  buf.row_buf);
                }
                for (int64_t blk = 0; blk < n_complete; blk++) {
                    buf.scores[blk] = {buf.score_f32[blk] * scale, static_cast<int32_t>(blk)};
                }
                k = select_topk(buf.scores,
                                buf.chosen,
                                static_cast<size_t>(n_complete),
                                c.block_topk,
                                c.order_by_score);
            }

            std::copy_n(buf.chosen, k, p.sel_indices.template ptr<int32_t>(b, sh, l_u, 0));
            *p.sel_count.template ptr<int32_t>(b, sh, l_u) = static_cast<int32_t>(k);
        }
    });
}

// Phase 2: brgemm where the node has a scorer for this call (x64 backends), the token scorer otherwise. ScalarBackend
// has no scorer, so it always takes the token scorer.
template <typename T, typename Backend>
void run_scorer(const Backend& backend,
                const Ports& p,
                const Config& c,
                const IndexerScratch<T>& scratch,
                const CpuParallel& parallel) {
#ifdef OPENVINO_ARCH_X86_64
    if (run_brgemm_scorer<T>(p, c, scratch, parallel)) {
        return;
    }
#endif
    run_token_scorer<T>(backend, p, c, scratch, parallel);
}

// The computation of the node, T is the precision of its data ports.
template <typename T, typename Backend>
void run_indexer_typed(const Backend& backend,
                       const MemoryArgs& memory,
                       const Config& config,
                       const ExecutorContext::CPtr& context) {
    const Ports ports(memory);
    const RowStats stats = row_stats(ports, config);
    const IndexerScratch<T> scratch(ports, config, stats, context, backend.scorer());
    const auto cpu_parallel = context->getCpuParallel();
    const CpuParallel& parallel = *cpu_parallel;

    initialize_summary_cache<T>(ports, parallel);
    append_complete_blocks<T>(backend, ports, config, stats, scratch, parallel);
    update_incomplete_block<T>(ports, config, scratch, parallel);
    run_scorer<T>(backend, ports, config, scratch, parallel);
}

template <typename Backend>
void run_indexer(const Backend& backend,
                 const MemoryArgs& memory,
                 const Config& config,
                 const ExecutorContext::CPtr& context) {
    const auto precision = memory.at(qsa_src_arg(Ports::In::Q_IDX))->getDesc().getPrecision();
    if (precision == ov::element::f32) {
        run_indexer_typed<float>(backend, memory, config, context);
    } else if (precision == ov::element::f16) {
        run_indexer_typed<ov::float16>(backend, memory, config, context);
    } else if (precision == ov::element::bf16) {
        run_indexer_typed<ov::bfloat16>(backend, memory, config, context);
    } else {
        OPENVINO_THROW("QSAIndexer: unsupported precision ", precision);
    }
}

}  // namespace ov::intel_cpu::node::qsa
