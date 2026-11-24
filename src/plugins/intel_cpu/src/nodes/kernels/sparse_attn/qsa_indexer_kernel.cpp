// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qsa_indexer_kernel.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "cpu_parallel.hpp"
#include "nodes/kernels/sparse_attn/qsa_kernel_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"

#ifdef OPENVINO_ARCH_X86_64
#    include "nodes/kernels/sparse_attn/qsa_brgemm.hpp"
#    include "nodes/kernels/sparse_attn/qsa_indexer_brgemm_kernel.hpp"
#endif

namespace ov::Extensions::Cpu::XARCH {

namespace {

using ov::intel_cpu::div_up;
#ifdef OPENVINO_ARCH_X86_64
using ov::intel_cpu::node::qsa::kScoreBlockRows;
#endif
using ov::intel_cpu::PlainTensor;
using ov::intel_cpu::node::qsa::Config;
using ov::intel_cpu::node::qsa::DensePorts;
using ov::intel_cpu::node::qsa::DenseScratchLayout;
using ov::intel_cpu::node::qsa::QsaBrgemm;
using ov::intel_cpu::node::qsa::rope_batch;
using ov::intel_cpu::node::qsa::row_plan;
using ov::intel_cpu::node::qsa::RowPlan;
using ov::intel_cpu::node::qsa::RowStats;
using ov::intel_cpu::node::qsa::ThreadLayout;

static_assert(ThreadLayout::kStageRows == qsa::kScoreRows);

// The buffers of one worker thread's slice of the scratch.
struct ThreadBuffers {
    float* q_normed;  // [H_iq, D_idx] normalized + rotated index queries (phase 1: the first row is the pooled block)
    float* row_buf;   // [D_idx] a converted row
    float* rope_cos_sin;      // [2 * rotary_dim] cos / sin of the first token of a block (phase 2: of the token)
    float* stage;             // [4, D_idx] summary rows converted to f32 (the scoring)
    float* score_f32;         // [blocks] the scores of the blocks of a token and selection head
    qsa::BlockScore* scores;  // [blocks] (score * scale, block), what the top-k selects from
    int32_t* chosen;          // [block_topk]
};

// The scratch of a call, bound to its memory.
template <typename T>
struct Scratch {
    Scratch(const DensePorts& p, const DenseScratchLayout& layout, uint8_t* memory)
        : shared(memory + layout.scorer_shared_offset),
          threads(memory + layout.threads_offset),
          m_layout(layout) {
        const size_t D_idx = p.q_idx.size(3);
        auto* const norm_weights = reinterpret_cast<float*>(memory);
        q_norm_w = qsa::row_as_f32(p.q_norm_weight.ptr<T>(), D_idx, norm_weights);
        k_norm_w = qsa::row_as_f32(p.k_norm_weight.ptr<T>(), D_idx, norm_weights + D_idx);
    }

    // This worker thread's slice of the scratch.
    [[nodiscard]] ThreadBuffers thread() const {
        const auto& t = m_layout.thread;
        auto* const slice = threads + static_cast<size_t>(parallel_get_thread_num()) * m_layout.thread_bytes;
        auto* const floats = reinterpret_cast<float*>(slice);
        return {floats,
                floats + t.row_buf_off,
                floats + t.rope_cos_sin_off,
                floats + t.stage_off,
                floats + t.score_f32_off,
                reinterpret_cast<qsa::BlockScore*>(slice + t.scores_off),
                reinterpret_cast<int32_t*>(slice + t.chosen_off)};
    }

    const float* q_norm_w = nullptr;  // f32 [D_idx] (the tensor itself when f32)
    const float* k_norm_w = nullptr;
    uint8_t* shared;   // the brgemm scorer's scratch shared by the threads
    uint8_t* threads;  // the slices of the worker threads

private:
    const DenseScratchLayout& m_layout;
};

// The pending window of a row is [the `tail` rows of the incomplete-block state | the real tokens of this call].
// Raw K row at pending position `pos` (a view of the tensor when it is f32, `buf` otherwise):
template <typename T>
const float* pending_row(const DensePorts& p, const RowPlan& plan, size_t b, size_t h, int64_t pos, float* buf) {
    const size_t D_idx = p.q_idx.size(3);
    if (pos < plan.tail) {
        return qsa::row_as_f32(p.past_raw_k_in.ptr<T>(b, h, pos), D_idx, buf);
    }
    return qsa::row_as_f32(p.k_idx.ptr<T>(b, plan.num_padding + pos - plan.tail, h), D_idx, buf);
}

// [rope_cos, rope_sin] of the first token of the block starting at pending position `pos`, in the layout of the
// past_incomplete_rope state:
template <typename T>
void get_first_token_cos_sin(const DensePorts& p,
                             const Config& c,
                             const RowPlan& plan,
                             size_t b,
                             int64_t pos,
                             float* dst) {
    const size_t rotary_dim = c.indexer_rotary_dim;
    if (pos < plan.tail) {
        cvt_copy(dst, p.past_rope_in.ptr<T>(b), 1, 2 * rotary_dim, 0, 0);
    } else {
        const int64_t l = plan.num_padding + pos - plan.tail;
        cvt_copy(dst, p.rope_cos.ptr<T>(rope_batch(p.rope_cos, b), l), 1, rotary_dim, 0, 0);
        cvt_copy(dst + rotary_dim, p.rope_sin.ptr<T>(rope_batch(p.rope_sin, b), l), 1, rotary_dim, 0, 0);
    }
}

// ---- Phase 0: the summary_cache output [B, H_ik, blocks_out, D] starts as the blocks of the earlier calls (the input
// [B, H_ik, blocks_in, D]), followed by zeros; the selection outputs start empty (indices -1, counts 0) ----
// The copy of a long cache is split in block ranges to use the threads (decode has a single row and head).
template <typename T>
void initialize_summary_cache(const DensePorts& p, const ov::intel_cpu::CpuParallel& parallel) {
    const size_t B = p.summary_cache.size(0);
    const size_t H_ik = p.summary_cache.size(1);
    const size_t blocks_out = p.summary_cache.size(2);
    const size_t blocks_in = p.summary_cache_in.size(2);

    std::fill_n(p.sel_indices.ptr<int32_t>(),
                p.sel_indices.size(0) * p.sel_indices.size(1) * p.sel_indices.size(2) * p.sel_indices.size(3),
                -1);
    std::fill_n(p.sel_count.ptr<int32_t>(), p.sel_count.size(0) * p.sel_count.size(1) * p.sel_count.size(2), 0);

    const size_t block_bytes = p.summary_cache.size(3) * sizeof(T);
    const size_t copy_blocks = std::min(blocks_in, blocks_out);
    constexpr size_t kPartBytes = 256UL << 10;
    const auto num_threads = static_cast<size_t>(ov::intel_cpu::CpuParallel::get_num_worker_threads());
    const size_t parts = std::clamp<size_t>(div_up(blocks_out * block_bytes, kPartBytes), 1, num_threads);
    const size_t blocks_per_part = div_up(blocks_out, parts);
    const auto* src_base = p.summary_cache_in.ptr<uint8_t>();
    auto* dst_base = p.summary_cache.ptr<uint8_t>();
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
template <typename T>
void append_complete_blocks(const DensePorts& p,
                            const Config& c,
                            const RowStats& stats,
                            const Scratch<T>& scratch,
                            const ov::intel_cpu::CpuParallel& parallel) {
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
                qsa::add_inplace(pooled, pending_row<T>(p, plan, b, h, pos, buf.row_buf), D_idx);
            }
            qsa::scale_inplace(pooled, 1.0F / static_cast<float>(r), D_idx);

            get_first_token_cos_sin<T>(p, c, plan, b, block_start, buf.rope_cos_sin);
            qsa::rms_rope(pooled,
                          pooled,
                          scratch.k_norm_w,
                          D_idx,
                          c.indexer_rotary_dim,
                          c.eps,
                          buf.rope_cos_sin,
                          buf.rope_cos_sin + c.indexer_rotary_dim);
            qsa::store_row(pooled, p.summary_cache.ptr<T>(b, h, static_cast<size_t>(plan.past_blocks) + j), D_idx);
        });
}

// ---- Phase 1b: fully overwrite the fixed-size incomplete-block state: the raw keys [B, H_ik, ratio - 1, D] of the
// tokens after the last complete block (zero-padded) and the cos / sin [B, 2 * Dr] of the first of them ----
template <typename T>
void update_incomplete_block(const DensePorts& p,
                             const Config& c,
                             const Scratch<T>& scratch,
                             const ov::intel_cpu::CpuParallel& parallel) {
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
                T* const dst = p.past_raw_k.ptr<T>(b, h, pos);
                if (pos < rest) {
                    qsa::store_row(pending_row<T>(p, plan, b, h, rest_start + pos, buf.row_buf), dst, D_idx);
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
        qsa::store_row(buf.rope_cos_sin, p.past_rope.ptr<T>(b), 2 * c.indexer_rotary_dim);
    });
}

#ifdef OPENVINO_ARCH_X86_64
// The layout of a call as the brgemm scoring sees it (see BrgemmRunner): the rows are the rows of the batch,
//   q_idx [B, L, H_iq, D], RoPE tables [1 | B, L, Dr], summary_cache [B, H_ik, NB, D] (with the blocks this call
//   completes), selection [B, H_sel, L, block_topk] / [B, H_sel, L].
// The window of a row is left-padded: the token v of the call is at the position num_padding + v of the window.
template <typename T>
class DenseSource {
public:
    using Element = T;

    DenseSource(const DensePorts& ports, const Config& config, const float* q_norm_weight)
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
        return qsa::row_as_f32(m_p.q_idx.ptr<T>(b, position(b, v), h), m_p.q_idx.size(3), buf);
    }
    [[nodiscard]] const float* rope_cos(size_t b, int64_t v, float* buf) const {
        return qsa::row_as_f32(m_p.rope_cos.ptr<T>(rope_batch(m_p.rope_cos, b), position(b, v)),
                               m_c.indexer_rotary_dim,
                               buf);
    }
    [[nodiscard]] const float* rope_sin(size_t b, int64_t v, float* buf) const {
        return qsa::row_as_f32(m_p.rope_sin.ptr<T>(rope_batch(m_p.rope_sin, b), position(b, v)),
                               m_c.indexer_rotary_dim,
                               buf);
    }
    [[nodiscard]] const float* q_norm_weight() const {
        return m_q_norm_weight;
    }
    // The blocks of a K head are contiguous in the cache: 32 of them are used in place, fewer (the end of the row) are
    // copied into the zero-padded stage.
    [[nodiscard]] const T* block_rows(size_t b, size_t hk, size_t first, size_t count, T* stage) const {
        const T* rows = m_p.summary_cache.ptr<T>(b, hk, first);
        if (count == kScoreBlockRows) {
            return rows;
        }
        const size_t dim = m_p.q_idx.size(3);
        std::copy_n(rows, count * dim, stage);
        std::fill_n(stage + count * dim, (kScoreBlockRows - count) * dim, T{});
        return stage;
    }
    [[nodiscard]] int32_t* selection_indices(size_t b, int64_t v, size_t sh) const {
        return m_p.sel_indices.ptr<int32_t>(b, sh, position(b, v), 0);
    }
    [[nodiscard]] int32_t* selection_count(size_t b, int64_t v, size_t sh) const {
        return m_p.sel_count.ptr<int32_t>(b, sh, position(b, v));
    }

private:
    [[nodiscard]] size_t position(size_t b, int64_t v) const {
        return static_cast<size_t>(row_plan(m_p, m_c, b).num_padding + v);
    }

    const DensePorts& m_p;
    const Config& m_c;
    const float* m_q_norm_weight;
};

// ---- Phase 2 on brgemm (the steps are described in qsa_indexer_brgemm_kernel.hpp) ----
template <typename T>
void run_brgemm_scorer(const QsaBrgemm& brgemm,
                       const DensePorts& p,
                       const Config& c,
                       const RowStats& stats,
                       const Scratch<T>& scratch,
                       const ov::intel_cpu::CpuParallel& parallel) {
    const auto plan = brgemm.plan(p.q_idx.size(0), stats.max_current, stats.max_row_blocks);
    const DenseSource<T> source(p, c, scratch.q_norm_w);
    qsa::BrgemmRunner(brgemm).run(plan, source, scratch.shared, scratch.threads, parallel);
}
#endif

// ---- Phase 2 per real token: the query RMSNorm + RoPE, then per selection head the scoring of the complete blocks
// and the top-k ----
template <typename T>
void run_token_scorer(const DensePorts& p,
                      const Config& c,
                      const Scratch<T>& scratch,
                      const ov::intel_cpu::CpuParallel& parallel) {
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
        const float* const cos_row =
            qsa::row_as_f32(p.rope_cos.ptr<T>(rope_batch(p.rope_cos, b), l), rotary_dim, rope_cos_buf);
        const float* const sin_row =
            qsa::row_as_f32(p.rope_sin.ptr<T>(rope_batch(p.rope_sin, b), l), rotary_dim, rope_sin_buf);
        for (size_t h = 0; h < H_iq; h++) {
            const float* const q_row = qsa::row_as_f32(p.q_idx.ptr<T>(b, l, h), D_idx, buf.row_buf);
            qsa::rms_rope(q_row,
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
                    qsa::score_run<T>(buf.q_normed + hk * G * D_idx,
                                      p.summary_cache.ptr<T>(b, hk, 0),
                                      static_cast<size_t>(n_complete),
                                      G,
                                      D_idx,
                                      buf.score_f32,
                                      buf.stage);
                }
                for (int64_t blk = 0; blk < n_complete; blk++) {
                    buf.scores[blk] = {buf.score_f32[blk] * scale, static_cast<int32_t>(blk)};
                }
                k = qsa::select_topk(buf.scores,
                                     buf.chosen,
                                     static_cast<size_t>(n_complete),
                                     c.block_topk,
                                     c.order_by_score);
            }

            std::copy_n(buf.chosen, k, p.sel_indices.ptr<int32_t>(b, sh, l_u, 0));
            *p.sel_count.ptr<int32_t>(b, sh, l_u) = static_cast<int32_t>(k);
        }
    });
}

// The computation of the node, T is the precision of its data ports.
template <typename T>
void run_indexer(const DensePorts& ports,
                 const Config& config,
                 const DenseScratchLayout& layout,
                 uint8_t* memory,
                 [[maybe_unused]] const QsaBrgemm* brgemm,
                 const ov::intel_cpu::CpuParallel& parallel) {
    const RowStats stats = ov::intel_cpu::node::qsa::row_stats(ports, config);
    const Scratch<T> scratch(ports, layout, memory);

    initialize_summary_cache<T>(ports, parallel);
    append_complete_blocks<T>(ports, config, stats, scratch, parallel);
    update_incomplete_block<T>(ports, config, scratch, parallel);
#ifdef OPENVINO_ARCH_X86_64
    if (brgemm != nullptr) {
        run_brgemm_scorer<T>(*brgemm, ports, config, stats, scratch, parallel);
        return;
    }
#endif
    run_token_scorer<T>(ports, config, scratch, parallel);
}

}  // namespace

void qsa_indexer_compute(const ov::intel_cpu::node::qsa::DensePorts& ports,
                         const ov::intel_cpu::QSAIndexerNode::Config& config,
                         const ov::intel_cpu::node::qsa::DenseScratchLayout& layout,
                         uint8_t* scratch,
                         const ov::intel_cpu::node::qsa::QsaBrgemm* brgemm,
                         const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    const auto precision = ports.q_idx.get_precision();
    if (precision == ov::element::f32) {
        run_indexer<float>(ports, config, layout, scratch, brgemm, *cpu_parallel);
    } else if (precision == ov::element::f16) {
        run_indexer<ov::float16>(ports, config, layout, scratch, brgemm, *cpu_parallel);
    } else if (precision == ov::element::bf16) {
        run_indexer<ov::bfloat16>(ports, config, layout, scratch, brgemm, *cpu_parallel);
    } else {
        OPENVINO_THROW("QSAIndexer: unsupported precision ", precision);
    }
}

}  // namespace ov::Extensions::Cpu::XARCH
