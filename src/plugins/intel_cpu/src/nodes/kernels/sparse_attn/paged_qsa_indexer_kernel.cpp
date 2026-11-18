// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_qsa_indexer_kernel.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "cpu_parallel.hpp"
#include "nodes/kernels/sparse_attn/qsa_kernel_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
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
using ov::intel_cpu::node::qsa::QsaBrgemm;
using ov::intel_cpu::node::qsa::paged::block_address;
using ov::intel_cpu::node::qsa::paged::Config;
using ov::intel_cpu::node::qsa::paged::index_key_heads;
using ov::intel_cpu::node::qsa::paged::index_query_heads;
using ov::intel_cpu::node::qsa::paged::num_sequences;
using ov::intel_cpu::node::qsa::paged::num_tokens;
using ov::intel_cpu::node::qsa::paged::PagedPorts;
using ov::intel_cpu::node::qsa::paged::PagedScratchLayout;
using ov::intel_cpu::node::qsa::paged::seq_plan;
using ov::intel_cpu::node::qsa::paged::SeqPlan;
using ov::intel_cpu::node::qsa::paged::SeqStats;
using ov::intel_cpu::node::qsa::paged::ThreadLayout;
using ov::intel_cpu::node::qsa::paged::token_address;

static_assert(ThreadLayout::kStageRows == qsa::kScoreRows);

// A row of a port as f32: the row itself when the data is f32 (no copy), otherwise converted into `scratch`. All the
// floating-point ports have the precision T (the op validates it), so no port needs a run-time lookup of its own.
template <typename T, typename... Idx>
const float* row_view(const PlainTensor& t, size_t len, float* scratch, Idx... idx) {
    if constexpr (std::is_same_v<T, float>) {
        return t.ptr<float>(idx...);
    } else {
        cvt_copy(scratch, t.ptr<T>(idx...), 1, len, 0, 0);
        return scratch;
    }
}

// The buffers of one worker thread's slice of the scratch; the phases run one after the other, so they share them.
struct ThreadBuffers {
    float* q_normed;          // [H_iq, D_idx] normalized + rotated index queries
    float* row_buf;           // [D_idx] a converted row
    float* summary_buf;       // [D_idx] a pooled summary row
    float* pooled;            // [D_idx] the mean of the raw keys of a block
    float* cos_sin;           // [2 * rotary_dim] the interleaved cos / sin row of a block
    float* cos_tmp;           // [rotary_dim] a cos row
    float* sin_tmp;           // [rotary_dim] a sin row
    float* stage;             // [4, D_idx] summary rows converted to f32 (the scoring)
    float* score_f32;         // [blocks] the scores of the blocks of a token and selection head
    qsa::BlockScore* scores;  // [blocks] (score * scale, block), what the top-k selects from
    int32_t* chosen;          // [block_topk]
};

// The scratch of a call, bound to its memory.
struct Scratch {
    template <typename T>
    Scratch(std::in_place_type_t<T>,
            const PagedPorts& p,
            const Config& c,
            const PagedScratchLayout& layout,
            uint8_t* memory)
        : token_seq(reinterpret_cast<int32_t*>(memory)),
          shared(memory + layout.scorer_shared_offset),
          threads(memory + layout.threads_offset),
          m_layout(layout) {
        const size_t D_idx = c.indexer_head_dim;
        auto* const q_norm = reinterpret_cast<float*>(memory + layout.norm_weights_offset);
        auto* const k_norm = q_norm + D_idx;
        // f32 weights are used in place; 16-bit ones are converted once per call into the scratch
        q_norm_w = qsa::row_as_f32(p.q_norm_weight.ptr<T>(), D_idx, q_norm);
        k_norm_w = qsa::row_as_f32(p.k_norm_weight.ptr<T>(), D_idx, k_norm);
        const size_t tokens = num_tokens(p);
        std::fill_n(token_seq, tokens, static_cast<int32_t>(-1));
        for (size_t seq = 0; seq < num_sequences(p); seq++) {
            const auto plan = seq_plan(p, c, seq);
            for (int64_t t = plan.token_begin; t < plan.token_begin + plan.q_len; t++) {
                if (static_cast<size_t>(t) < tokens) {
                    token_seq[t] = static_cast<int32_t>(seq);
                }
            }
        }
    }

    // This worker thread's slice of the scratch.
    [[nodiscard]] ThreadBuffers thread() const {
        const auto& t = m_layout.thread;
        auto* const slice = threads + static_cast<size_t>(parallel_get_thread_num()) * m_layout.thread_bytes;
        auto* const floats = reinterpret_cast<float*>(slice);
        return {floats + t.q_normed_off,
                floats + t.row_buf_off,
                floats + t.summary_buf_off,
                floats + t.pooled_off,
                floats + t.cos_sin_off,
                floats + t.cos_tmp_off,
                floats + t.sin_tmp_off,
                floats + t.stage_off,
                floats + t.score_f32_off,
                reinterpret_cast<qsa::BlockScore*>(slice + t.scores_off),
                reinterpret_cast<int32_t*>(slice + t.chosen_off)};
    }

    int32_t* token_seq;  // per token: its sequence, -1 for a token that belongs to none
    const float* q_norm_w = nullptr;
    const float* k_norm_w = nullptr;
    uint8_t* shared;   // the brgemm scorer's scratch shared by the threads
    uint8_t* threads;  // the slices of the worker threads

private:
    const PagedScratchLayout& m_layout;
};

#ifdef OPENVINO_ARCH_X86_64
// The paged layout of a call as the brgemm scoring sees it (see BrgemmRunner): rows are sequences, token v of a
// sequence is flat token plan.token_begin + v, and summary blocks are addressed through the page table. `T` is the
// summary cache precision.
template <typename T>
class PagedSource {
public:
    using Element = T;

    PagedSource(const PagedPorts& ports, const Config& config, const float* q_norm_weight)
        : m_p(ports),
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
        return row_view<T>(m_p.q_idx, m_c.indexer_head_dim, buf, token(seq, v), h * m_c.indexer_head_dim);
    }
    [[nodiscard]] const float* rope_cos(size_t seq, int64_t v, float* buf) const {
        return row_view<T>(m_p.rope_cos, m_c.indexer_rotary_dim, buf, token(seq, v));
    }
    [[nodiscard]] const float* rope_sin(size_t seq, int64_t v, float* buf) const {
        return row_view<T>(m_p.rope_sin, m_c.indexer_rotary_dim, buf, token(seq, v));
    }
    [[nodiscard]] const float* q_norm_weight() const {
        return m_q_norm_w;
    }
    // 32 blocks that lie in one page are used in place; blocks of several pages (a small page, or the end of the
    // sequence) are gathered, run by run, into the zero-padded stage.
    [[nodiscard]] const T* block_rows(size_t seq, size_t key_head, size_t first, size_t count, T* stage) const {
        const auto seq_plan = plan(seq);
        const auto slots = static_cast<size_t>(seq_plan.slots_per_page);
        const auto [first_page, first_slot] = block_address(m_p, seq_plan, static_cast<int64_t>(first));
        if (count == kScoreBlockRows && first_slot + count <= slots) {
            return m_p.summary_cache.ptr<T>(first_page, key_head, first_slot);
        }
        const size_t dim = m_c.indexer_head_dim;
        size_t done = 0;
        while (done < count) {
            const auto [page, slot] = block_address(m_p, seq_plan, static_cast<int64_t>(first + done));
            const size_t run = std::min(count - done, slots - slot);
            std::copy_n(m_p.summary_cache.ptr<T>(page, key_head, slot), run * dim, stage + done * dim);
            done += run;
        }
        std::fill_n(stage + count * dim, (kScoreBlockRows - count) * dim, T{});
        return stage;
    }
    [[nodiscard]] int32_t* selection_indices(size_t seq, int64_t v, size_t sel_head) const {
        return m_p.sel_indices.ptr<int32_t>(token(seq, v), sel_head, 0);
    }
    [[nodiscard]] int32_t* selection_count(size_t seq, int64_t v, size_t sel_head) const {
        return m_p.sel_count.ptr<int32_t>(token(seq, v), sel_head);
    }

private:
    [[nodiscard]] SeqPlan plan(size_t seq) const {
        return seq_plan(m_p, m_c, seq);
    }
    [[nodiscard]] size_t token(size_t seq, int64_t v) const {
        return static_cast<size_t>(plan(seq).token_begin + v);
    }

    const PagedPorts& m_p;
    const Config& m_c;
    const float* m_q_norm_w;
};

// ---- Phase 3 on brgemm (the steps are described in qsa_indexer_brgemm_kernel.hpp) ----
template <typename T>
void run_brgemm_scorer(const QsaBrgemm& brgemm,
                       const PagedPorts& p,
                       const Config& c,
                       const SeqStats& stats,
                       const Scratch& scratch,
                       const ov::intel_cpu::CpuParallel& parallel) {
    const auto plan = brgemm.plan(num_sequences(p), stats.max_q_len, stats.max_blocks);
    const PagedSource<T> source(p, c, scratch.q_norm_w);
    qsa::BrgemmRunner(brgemm).run(plan, source, scratch.shared, scratch.threads, parallel);
}
#endif

// ---- Phase 1, per token: append the raw index keys to the paged cache. A token that starts a block also stores its
// interleaved cos/sin row, for the block's summary to use once the block completes. A block that can never complete
// is skipped. ----
template <typename T>
void store_token_rows(const PagedPorts& p, const Config& c, const Scratch& scratch, size_t token) {
    const auto plan = seq_plan(p, c, static_cast<size_t>(scratch.token_seq[token]));
    const int64_t logical_pos = plan.past_len + (static_cast<int64_t>(token) - plan.token_begin);
    const auto buf = scratch.thread();
    const size_t D_idx = c.indexer_head_dim;
    const size_t rotary_dim = c.indexer_rotary_dim;

    if (logical_pos % plan.key_block_size == 0) {
        const int64_t block_idx = logical_pos / plan.key_block_size;
        if (block_idx < plan.block_capacity) {
            float* const cos_sin = buf.cos_sin;
            const float* const cos_row = row_view<T>(p.rope_cos, rotary_dim, buf.cos_tmp, token);
            const float* const sin_row = row_view<T>(p.rope_sin, rotary_dim, buf.sin_tmp, token);
            for (size_t j = 0; j < rotary_dim; j++) {
                cos_sin[2 * j] = cos_row[j];
                cos_sin[2 * j + 1] = sin_row[j];
            }
            const auto [page, slot] = block_address(p, plan, block_idx);
            cvt_copy(p.indexer_rope_cache.ptr<T>(page, slot), cos_sin, 1, 2 * rotary_dim, 0, 0);
        }
    }
    const auto [cur_page, cur_slot] = token_address(p, plan, logical_pos);
    for (size_t key_head = 0; key_head < index_key_heads(p, c); key_head++) {
        const float* const row = row_view<T>(p.k_idx, D_idx, buf.row_buf, token, key_head * D_idx);
        cvt_copy(p.raw_k_cache.ptr<T>(cur_page, key_head, cur_slot), row, 1, D_idx, 0, 0);
    }
}

// ---- Phase 2, per (sequence, index-key head, block completed by the call): the summary of the block is the mean of
// its raw keys, RMSNorm'ed and RoPE'd with the coefficients of its first token. The blocks are independent of each
// other. ----
template <typename T>
void pool_block(const PagedPorts& p, const Config& c, const Scratch& scratch, size_t seq, size_t key_head, size_t j) {
    const auto plan = seq_plan(p, c, seq);
    const auto buf = scratch.thread();
    float* const pooled = buf.pooled;
    const size_t D_idx = c.indexer_head_dim;
    const size_t rotary_dim = c.indexer_rotary_dim;

    const int64_t block_idx = plan.past_blocks + static_cast<int64_t>(j);
    const int64_t block_start = block_idx * plan.key_block_size;
    const auto [block_page, block_slot] = block_address(p, plan, block_idx);
    std::fill_n(pooled, D_idx, 0.0F);
    for (int64_t pos = block_start; pos < block_start + plan.key_block_size; pos++) {
        const auto [p_page, p_slot] = token_address(p, plan, pos);
        qsa::add_inplace(pooled, row_view<T>(p.raw_k_cache, D_idx, buf.row_buf, p_page, key_head, p_slot), D_idx);
    }
    qsa::scale_inplace(pooled, 1.0F / static_cast<float>(plan.key_block_size), D_idx);
    // the cos / sin of the first token of the block were stored interleaved; the RoPE takes them apart
    const float* const cos_sin = row_view<T>(p.indexer_rope_cache, 2 * rotary_dim, buf.cos_sin, block_page, block_slot);
    for (size_t k = 0; k < rotary_dim; k++) {
        buf.cos_tmp[k] = cos_sin[2 * k];
        buf.sin_tmp[k] = cos_sin[2 * k + 1];
    }
    qsa::rms_rope(pooled, buf.summary_buf, scratch.k_norm_w, D_idx, rotary_dim, c.epsilon, buf.cos_tmp, buf.sin_tmp);
    cvt_copy(p.summary_cache.ptr<T>(block_page, key_head, block_slot), buf.summary_buf, 1, D_idx, 0, 0);
}

// ---- Phase 3, per token: RMSNorm + partial RoPE of every index-query head, then per selection head the score of
// each complete block (sum over its heads of max(0, q . summary)) and the top-k blocks in ascending order. ----
template <typename T>
void select_blocks(const PagedPorts& p, const Config& c, const Scratch& scratch, size_t token) {
    const auto plan = seq_plan(p, c, static_cast<size_t>(scratch.token_seq[token]));
    const auto buf = scratch.thread();
    const size_t D_idx = c.indexer_head_dim;
    const size_t rotary_dim = c.indexer_rotary_dim;
    const size_t index_keys = index_key_heads(p, c);
    const size_t index_queries = index_query_heads(p, c);
    const size_t heads_per_key = index_queries / index_keys;
    const size_t selection_heads = c.num_selection_heads;
    const float scale = ov::op::internal::PagedQSAIndexer::effective_scale(c);

    const float* const cos_row = row_view<T>(p.rope_cos, rotary_dim, buf.cos_tmp, token);
    const float* const sin_row = row_view<T>(p.rope_sin, rotary_dim, buf.sin_tmp, token);
    for (size_t h = 0; h < index_queries; h++) {
        const float* const q_row = row_view<T>(p.q_idx, D_idx, buf.row_buf, token, h * D_idx);
        qsa::rms_rope(q_row,
                      buf.q_normed + h * D_idx,
                      scratch.q_norm_w,
                      D_idx,
                      rotary_dim,
                      c.epsilon,
                      cos_row,
                      sin_row);
    }

    // The complete blocks visible to this token, including the one this token completes.
    const int64_t n_complete = plan.blocks_visible(static_cast<int64_t>(token) - plan.token_begin);
    const auto slots = static_cast<size_t>(plan.slots_per_page);
    for (size_t sel_head = 0; sel_head < selection_heads; sel_head++) {
        // The index key heads of the selection head, and so their query heads: contiguous ranges.
        const size_t key_head_begin = div_up(sel_head * index_keys, selection_heads);
        const size_t key_head_end = div_up((sel_head + 1) * index_keys, selection_heads);
        size_t k = 0;
        if (n_complete > 0) {
            // the scores of the complete blocks, summed over the index key heads of the selection head: the blocks
            // of a head lie in runs of consecutive rows, one per page
            std::fill_n(buf.score_f32, static_cast<size_t>(n_complete), 0.0F);
            for (size_t key_head = key_head_begin; key_head < key_head_end; key_head++) {
                for (size_t blk = 0; blk < static_cast<size_t>(n_complete);) {
                    const auto [page, slot] = block_address(p, plan, static_cast<int64_t>(blk));
                    const size_t run = std::min(static_cast<size_t>(n_complete) - blk, slots - slot);
                    qsa::score_run<T>(buf.q_normed + key_head * heads_per_key * D_idx,
                                      p.summary_cache.ptr<T>(page, key_head, slot),
                                      run,
                                      heads_per_key,
                                      D_idx,
                                      buf.score_f32 + blk,
                                      buf.stage);
                    blk += run;
                }
            }
            for (int64_t blk = 0; blk < n_complete; blk++) {
                buf.scores[blk] = {buf.score_f32[blk] * scale, static_cast<int32_t>(blk)};
            }
            k = qsa::select_topk(buf.scores, buf.chosen, static_cast<size_t>(n_complete), c.block_topk, false);
        }

        std::copy_n(buf.chosen, k, p.sel_indices.ptr<int32_t>(token, sel_head, 0));
        *p.sel_count.ptr<int32_t>(token, sel_head) = static_cast<int32_t>(k);
    }
}

// The computation of the node, T is the precision of the summary blocks.
template <typename T>
void run_indexer(const PagedPorts& ports,
                 const Config& config,
                 const PagedScratchLayout& layout,
                 uint8_t* memory,
                 [[maybe_unused]] const QsaBrgemm* brgemm,
                 const ov::intel_cpu::CpuParallel& parallel) {
    const size_t tokens = num_tokens(ports);
    const auto stats = ov::intel_cpu::node::qsa::paged::seq_stats(ports, config);
    const Scratch scratch(std::in_place_type<T>, ports, config, layout, memory);

    std::fill_n(ports.sel_indices.ptr<int32_t>(), tokens * config.num_selection_heads * config.block_topk, -1);
    std::fill_n(ports.sel_count.ptr<int32_t>(), tokens * config.num_selection_heads, 0);

    // Phase 1: the raw keys of the tokens (and the rope rows of the blocks they start) go to the caches; disjoint
    // slots.
    parallel.parallel_for(tokens, [&](size_t token) {
        if (scratch.token_seq[token] >= 0) {
            store_token_rows<T>(ports, config, scratch, token);
        }
    });
    // Phase 2: the blocks this call completes are independent of each other (they read the raw rows of this and of
    // earlier calls, written by phase 1 or before).
    if (stats.max_new_blocks > 0) {
        parallel.parallel_for3d(num_sequences(ports),
                                index_key_heads(ports, config),
                                static_cast<size_t>(stats.max_new_blocks),
                                [&](size_t seq, size_t key_head, size_t j) {
                                    if (static_cast<int64_t>(j) < seq_plan(ports, config, seq).new_blocks) {
                                        pool_block<T>(ports, config, scratch, seq, key_head, j);
                                    }
                                });
    }
    // Phase 3: a token scores the complete blocks up to the one it completes, all written by now.
#ifdef OPENVINO_ARCH_X86_64
    if (brgemm != nullptr) {
        run_brgemm_scorer<T>(*brgemm, ports, config, stats, scratch, parallel);
        return;
    }
#endif
    parallel.parallel_for(tokens, [&](size_t token) {
        if (scratch.token_seq[token] >= 0) {
            select_blocks<T>(ports, config, scratch, token);
        }
    });
}

}  // namespace

void paged_qsa_indexer_compute(const ov::intel_cpu::node::qsa::paged::PagedPorts& ports,
                               const ov::op::internal::PagedQSAIndexer::Config& config,
                               const ov::intel_cpu::node::qsa::paged::PagedScratchLayout& layout,
                               uint8_t* scratch,
                               const ov::intel_cpu::node::qsa::QsaBrgemm* brgemm,
                               const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    // The precision of the summary blocks: the one the block scoring works in.
    const auto precision = ports.summary_cache.get_precision();
    if (precision == ov::element::f32) {
        run_indexer<float>(ports, config, layout, scratch, brgemm, *cpu_parallel);
    } else if (precision == ov::element::f16) {
        run_indexer<ov::float16>(ports, config, layout, scratch, brgemm, *cpu_parallel);
    } else if (precision == ov::element::bf16) {
        run_indexer<ov::bfloat16>(ports, config, layout, scratch, brgemm, *cpu_parallel);
    } else {
        OPENVINO_THROW("PagedQSAIndexer: unsupported summary precision ", precision);
    }
}

}  // namespace ov::Extensions::Cpu::XARCH
