// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// The kernel of the PagedQSAIndexer node: the per-token math of the QSAIndexer (see qsa_indexer_kernel.cpp) over caches
// addressed through the PagedAttention page table. A call appends the new raw keys, pools the blocks they complete, and
// keeps each token's block_topk best blocks. prepare() does the first two, score_tokens() the last
// one (the brgemm scorer of brgemm_scorer.hpp replaces it on AVX-512). Cross-compiled: one source for
// AVX-512, AVX2 and plain C++.

#include "paged_qsa_indexer_kernel.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "cache/multi_cache.h"
#include "cpu_parallel.hpp"
#include "nodes/kernels/sparse_attn/brgemm_scorer.hpp"
#include "nodes/kernels/sparse_attn/qsa_executor.hpp"
#include "nodes/kernels/sparse_attn/qsa_kernel_utils.hpp"
#include "nodes/kernels/sparse_attn/sparse_attn_common.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

namespace {

using ov::Extensions::Cpu::qsa::Config;
using ov::Extensions::Cpu::qsa::PagedIndexerArgs;
using ov::Extensions::Cpu::qsa::PagedQSAIndexerExecutor;
using ov::intel_cpu::CpuParallelPtr;
using ov::intel_cpu::div_up;
using ov::intel_cpu::rnd_up;
using qsa_kernel::block_address;
using qsa_kernel::BlockScore;
using qsa_kernel::row_view;
using qsa_kernel::seq_plan;
using qsa_kernel::token_address;

// The buffers of one worker thread's slice of the scratch; the stages run one after the other, so they share them.
struct ThreadBuffers {
    float* q_normed;     // [H_iq, D_idx] normalized + rotated index queries
    float* row_buf;      // [D_idx] a converted row
    float* summary_buf;  // [D_idx] a pooled summary row
    float* pooled;       // [D_idx] the mean of the raw keys of a block
    float* cos_sin;      // [2 * rotary_dim] the interleaved cos / sin row of a block
    float* cos_tmp;      // [rotary_dim] a cos row
    float* sin_tmp;      // [rotary_dim] a sin row
    float* stage;        // [kScoreRows, D_idx] summary rows converted to f32 (the scoring)
    float* score_f32;    // [blocks] the scores of the blocks of a token and selection head
    BlockScore* scores;  // [blocks] (score * scale, block), what the top-k selects from
    int32_t* chosen;     // [block_topk]
    size_t bytes;        // the size of the slice

    // The buffers of the slice that starts at `base`; with a null base only `bytes` is meaningful.
    static ThreadBuffers partition(uint8_t* base, const Config& c, size_t max_blocks) {
        qsa_kernel::ScratchPartition slice(base);
        ThreadBuffers b{};
        b.q_normed = slice.take<float>(c.index_query_heads * c.head_dim);
        b.row_buf = slice.take<float>(c.head_dim);
        b.summary_buf = slice.take<float>(c.head_dim);
        b.pooled = slice.take<float>(c.head_dim);
        b.cos_sin = slice.take<float>(2 * c.rotary_dim);
        b.cos_tmp = slice.take<float>(c.rotary_dim);
        b.sin_tmp = slice.take<float>(c.rotary_dim);
        b.stage = slice.take<float>(qsa_kernel::kScoreRows * c.head_dim);
        b.score_f32 = slice.take<float>(max_blocks);
        b.scores = slice.take<BlockScore>(max_blocks);
        b.chosen = slice.take<int32_t>(c.block_topk);
        b.bytes = slice.bytes();
        return b;
    }
};

// The scratch of a call, bound to its memory: [ token -> sequence array | f32 norm weights | one slice per worker
// thread ].
template <typename T>
struct Scratch {
    static size_t bytes(const PagedIndexerArgs& call, const Config& c) {
        return threads_offset(call, c) + c.num_threads * ThreadBuffers::partition(nullptr, c, call.max_blocks).bytes;
    }

    Scratch(const PagedIndexerArgs& call, const Config& c, uint8_t* memory)
        : token_seq(reinterpret_cast<int32_t*>(memory)),
          m_config(c),
          m_max_blocks(static_cast<size_t>(call.max_blocks)),
          m_thread_bytes(ThreadBuffers::partition(nullptr, c, m_max_blocks).bytes),
          m_threads(memory + threads_offset(call, c)) {
        const size_t D_idx = c.head_dim;
        auto* const q_norm = reinterpret_cast<float*>(memory + norm_weights_offset(call));
        auto* const k_norm = q_norm + D_idx;
        // f32 weights are used in place; 16-bit ones are converted once per call into the scratch
        q_norm_w = qsa_kernel::row_as_f32(call.q_norm_weight.ptr<T>(), D_idx, q_norm);
        k_norm_w = qsa_kernel::row_as_f32(call.k_norm_weight.ptr<T>(), D_idx, k_norm);
        const size_t tokens = call.q_idx.size(0);
        std::fill_n(token_seq, tokens, static_cast<int32_t>(-1));
        for (size_t seq = 0; seq < call.subsequence_begins.size(0) - 1; seq++) {
            const auto plan = seq_plan(call, c, seq);
            for (int64_t t = plan.token_begin; t < plan.token_begin + plan.q_len; t++) {
                if (static_cast<size_t>(t) < tokens) {
                    token_seq[t] = static_cast<int32_t>(seq);
                }
            }
        }
    }

    // This worker thread's slice of the scratch.
    [[nodiscard]] ThreadBuffers thread() const {
        return ThreadBuffers::partition(m_threads + static_cast<size_t>(parallel_get_thread_num()) * m_thread_bytes,
                                        m_config,
                                        m_max_blocks);
    }

    int32_t* token_seq;  // per token: its sequence, -1 for a token that belongs to none
    const float* q_norm_w = nullptr;
    const float* k_norm_w = nullptr;

private:
    static size_t norm_weights_offset(const PagedIndexerArgs& call) {
        return rnd_up(call.q_idx.size(0) * sizeof(int32_t), qsa_kernel::ScratchPartition::kAlign);
    }
    static size_t threads_offset(const PagedIndexerArgs& call, const Config& c) {
        return rnd_up(norm_weights_offset(call) + 2 * c.head_dim * sizeof(float), qsa_kernel::ScratchPartition::kAlign);
    }

    const Config& m_config;
    size_t m_max_blocks;
    size_t m_thread_bytes;
    uint8_t* m_threads;  // the slices of the worker threads
};

// ---- Phase 1, per token: append the raw index keys to the paged cache. A token that starts a block also stores its
// interleaved cos/sin row, for the block's summary to use once the block completes. A block that can never complete
// is skipped. ----
template <typename T>
void store_token_rows(const PagedIndexerArgs& call, const Config& c, const Scratch<T>& scratch, size_t token) {
    const auto plan = seq_plan(call, c, static_cast<size_t>(scratch.token_seq[token]));
    const int64_t logical_pos = plan.past_len + (static_cast<int64_t>(token) - plan.token_begin);
    const auto buf = scratch.thread();
    const size_t D_idx = c.head_dim;
    const size_t rotary_dim = c.rotary_dim;

    if (logical_pos % plan.key_block_size == 0) {
        const int64_t block_idx = logical_pos / plan.key_block_size;
        if (block_idx < plan.block_capacity) {
            float* const cos_sin = buf.cos_sin;
            const float* const cos_row = row_view<T>(call.rope_cos, rotary_dim, buf.cos_tmp, token);
            const float* const sin_row = row_view<T>(call.rope_sin, rotary_dim, buf.sin_tmp, token);
            for (size_t j = 0; j < rotary_dim; j++) {
                cos_sin[2 * j] = cos_row[j];
                cos_sin[2 * j + 1] = sin_row[j];
            }
            const auto [page, slot] = block_address(call, plan, block_idx);
            cvt_copy(call.indexer_rope_cache.ptr<T>(page, slot), cos_sin, 1, 2 * rotary_dim, 0, 0);
        }
    }
    const auto [cur_page, cur_slot] = token_address(call, plan, logical_pos);
    for (size_t key_head = 0; key_head < c.index_key_heads; key_head++) {
        const float* const row = row_view<T>(call.k_idx, D_idx, buf.row_buf, token, key_head * D_idx);
        cvt_copy(call.raw_k_cache.ptr<T>(cur_page, key_head, cur_slot), row, 1, D_idx, 0, 0);
    }
}

// ---- Phase 2, per (sequence, index-key head, block completed by the call): the summary of the block is the mean of
// its raw keys, RMSNorm'ed and RoPE'd with the coefficients of its first token. The blocks are independent of each
// other. ----
template <typename T>
void pool_block(const PagedIndexerArgs& call,
                const Config& c,
                const Scratch<T>& scratch,
                size_t seq,
                size_t key_head,
                size_t j) {
    const auto plan = seq_plan(call, c, seq);
    const auto buf = scratch.thread();
    float* const pooled = buf.pooled;
    const size_t D_idx = c.head_dim;
    const size_t rotary_dim = c.rotary_dim;

    const int64_t block_idx = plan.past_blocks + static_cast<int64_t>(j);
    const int64_t block_start = block_idx * plan.key_block_size;
    const auto [block_page, block_slot] = block_address(call, plan, block_idx);
    std::fill_n(pooled, D_idx, 0.0F);
    for (int64_t pos = block_start; pos < block_start + plan.key_block_size; pos++) {
        const auto [p_page, p_slot] = token_address(call, plan, pos);
        qsa_kernel::add_inplace(pooled,
                                row_view<T>(call.raw_k_cache, D_idx, buf.row_buf, p_page, key_head, p_slot),
                                D_idx);
    }
    qsa_kernel::scale_inplace(pooled, 1.0F / static_cast<float>(plan.key_block_size), D_idx);
    // the cos / sin of the first token of the block were stored interleaved; the RoPE takes them apart
    const float* const cos_sin =
        row_view<T>(call.indexer_rope_cache, 2 * rotary_dim, buf.cos_sin, block_page, block_slot);
    for (size_t k = 0; k < rotary_dim; k++) {
        buf.cos_tmp[k] = cos_sin[2 * k];
        buf.sin_tmp[k] = cos_sin[2 * k + 1];
    }
    qsa_kernel::rms_rope(pooled, buf.summary_buf, scratch.k_norm_w, D_idx, rotary_dim, c.eps, buf.cos_tmp, buf.sin_tmp);
    cvt_copy(call.summary_cache.ptr<T>(block_page, key_head, block_slot), buf.summary_buf, 1, D_idx, 0, 0);
}

// ---- Phase 3, per token: RMSNorm + partial RoPE of every index-query head, then per selection head the score of
// each complete block (sum over its heads of max(0, q . summary)) and the top-k blocks in ascending order. ----
template <typename T>
void select_blocks(const PagedIndexerArgs& call, const Config& c, const Scratch<T>& scratch, size_t token) {
    const auto plan = seq_plan(call, c, static_cast<size_t>(scratch.token_seq[token]));
    const auto buf = scratch.thread();
    const size_t D_idx = c.head_dim;
    const size_t rotary_dim = c.rotary_dim;
    const size_t index_keys = c.index_key_heads;
    const size_t index_queries = c.index_query_heads;
    const size_t heads_per_key = index_queries / index_keys;
    const size_t selection_heads = c.num_selection_heads;

    const float* const cos_row = row_view<T>(call.rope_cos, rotary_dim, buf.cos_tmp, token);
    const float* const sin_row = row_view<T>(call.rope_sin, rotary_dim, buf.sin_tmp, token);
    for (size_t h = 0; h < index_queries; h++) {
        const float* const q_row = row_view<T>(call.q_idx, D_idx, buf.row_buf, token, h * D_idx);
        qsa_kernel::rms_rope(q_row,
                             buf.q_normed + h * D_idx,
                             scratch.q_norm_w,
                             D_idx,
                             rotary_dim,
                             c.eps,
                             cos_row,
                             sin_row,
                             c.query_norm,
                             c.query_rope);
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
                    const auto [page, slot] = block_address(call, plan, static_cast<int64_t>(blk));
                    const size_t run = std::min(static_cast<size_t>(n_complete) - blk, slots - slot);
                    qsa_kernel::score_run<T>(buf.q_normed + key_head * heads_per_key * D_idx,
                                             call.summary_cache.ptr<T>(page, key_head, slot),
                                             run,
                                             heads_per_key,
                                             D_idx,
                                             buf.score_f32 + blk,
                                             buf.stage);
                    blk += run;
                }
            }
            for (int64_t blk = 0; blk < n_complete; blk++) {
                buf.scores[blk] = {buf.score_f32[blk] * c.scale, static_cast<int32_t>(blk)};
            }
            k = qsa_kernel::select_topk(buf.scores, buf.chosen, static_cast<size_t>(n_complete), c.block_topk);
        }

        std::copy_n(buf.chosen, k, call.sel_indices.ptr<int32_t>(token, sel_head, 0));
        *call.sel_count.ptr<int32_t>(token, sel_head) = static_cast<int32_t>(k);
    }
}

// T is the precision of the floating-point ports.
template <typename T>
class PagedExecutorImpl final : public PagedQSAIndexerExecutor {
public:
    // The scorer on brgemm gets its kernels from `cache`: none where the variant, the CPU, the precision or the head
    // counts do not allow it.
    PagedExecutorImpl(const Config& config, const ov::intel_cpu::MultiCachePtr& cache)
        : m_config(config),
          m_brgemm_scorer(BrgemmScorer::create(config, cache)) {}

    [[nodiscard]] ov::intel_cpu::impl_desc_type impl_type() const override {
        return m_brgemm_scorer ? m_brgemm_scorer->impl_type() : variant_impl_type();
    }

    void find_batch_max_params(PagedIndexerArgs& call) const override {
        call.max_blocks = call.max_new_blocks = call.max_q_len = 0;
        for (size_t seq = 0; seq < call.subsequence_begins.size(0) - 1; seq++) {
            const auto plan = seq_plan(call, m_config, seq);
            OPENVINO_ASSERT(plan.total_blocks() <= plan.block_capacity,
                            "[CPU] paged_qsa_indexer: the completed blocks do not fit the sequence's cache blocks");
            call.max_blocks = std::max(call.max_blocks, plan.total_blocks());
            call.max_new_blocks = std::max(call.max_new_blocks, plan.new_blocks);
            call.max_q_len = std::max(call.max_q_len, plan.q_len);
        }
    }

    // The stages and the scorer run one after the other on the same scratch.
    [[nodiscard]] size_t scratch_bytes(const PagedIndexerArgs& call) const override {
        const size_t bytes = Scratch<T>::bytes(call, m_config);
        return scores_on_brgemm(call) ? std::max(bytes, m_brgemm_scorer->scratch_bytes(call)) : bytes;
    }

    void execute(const PagedIndexerArgs& call, uint8_t* memory, const CpuParallelPtr& cpu_parallel) const override {
        prepare(call, memory, cpu_parallel);
        if (scores_on_brgemm(call)) {
            m_brgemm_scorer->score(call, memory, cpu_parallel);
        } else {
            score_tokens(call, memory, cpu_parallel);
        }
    }

private:
    using BrgemmScorer = brgemm_scorer_impl::BrgemmScorer<brgemm_scorer_impl::PagedSource<T>>;

    // A call that has complete blocks to score goes to the scorer on brgemm, the others are scored by the stages.
    [[nodiscard]] bool scores_on_brgemm(const PagedIndexerArgs& call) const {
        return m_brgemm_scorer && call.max_blocks > 0;
    }

    void prepare(const PagedIndexerArgs& call, uint8_t* memory, const CpuParallelPtr& cpu_parallel) const {
        const auto& parallel = *cpu_parallel;
        const size_t tokens = call.q_idx.size(0);
        const Scratch<T> scratch(call, m_config, memory);

        std::fill_n(call.sel_indices.ptr<int32_t>(), tokens * m_config.num_selection_heads * m_config.block_topk, -1);
        std::fill_n(call.sel_count.ptr<int32_t>(), tokens * m_config.num_selection_heads, 0);

        // Phase 1: the raw keys of the tokens (and the rope rows of the blocks they start) go to the caches; disjoint
        // slots.
        parallel.parallel_for(tokens, [&](size_t token) {
            if (scratch.token_seq[token] >= 0) {
                store_token_rows<T>(call, m_config, scratch, token);
            }
        });
        // Phase 2: the blocks this call completes are independent of each other (they read the raw rows of this and of
        // earlier calls, written by phase 1 or before).
        if (call.max_new_blocks > 0) {
            parallel.parallel_for3d(call.subsequence_begins.size(0) - 1,
                                    m_config.index_key_heads,
                                    static_cast<size_t>(call.max_new_blocks),
                                    [&](size_t seq, size_t key_head, size_t j) {
                                        if (static_cast<int64_t>(j) < seq_plan(call, m_config, seq).new_blocks) {
                                            pool_block<T>(call, m_config, scratch, seq, key_head, j);
                                        }
                                    });
        }
    }

    // A token scores the complete blocks up to the one it completes, all written by prepare().
    void score_tokens(const PagedIndexerArgs& call, uint8_t* memory, const CpuParallelPtr& cpu_parallel) const {
        const Scratch<T> scratch(call, m_config, memory);
        cpu_parallel->parallel_for(call.q_idx.size(0), [&](size_t token) {
            if (scratch.token_seq[token] >= 0) {
                select_blocks<T>(call, m_config, scratch, token);
            }
        });
    }

    Config m_config;
    std::unique_ptr<BrgemmScorer> m_brgemm_scorer;
};

}  // namespace

std::shared_ptr<ov::Extensions::Cpu::qsa::PagedQSAIndexerExecutor> make_paged_indexer_executor(
    const ov::Extensions::Cpu::qsa::Config& config,
    const ov::intel_cpu::MultiCachePtr& cache) {
    // The precision of the summary blocks: the one the block scoring works in.
    if (config.precision == ov::element::f32) {
        return std::make_shared<PagedExecutorImpl<float>>(config, cache);
    }
    if (config.precision == ov::element::f16) {
        return std::make_shared<PagedExecutorImpl<ov::float16>>(config, cache);
    }
    if (config.precision == ov::element::bf16) {
        return std::make_shared<PagedExecutorImpl<ov::bfloat16>>(config, cache);
    }
    OPENVINO_THROW("PagedQSAIndexer: unsupported summary precision ", config.precision);
}

}  // namespace ov::Extensions::Cpu::XARCH
