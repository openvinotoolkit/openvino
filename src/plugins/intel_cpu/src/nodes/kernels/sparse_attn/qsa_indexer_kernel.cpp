// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// The kernel of the QSAIndexer node. One call, per row b of the batch:
//   1. appends the window of L tokens (q_idx [B, L, H_iq, D], k_idx [B, L, H_ik, D]) to the key state: every `ratio`
//      keys complete a block, and the index key of the block (pooled keys -> RMSNorm -> partial RoPE) is stored in
//      summary_cache [B, H_ik, NB, D];
//   2. scores, for every token of the window, the complete blocks it can see (q . index key, summed over the query
//      heads of a selection head) and keeps the block_topk best block ids: sel_indices [B, H_sel, L, block_topk],
//      sel_count [B, H_sel, L].
// Stage 1 is the prepare part of execute(), stage 2 run_token_scorer() (the brgemm scorer of brgemm_scorer.hpp replaces
// it on AVX-512). The kernel is cross-compiled: one source for AVX-512, AVX2 and plain C++.

#include "qsa_indexer_kernel.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
using ov::Extensions::Cpu::qsa::IndexerArgs;
using ov::Extensions::Cpu::qsa::QSAIndexerExecutor;
using ov::intel_cpu::CpuParallel;
using ov::intel_cpu::CpuParallelPtr;
using ov::intel_cpu::div_up;
using ov::intel_cpu::rnd_up;
using qsa_kernel::batch_entry_plan;
using qsa_kernel::BatchEntryPlan;
using qsa_kernel::BlockScore;
using qsa_kernel::rope_batch;

// The buffers of one worker thread's slice of the scratch. The stages run one after the other, so they share them; the
// (score, block) pairs and the chosen ids are for the token scorer.
struct ThreadBuffers {
    float* q_normed;  // [H_iq, D_idx] normalized + rotated index queries (stage 1: the first row is the pooled block)
    float* row_buf;   // [D_idx] a converted row
    float* rope_cos_sin;  // [2 * rotary_dim] cos / sin of the first token of a block (stage 2: of the token)
    float* stage;         // [kScoreRows, D_idx] summary rows converted to f32 (the scoring)
    float* score_f32;     // [blocks] the scores of the blocks of a token and selection head
    BlockScore* scores;   // [blocks] (score * scale, block), what the top-k selects from
    int32_t* chosen;      // [block_topk]
    size_t bytes;         // the size of the slice

    // The buffers of the slice that starts at `base`; with a null base only `bytes` is meaningful.
    static ThreadBuffers partition(uint8_t* base, const Config& c, size_t max_blocks) {
        qsa_kernel::ScratchPartition slice(base);
        ThreadBuffers b{};
        b.q_normed = slice.take<float>(c.index_query_heads * c.head_dim);
        b.row_buf = slice.take<float>(c.head_dim);
        b.rope_cos_sin = slice.take<float>(2 * c.rotary_dim);
        b.stage = slice.take<float>(qsa_kernel::kScoreRows * c.head_dim);
        b.score_f32 = slice.take<float>(max_blocks);
        b.scores = slice.take<BlockScore>(max_blocks);
        b.chosen = slice.take<int32_t>(c.block_topk);
        b.bytes = slice.bytes();
        return b;
    }
};

// The scratch of a call, bound to its memory: [ f32 q / k norm weights | one slice per worker thread ].
template <typename T>
struct Scratch {
    static size_t bytes(const IndexerArgs& call, const Config& c) {
        return threads_offset(c) + c.num_threads * ThreadBuffers::partition(nullptr, c, call.max_blocks).bytes;
    }

    Scratch(const IndexerArgs& call, const Config& c, uint8_t* memory)
        : m_config(c),
          m_max_blocks(static_cast<size_t>(call.max_blocks)),
          m_thread_bytes(ThreadBuffers::partition(nullptr, c, m_max_blocks).bytes),
          m_threads(memory + threads_offset(c)) {
        const size_t D_idx = c.head_dim;
        auto* const norm_weights = reinterpret_cast<float*>(memory);
        q_norm_w = qsa_kernel::row_as_f32(call.q_norm_weight.ptr<T>(), D_idx, norm_weights);
        k_norm_w = qsa_kernel::row_as_f32(call.k_norm_weight.ptr<T>(), D_idx, norm_weights + D_idx);
    }

    // This worker thread's slice of the scratch.
    [[nodiscard]] ThreadBuffers thread() const {
        return ThreadBuffers::partition(m_threads + static_cast<size_t>(parallel_get_thread_num()) * m_thread_bytes,
                                        m_config,
                                        m_max_blocks);
    }

    const float* q_norm_w = nullptr;  // f32 [D_idx] (the tensor itself when f32)
    const float* k_norm_w = nullptr;

private:
    static size_t threads_offset(const Config& c) {
        return rnd_up(2 * c.head_dim * sizeof(float), qsa_kernel::ScratchPartition::kAlign);
    }

    const Config& m_config;
    size_t m_max_blocks;
    size_t m_thread_bytes;
    uint8_t* m_threads;  // the slices of the worker threads
};

// The pending window of a row is [the `tail` rows of the incomplete-block state | the real tokens of this call].
// Raw K row at pending position `pos` (a view of the tensor when it is f32, `buf` otherwise):
template <typename T>
const float* pending_row(const IndexerArgs& call,
                         const Config& c,
                         const BatchEntryPlan& plan,
                         size_t b,
                         size_t h,
                         int64_t pos,
                         float* buf) {
    const size_t D_idx = c.head_dim;
    if (pos < plan.tail) {
        return qsa_kernel::row_as_f32(call.past_raw_k_in.ptr<T>(b, h, pos), D_idx, buf);
    }
    return qsa_kernel::row_as_f32(call.k_idx.ptr<T>(b, plan.num_padding + pos - plan.tail, h), D_idx, buf);
}

// [rope_cos, rope_sin] of the first token of the block starting at pending position `pos`, in the layout of the
// past_incomplete_rope state:
template <typename T>
void get_first_token_cos_sin(const IndexerArgs& call,
                             const Config& c,
                             const BatchEntryPlan& plan,
                             size_t b,
                             int64_t pos,
                             float* dst) {
    const size_t rotary_dim = c.rotary_dim;
    if (pos < plan.tail) {
        cvt_copy(dst, call.past_rope_in.ptr<T>(b), 1, 2 * rotary_dim, 0, 0);
    } else {
        const int64_t l = plan.num_padding + pos - plan.tail;
        cvt_copy(dst, call.rope_cos.ptr<T>(rope_batch(call.rope_cos, b), l), 1, rotary_dim, 0, 0);
        cvt_copy(dst + rotary_dim, call.rope_sin.ptr<T>(rope_batch(call.rope_sin, b), l), 1, rotary_dim, 0, 0);
    }
}

// ---- Phase 0: the summary_cache output [B, H_ik, blocks_out, D] starts as the blocks of the earlier calls (the input
// [B, H_ik, blocks_in, D]), followed by zeros; the selection outputs start empty (indices -1, counts 0) ----
// The copy of a long cache is split in block ranges to use the threads (decode has a single row and head).
template <typename T>
void initialize_summary_cache(const IndexerArgs& call, const Config& c, const CpuParallel& parallel) {
    const size_t B = call.summary_cache.size(0);
    const size_t H_ik = c.index_key_heads;
    const size_t blocks_out = call.summary_cache.size(2);
    const size_t blocks_in = call.summary_cache_in.size(2);

    std::fill_n(
        call.sel_indices.ptr<int32_t>(),
        call.sel_indices.size(0) * call.sel_indices.size(1) * call.sel_indices.size(2) * call.sel_indices.size(3),
        -1);
    std::fill_n(call.sel_count.ptr<int32_t>(),
                call.sel_count.size(0) * call.sel_count.size(1) * call.sel_count.size(2),
                0);

    const size_t block_bytes = c.head_dim * sizeof(T);
    const size_t copy_blocks = std::min(blocks_in, blocks_out);
    constexpr size_t kPartBytes = 256UL << 10;
    const size_t parts = std::clamp<size_t>(div_up(blocks_out * block_bytes, kPartBytes), 1, c.num_threads);
    const size_t blocks_per_part = div_up(blocks_out, parts);
    const auto* src_base = call.summary_cache_in.ptr<uint8_t>();
    auto* dst_base = call.summary_cache.ptr<uint8_t>();
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
void append_complete_blocks(const IndexerArgs& call,
                            const Config& c,
                            const Scratch<T>& scratch,
                            const CpuParallel& parallel) {
    const size_t D_idx = c.head_dim;
    const auto r = static_cast<int64_t>(c.key_block_size);
    parallel.parallel_for3d(
        call.q_idx.size(0),
        c.index_key_heads,
        static_cast<size_t>(call.max_new_blocks),
        [&](size_t b, size_t h, size_t j) {
            const auto plan = batch_entry_plan(call, c, b);
            if (static_cast<int64_t>(j) >= plan.new_blocks) {
                return;
            }
            const auto buf = scratch.thread();
            float* const pooled = buf.q_normed;

            const int64_t block_start = static_cast<int64_t>(j) * r;
            std::fill_n(pooled, D_idx, 0.0F);
            for (int64_t pos = block_start; pos < block_start + r; pos++) {
                qsa_kernel::add_inplace(pooled, pending_row<T>(call, c, plan, b, h, pos, buf.row_buf), D_idx);
            }
            qsa_kernel::scale_inplace(pooled, 1.0F / static_cast<float>(r), D_idx);

            get_first_token_cos_sin<T>(call, c, plan, b, block_start, buf.rope_cos_sin);
            qsa_kernel::rms_rope(pooled,
                                 pooled,
                                 scratch.k_norm_w,
                                 D_idx,
                                 c.rotary_dim,
                                 c.eps,
                                 buf.rope_cos_sin,
                                 buf.rope_cos_sin + c.rotary_dim);
            qsa_kernel::store_row(pooled,
                                  call.summary_cache.ptr<T>(b, h, static_cast<size_t>(plan.past_blocks) + j),
                                  D_idx);
        });
}

// ---- Phase 1b: fully overwrite the fixed-size incomplete-block state: the raw keys [B, H_ik, ratio - 1, D] of the
// tokens after the last complete block (zero-padded) and the cos / sin [B, 2 * Dr] of the first of them ----
template <typename T>
void update_incomplete_block(const IndexerArgs& call,
                             const Config& c,
                             const Scratch<T>& scratch,
                             const CpuParallel& parallel) {
    const size_t D_idx = c.head_dim;
    const size_t H_ik = c.index_key_heads;
    const auto r = static_cast<int64_t>(c.key_block_size);
    parallel.parallel_for(call.q_idx.size(0), [&](size_t b) {
        const auto plan = batch_entry_plan(call, c, b);
        const auto buf = scratch.thread();

        const int64_t rest = plan.tail + plan.current - plan.new_blocks * r;
        const int64_t rest_start = plan.new_blocks * r;
        for (int64_t pos = 0; pos < r - 1; pos++) {
            for (size_t h = 0; h < H_ik; h++) {
                T* const dst = call.past_raw_k.ptr<T>(b, h, pos);
                if (pos < rest) {
                    qsa_kernel::store_row(pending_row<T>(call, c, plan, b, h, rest_start + pos, buf.row_buf),
                                          dst,
                                          D_idx);
                } else {
                    std::fill_n(dst, D_idx, T{});
                }
            }
        }
        if (rest > 0) {
            get_first_token_cos_sin<T>(call, c, plan, b, rest_start, buf.rope_cos_sin);
        } else {
            std::fill_n(buf.rope_cos_sin, 2 * c.rotary_dim, 0.0F);
        }
        qsa_kernel::store_row(buf.rope_cos_sin, call.past_rope.ptr<T>(b), 2 * c.rotary_dim);
    });
}

// ---- Phase 2 per real token: the query RMSNorm + RoPE, then per selection head the scoring of the complete blocks
// and the top-k ----
template <typename T>
void run_token_scorer(const IndexerArgs& call,
                      const Config& c,
                      const Scratch<T>& scratch,
                      const CpuParallel& parallel) {
    const size_t D_idx = c.head_dim;
    const size_t H_iq = c.index_query_heads;
    const size_t H_ik = c.index_key_heads;
    const size_t G = H_iq / H_ik;  // index query heads per index key head
    const size_t rotary_dim = c.rotary_dim;
    const auto r = static_cast<int64_t>(c.key_block_size);
    parallel.parallel_for2d(call.q_idx.size(0), call.q_idx.size(1), [&](size_t b, size_t l_u) {
        const auto plan = batch_entry_plan(call, c, b);
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
            qsa_kernel::row_as_f32(call.rope_cos.ptr<T>(rope_batch(call.rope_cos, b), l), rotary_dim, rope_cos_buf);
        const float* const sin_row =
            qsa_kernel::row_as_f32(call.rope_sin.ptr<T>(rope_batch(call.rope_sin, b), l), rotary_dim, rope_sin_buf);
        for (size_t h = 0; h < H_iq; h++) {
            const float* const q_row = qsa_kernel::row_as_f32(call.q_idx.ptr<T>(b, l, h), D_idx, buf.row_buf);
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

        // ---- Phase 2b: per selection head sh and complete block blk visible to the token:
        //   score = scale * sum over the query heads h of sh of max(0, q_normed[h] . summary_cache[kv(h), blk])
        // then the top-k blocks by score (ties: the lower block first) ----
        // The complete blocks visible to this token: the earlier calls' plus those finished up to this token.
        const int64_t n_complete = plan.blocks_visible(l - plan.num_padding, r);
        for (size_t sh = 0; sh < c.num_selection_heads; sh++) {
            // The index key heads of the selection head, and so their query heads: contiguous ranges.
            const size_t hk_begin = div_up(sh * H_ik, c.num_selection_heads);
            const size_t hk_end = div_up((sh + 1) * H_ik, c.num_selection_heads);
            size_t k = 0;
            if (n_complete > 0) {
                // the scores of the complete blocks, summed over the index key heads of the selection head: the
                // blocks of a head are contiguous rows of the summary cache
                std::fill_n(buf.score_f32, static_cast<size_t>(n_complete), 0.0F);
                for (size_t hk = hk_begin; hk < hk_end; hk++) {
                    qsa_kernel::score_run<T>(buf.q_normed + hk * G * D_idx,
                                             call.summary_cache.ptr<T>(b, hk, 0),
                                             static_cast<size_t>(n_complete),
                                             G,
                                             D_idx,
                                             buf.score_f32,
                                             buf.stage);
                }
                for (int64_t blk = 0; blk < n_complete; blk++) {
                    buf.scores[blk] = {buf.score_f32[blk] * c.scale, static_cast<int32_t>(blk)};
                }
                k = qsa_kernel::select_topk(buf.scores, buf.chosen, static_cast<size_t>(n_complete), c.block_topk);
            }

            std::copy_n(buf.chosen, k, call.sel_indices.ptr<int32_t>(b, sh, l_u, 0));
            *call.sel_count.ptr<int32_t>(b, sh, l_u) = static_cast<int32_t>(k);
        }
    });
}

// T is the precision of the data ports.
template <typename T>
class DenseExecutorImpl final : public QSAIndexerExecutor {
public:
    // The scorer on brgemm gets its kernels from `cache`: none where the variant, the CPU, the precision or the head
    // counts do not allow it.
    DenseExecutorImpl(const Config& config, const ov::intel_cpu::MultiCachePtr& cache)
        : m_config(config),
          m_brgemm_scorer(BrgemmScorer::create(config, cache)) {}

    [[nodiscard]] ov::intel_cpu::impl_desc_type impl_type() const override {
        return m_brgemm_scorer ? m_brgemm_scorer->impl_type() : variant_impl_type();
    }

    void find_batch_max_params(IndexerArgs& call) const override {
        const auto blocks_out = call.summary_cache.size(2);
        call.max_blocks = call.max_new_blocks = call.max_q_len = 0;
        for (size_t b = 0; b < call.q_idx.size(0); b++) {
            const auto plan = batch_entry_plan(call, m_config, b);
            OPENVINO_ASSERT(static_cast<size_t>(plan.past_blocks + plan.new_blocks) <= blocks_out,
                            "QSAIndexer: ",
                            plan.past_blocks + plan.new_blocks,
                            " blocks do not fit the summary_cache output of ",
                            blocks_out,
                            " blocks");
            call.max_q_len = std::max(call.max_q_len, plan.current);
            call.max_blocks = std::max(call.max_blocks, plan.past_blocks + plan.new_blocks);
            call.max_new_blocks = std::max(call.max_new_blocks, plan.new_blocks);
        }
    }

    // The stages and the scorer run one after the other on the same scratch.
    [[nodiscard]] size_t scratch_bytes(const IndexerArgs& call) const override {
        const size_t bytes = Scratch<T>::bytes(call, m_config);
        return scores_on_brgemm(call) ? std::max(bytes, m_brgemm_scorer->scratch_bytes(call)) : bytes;
    }

    void execute(const IndexerArgs& call, uint8_t* memory, const CpuParallelPtr& cpu_parallel) const override {
        const Scratch<T> scratch(call, m_config, memory);
        initialize_summary_cache<T>(call, m_config, *cpu_parallel);
        append_complete_blocks<T>(call, m_config, scratch, *cpu_parallel);
        update_incomplete_block<T>(call, m_config, scratch, *cpu_parallel);
        if (scores_on_brgemm(call)) {
            m_brgemm_scorer->score(call, memory, cpu_parallel);
        } else {
            run_token_scorer<T>(call, m_config, scratch, *cpu_parallel);
        }
    }

private:
    using BrgemmScorer = brgemm_scorer_impl::BrgemmScorer<brgemm_scorer_impl::DenseSource<T>>;

    // A call that has complete blocks to score goes to the scorer on brgemm, the others are scored by the stages.
    [[nodiscard]] bool scores_on_brgemm(const IndexerArgs& call) const {
        return m_brgemm_scorer && call.max_blocks > 0;
    }

    Config m_config;
    std::unique_ptr<BrgemmScorer> m_brgemm_scorer;
};

}  // namespace

std::shared_ptr<ov::Extensions::Cpu::qsa::QSAIndexerExecutor> make_indexer_executor(
    const ov::Extensions::Cpu::qsa::Config& config,
    const ov::intel_cpu::MultiCachePtr& cache) {
    if (config.precision == ov::element::f32) {
        return std::make_shared<DenseExecutorImpl<float>>(config, cache);
    }
    if (config.precision == ov::element::f16) {
        return std::make_shared<DenseExecutorImpl<ov::float16>>(config, cache);
    }
    if (config.precision == ov::element::bf16) {
        return std::make_shared<DenseExecutorImpl<ov::bfloat16>>(config, cache);
    }
    OPENVINO_THROW("QSAIndexer: unsupported precision ", config.precision);
}

}  // namespace ov::Extensions::Cpu::XARCH
