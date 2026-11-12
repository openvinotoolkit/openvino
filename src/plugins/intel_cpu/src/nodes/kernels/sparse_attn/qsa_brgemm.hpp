// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

// The brgemm scoring of the QSAIndexer and the PagedQSAIndexer (x64): the configuration, the kernels (shared through
// the params cache) and the plan of the scratch. The computation itself is in the AVX-512 kernel qsa_brgemm_kernel.cpp,
// see there for the steps.

#include <algorithm>
#include <common/utils.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "cache/multi_cache.h"
#include "nodes/kernels/x64/brgemm_kernel.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "utils/general_utils.h"

namespace ov::Extensions::Cpu::qsa {

using ov::intel_cpu::BrgemmKernel;
using ov::intel_cpu::div_up;
using ov::intel_cpu::impl_desc_type;
using ov::intel_cpu::MultiCachePtr;
using ov::intel_cpu::rnd_up;

// The blocks one brgemm call scores: the M of the kernel (BrgemmKernel::get_mblk_size()).
inline constexpr size_t kScoreBlockRows = 32;
// Rows of the query tile of a call: (tokens x heads of a K head) up to this many.
inline constexpr size_t kScoreMaxCols = 128;
inline constexpr size_t kSelectBuckets = 256;

// The scorer of a node on brgemm: what it needs from the layers that share it, the brgemm kernels (JIT products that
// are expensive to build and depend only on the shape of their columns, the head dim and the precision) and the plan of
// the scratch. The layers of a model share it through the params cache of the graph context (like the SDPA and
// PagedAttention kernels), and a node fetches it once when it is created, not per call. Created only by get().
class QsaBrgemm {
public:
    // Everything that does not change from call to call; also the key of the scorer in the params cache.
    struct Params {
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
        bool operator==(const Params& rhs) const {
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

    struct ThreadOffsets {
        size_t rows, query_tile, packed_blocks, packed_query, gemm_scores, workspace, scratch_a, key_chunk_stage;
    };

    // The scorer of `params` from the params cache (built on first use, shared by the layers of the same
    // configuration), nullptr if this CPU or the configuration has no brgemm scoring. The selection heads have to be
    // covered by the K heads.
    static std::shared_ptr<const QsaBrgemm> get(const MultiCachePtr& cache, const Params& params) {
        if (params.index_key_heads == 0 || params.index_query_heads % params.index_key_heads != 0 ||
            params.selection_heads > params.index_key_heads || !score_kernel_supported(params.precision, params.dim)) {
            return nullptr;
        }
        const auto builder = [&cache](const Params& key) -> std::shared_ptr<const QsaBrgemm> {
            const size_t heads_per_key = key.index_query_heads / key.index_key_heads;
            const size_t tokens_per_call = score_tokens_per_call(heads_per_key);
            // `q_k_kernel` scores a full set of tokens (query rows: tokens_per_call * heads_per_key), `k_q_kernel` a
            // single token (heads_per_key rows): the decode step is one token wide.
            auto q_k_kernel = get_score_kernel(cache, key.precision, tokens_per_call * heads_per_key, key.dim, false);
            auto k_q_kernel = get_score_kernel(cache, key.precision, heads_per_key, key.dim, true);
            if (!q_k_kernel || !k_q_kernel) {
                return nullptr;
            }
            return std::shared_ptr<const QsaBrgemm>(  // NOLINT(modernize-make-shared): the constructor is private
                new QsaBrgemm(key, std::move(q_k_kernel), std::move(k_q_kernel)));
        };
        try {
            return cache->getOrCreate(params, builder).first;
        } catch (const ov::Exception&) {
            return nullptr;
        }
    }

    // The implementation type a node that scores on brgemm reports (execution graph, profiling): the matrix extension
    // of the kernels of `precision` on this CPU.
    static impl_desc_type impl_type(const ov::element::Type& precision) {
        using namespace dnnl::impl::cpu::x64;
        const bool amx = (precision == ov::element::bf16 && mayiuse(avx512_core_amx)) ||
                         (precision == ov::element::f16 && mayiuse(avx512_core_amx_fp16));
        return amx ? impl_desc_type::brgemm_avx512_amx : impl_desc_type::brgemm_avx512;
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
        p.score_tile_bytes = rnd_up(alloc_tokens * m_params.selection_heads * p.score_stride * sizeof(float), kAlign);
        if (needs_scoring(p) && p.score_tile_bytes != 0) {
            p.tiles_per_round = std::clamp<size_t>(kScoreBudgetBytes / (batch * p.score_tile_bytes), 1, p.tiles_total);
        }
        p.shared_bytes = p.tiles_per_round * batch * p.score_tile_bytes;
        p.thread_bytes =
            m_select_offset + rnd_up((p.padded_blocks + 1) * sizeof(float) + kSelectBuckets * sizeof(uint32_t), kAlign);
        return p;
    }

    // Scores are only needed when a token has more blocks than it can choose, or, to order the blocks by score, when it
    // has any.
    [[nodiscard]] bool needs_scoring(const Plan& plan) const {
        return static_cast<size_t>(plan.max_row_blocks) > (m_params.order_by_score ? 0 : m_params.block_topk);
    }

    // What the scorer kernel reads (qsa_brgemm_kernel.cpp).
    [[nodiscard]] const Params& params() const {
        return m_params;
    }
    [[nodiscard]] const std::shared_ptr<BrgemmKernel>& q_k_kernel() const {
        return m_q_k_kernel;
    }
    [[nodiscard]] const std::shared_ptr<BrgemmKernel>& k_q_kernel() const {
        return m_k_q_kernel;
    }
    [[nodiscard]] size_t heads_per_key() const {
        return m_heads_per_key;
    }
    [[nodiscard]] size_t tokens_per_call() const {
        return m_tokens_per_call;
    }
    [[nodiscard]] size_t tile_tokens() const {
        return m_tile_tokens;
    }
    [[nodiscard]] size_t packed_query_bytes() const {
        return m_packed_query_bytes;
    }
    [[nodiscard]] const ThreadOffsets& offsets() const {
        return m_offsets;
    }
    [[nodiscard]] size_t select_offset() const {
        return m_select_offset;
    }

    static constexpr size_t kAlign = 64;
    // Query slices (the tokens of one brgemm call each) an item prepares and scores.
    static constexpr size_t kQuerySlicesPerItem = 4;
    // Bound of the score matrices of the tokens in flight.
    static constexpr size_t kScoreBudgetBytes = 32UL << 20;

private:
    // The key of a score kernel in the params cache.
    struct ScoreKernelKey {
        size_t query_rows = 0;  // rows of the query tile: tokens x heads of a K head
        size_t dim = 0;         // head dim, the K of the product
        ov::element::Type precision;
        bool is_k_q_gemm =
            false;  // C = K.Q^T, the blocks are the A matrix (one token); else C = Q.K^T, the query tile is

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

    QsaBrgemm(const Params& params, std::shared_ptr<BrgemmKernel> q_k_kernel, std::shared_ptr<BrgemmKernel> k_q_kernel)
        : m_params(params),
          m_q_k_kernel(std::move(q_k_kernel)),
          m_k_q_kernel(std::move(k_q_kernel)),
          m_heads_per_key(params.index_query_heads / params.index_key_heads),
          m_tokens_per_call(score_tokens_per_call(m_heads_per_key)),
          m_tile_tokens(m_tokens_per_call * kQuerySlicesPerItem),
          m_packed_blocks_bytes(rnd_up(m_q_k_kernel->get_scratch_b_size(), kAlign)),
          m_packed_query_bytes(rnd_up(m_k_q_kernel->get_scratch_b_size(), kAlign)) {
        init_thread_layout();
    }

    // Tokens one brgemm call scores together for K heads of `heads_per_key` query heads.
    static size_t score_tokens_per_call(size_t heads_per_key) {
        return std::max<size_t>(1, kScoreMaxCols / heads_per_key);
    }

    // Whether this CPU runs the scorer for `precision` and head dim `dim`. The scorer kernel (qsa_brgemm_kernel.cpp) is
    // cross-compiled for AVX-512 only, so the dispatcher has to select that build too.
    static bool score_kernel_supported(const ov::element::Type& precision, size_t dim) {
        using namespace dnnl::impl::cpu::x64;
        if (dim % 2 != 0 || !ov::with_cpu_x86_avx512f()) {
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

    // The kernel of the q . k of 32 blocks and `query_rows` query rows, with the ReLU, from the cache; nullptr if it
    // can not be built. Two forms, by how many tokens a call scores:
    //   is_k_q_gemm == false (q.k, several tokens)  C[query_rows, 32] = max(0, query_tile[query_rows, D] x blocks[32,
    //   D]^T)
    //       A: the query tile, B: the transposed packed blocks. A row of C is 32 scores of one token.
    //   is_k_q_gemm == true (k.q, one token)        C[32, query_rows] = max(0, blocks[32, D] x query_tile[query_rows,
    //   D]^T)
    //       A: the blocks, read in place; B: the transposed packed query tile. A decode step does not make an extra
    //       pass over the cache (packing it would double its memory traffic, the cache is what a decode step is bound
    //       by).
    static std::shared_ptr<BrgemmKernel> get_score_kernel(const MultiCachePtr& cache,
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

    // Layout of a thread slice: f32 rows | query tiles | packed blocks | packed query | brgemm output | workspace |
    // A scratch | zero-padded chunk | selection scratch (its size follows from the call, see plan()).
    void init_thread_layout() {
        const size_t dim = m_params.dim;
        const size_t element = m_params.precision.size();
        const size_t query_rows = m_tokens_per_call * m_heads_per_key;
        size_t offset = 0;
        auto reserve = [&offset](size_t bytes) {
            const size_t start = offset;
            offset += rnd_up(bytes, kAlign);
            return start;
        };
        m_offsets.rows = reserve((2 * dim + 2 * m_params.rotary_dim) * sizeof(float));
        m_offsets.query_tile = reserve(kQuerySlicesPerItem * m_params.index_key_heads * query_rows * dim * element);
        m_offsets.packed_blocks = reserve(m_packed_blocks_bytes);
        m_offsets.packed_query = reserve(m_params.index_key_heads * m_packed_query_bytes);
        m_offsets.gemm_scores = reserve(query_rows * kScoreBlockRows * sizeof(float));
        m_offsets.workspace = reserve(BrgemmKernel::get_wsp_size());
        m_offsets.scratch_a = reserve(std::max(m_q_k_kernel->get_scratch_a_size(), m_k_q_kernel->get_scratch_a_size()));
        m_offsets.key_chunk_stage = reserve(kScoreBlockRows * dim * element);
        m_select_offset = offset;
    }

    Params m_params;
    std::shared_ptr<BrgemmKernel> m_q_k_kernel;
    std::shared_ptr<BrgemmKernel> m_k_q_kernel;
    size_t m_heads_per_key;
    size_t m_tokens_per_call;
    size_t m_tile_tokens;
    size_t m_packed_blocks_bytes;  // the B matrix of 32 blocks (several tokens)
    size_t m_packed_query_bytes;   // the B matrix of the query tile of a K head (one token)
    ThreadOffsets m_offsets{};
    size_t m_select_offset = 0;  // the selection scratch follows the fixed part of a thread slice
};

}  // namespace ov::Extensions::Cpu::qsa
