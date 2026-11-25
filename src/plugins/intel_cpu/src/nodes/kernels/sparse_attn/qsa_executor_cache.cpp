// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/sparse_attn/qsa_executor_cache.hpp"

#include <common/utils.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <memory>
#include <tuple>

#include "cache/multi_cache.h"
#include "nodes/kernels/sparse_attn/qsa_executor.hpp"
#include "nodes/kernels/x64/brgemm_kernel.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::Extensions::Cpu::qsa {

using ov::intel_cpu::BrgemmKernel;
using ov::intel_cpu::MultiCachePtr;

size_t QsaExecutorKey::hash() const {
    const auto& c = config;
    size_t seed = 0;
    for (const size_t value : {c.key_block_size,
                               c.num_selection_heads,
                               c.block_topk,
                               c.head_dim,
                               c.index_query_heads,
                               c.index_key_heads,
                               c.rotary_dim,
                               c.num_threads}) {
        seed = dnnl::impl::hash_combine(seed, value);
    }
    seed = dnnl::impl::hash_combine(seed, c.eps);
    seed = dnnl::impl::hash_combine(seed, c.scale);
    seed = dnnl::impl::hash_combine(seed, c.precision.hash());
    seed = dnnl::impl::hash_combine(seed, c.query_norm);
    return dnnl::impl::hash_combine(seed, c.query_rope);
}

bool QsaExecutorKey::operator==(const QsaExecutorKey& rhs) const {
    const auto& l = config;
    const auto& r = rhs.config;
    return std::tie(l.key_block_size,
                    l.num_selection_heads,
                    l.block_topk,
                    l.head_dim,
                    l.index_query_heads,
                    l.index_key_heads,
                    l.rotary_dim,
                    l.num_threads,
                    l.eps,
                    l.scale,
                    l.precision,
                    l.query_norm,
                    l.query_rope) == std::tie(r.key_block_size,
                                              r.num_selection_heads,
                                              r.block_topk,
                                              r.head_dim,
                                              r.index_query_heads,
                                              r.index_key_heads,
                                              r.rotary_dim,
                                              r.num_threads,
                                              r.eps,
                                              r.scale,
                                              r.precision,
                                              r.query_norm,
                                              r.query_rope);
}

namespace {

// Whether this CPU runs the scorer for `precision` and head dim `dim`. Only the AVX-512 variant of an executor asks
// (the other variants have no scorer).
bool scoreKernelSupported(const ov::element::Type& precision, size_t dim) {
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

// The key of a score kernel in the params cache.
struct ScoreKernelKey {
    size_t query_rows = 0;  // rows of the query tile: tokens x heads of a K head
    size_t dim = 0;         // head dim, the K of the product
    ov::element::Type precision;
    // C = K.Q^T, the blocks are the A matrix (one token); else C = Q.K^T, the query tile is the A matrix
    bool is_k_q_gemm = false;

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

// The kernel of the q . k of 32 blocks and `query_rows` query rows, with the ReLU, from the cache; nullptr if it can
// not be built. Two forms, by how many tokens a call scores:
//   is_k_q_gemm == false (q.k, several tokens)  C[query_rows, 32] = max(0, query_tile[query_rows, D] x blocks[32, D]^T)
//       A: the query tile, B: the transposed packed blocks. A row of C is 32 scores of one token.
//   is_k_q_gemm == true (k.q, one token)        C[32, query_rows] = max(0, blocks[32, D] x query_tile[query_rows, D]^T)
//       A: the blocks, read in place; B: the transposed packed query tile. A decode step does not make an extra pass
//       over the cache (packing it would double its memory traffic, the cache is what a decode step is bound by).
std::shared_ptr<BrgemmKernel> getScoreKernel(const MultiCachePtr& cache,
                                             const ov::element::Type& precision,
                                             size_t query_rows,
                                             size_t dim,
                                             bool is_k_q_gemm) {
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

}  // namespace

ScoreKernels getScoreKernels(const MultiCachePtr& cache, const Config& config) {
    // The selection heads have to be covered by the K heads.
    if (config.index_key_heads == 0 || config.index_query_heads % config.index_key_heads != 0 ||
        config.num_selection_heads > config.index_key_heads ||
        !scoreKernelSupported(config.precision, config.head_dim)) {
        return {};
    }
    const size_t heads_per_key = config.index_query_heads / config.index_key_heads;
    // `q_k` scores a full set of tokens (query rows: tokens_per_call * heads_per_key), `k_q` a single token
    // (heads_per_key rows): the decode step is one token wide.
    ScoreKernels kernels;
    kernels.q_k = getScoreKernel(cache,
                                 config.precision,
                                 score_tokens_per_call(heads_per_key) * heads_per_key,
                                 config.head_dim,
                                 false);
    kernels.k_q = getScoreKernel(cache, config.precision, heads_per_key, config.head_dim, true);
    if (!kernels.q_k || !kernels.k_q) {
        return {};
    }
    return kernels;
}

}  // namespace ov::Extensions::Cpu::qsa
