// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// The interface between the QSA nodes and their cross-compiled kernels: the call tensors, the executors and the scorer.
// Nothing here is instruction-set specific.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "cpu_parallel.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/type/element_type.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::intel_cpu {
class BrgemmKernel;
}  // namespace ov::intel_cpu

namespace ov::Extensions::Cpu::qsa {

/// @brief What is fixed per node, for both ops.
struct Config {
    size_t key_block_size = 0;       ///< tokens that make a block
    size_t num_selection_heads = 0;  ///< selection heads of the output
    size_t block_topk = 0;           ///< blocks kept per token and selection head
    size_t head_dim = 0;             ///< channels of an index head
    size_t index_query_heads = 0;    ///< heads of the index query
    size_t index_key_heads = 0;      ///< heads of the index keys
    size_t rotary_dim = 0;           ///< channels rotated by RoPE
    size_t num_threads = 0;          ///< worker threads of the node (sizes the scratch)
    float eps = 0.0F;                ///< RMSNorm epsilon
    float scale = 0.0F;              ///< multiplier of the scores
    ov::element::Type precision;     ///< of the floating-point ports (f32, f16 or bf16)
    bool query_norm = true;          ///< the op applies the RMSNorm to the query (else it is done in front of the op)
    bool query_rope = true;          ///< the op applies the RoPE to the query (else it is done in front of the op)
};

// The cache key of an executor and the cache of the brgemm kernels are in qsa_executor_cache.hpp: the executor asks
// for its kernels there.

/// @brief The blocks one brgemm call scores: the M of the k.q kernel, the N of the q.k kernel.
inline constexpr size_t kScoreBlockRows = 32;
/// @brief Rows of the query tile of a call: (tokens x heads of a K head) up to this many.
inline constexpr size_t kScoreMaxCols = 128;

/// @brief Tokens one brgemm call scores together for K heads of `heads_per_key` query heads.
constexpr size_t score_tokens_per_call(size_t heads_per_key) {
    return std::max<size_t>(1, kScoreMaxCols / heads_per_key);
}

/// @brief The brgemm kernels of the block scorer, got by the executor from the params cache; empty where this CPU or
/// the configuration has no brgemm scoring (the stages of the executor score then).
struct ScoreKernels {
    std::shared_ptr<ov::intel_cpu::BrgemmKernel> q_k;  ///< several tokens: C[tokens x heads, 32] = Q . K^T
    std::shared_ptr<ov::intel_cpu::BrgemmKernel> k_q;  ///< one token (decode):  C[32, heads] = K . Q^T
};

/// @brief The tensors of a call of the QSAIndexer node (dense caches, see qsa_indexer_kernel.cpp).
struct IndexerArgs {
    // Set by IndexerExecutor::find_batch_max_params() from the longest row of the batch; read by the stages and the
    // scorer.
    int64_t max_blocks = 0;                         ///< most complete blocks a row of the batch has after the call
    int64_t max_new_blocks = 0;                     ///< most blocks a row of the batch completes in the call
    int64_t max_q_len = 0;                          ///< most tokens of a row of the batch in the call
    ov::intel_cpu::PlainTensor q_idx;               ///< [B, L, H_iq, D]
    ov::intel_cpu::PlainTensor k_idx;               ///< [B, L, H_ik, D]
    ov::intel_cpu::PlainTensor rope_cos, rope_sin;  ///< [1 | B, L, Dr]
    ov::intel_cpu::PlainTensor key_length;          ///< [1 | B]
    ov::intel_cpu::PlainTensor q_norm_weight, k_norm_weight;                   ///< [D]
    ov::intel_cpu::PlainTensor past_raw_k_in, past_rope_in, summary_cache_in;  ///< the state inputs (read only)
    ov::intel_cpu::PlainTensor summary_cache, past_raw_k, past_rope;           ///< the updated state
    ov::intel_cpu::PlainTensor sel_indices, sel_count;                         ///< [B, H_sel, L, K], [B, H_sel, L]
};

/// @brief The tensors of a call of the PagedQSAIndexer node (caches addressed through the PagedAttention page table).
struct PagedIndexerArgs {
    // Set by IndexerExecutor::find_batch_max_params() from the longest sequence; read by the stages and the scorer.
    int64_t max_blocks = 0;                                    ///< most complete blocks a sequence has after the call
    int64_t max_new_blocks = 0;                                ///< most blocks a sequence completes in the call
    int64_t max_q_len = 0;                                     ///< most tokens of a sequence in the call
    ov::intel_cpu::PlainTensor q_idx;                          ///< [tokens, H_iq * D]
    ov::intel_cpu::PlainTensor k_idx;                          ///< [tokens, H_ik * D]
    ov::intel_cpu::PlainTensor rope_cos, rope_sin;             ///< [tokens, Dr]
    ov::intel_cpu::PlainTensor q_norm_weight, k_norm_weight;   ///< [D]
    ov::intel_cpu::PlainTensor past_lens, subsequence_begins;  ///< [seqs], [seqs + 1]
    ov::intel_cpu::PlainTensor block_indices, block_indices_begins;             ///< page table
    ov::intel_cpu::PlainTensor raw_k_cache, indexer_rope_cache, summary_cache;  ///< paged caches, written in place
    ov::intel_cpu::PlainTensor sel_indices, sel_count;                          ///< [tokens, H_sel, K], [tokens, H_sel]
};

/// @brief The executor of a QSA node: prepare the caches (summaries, state), then score the tokens with the stages or
/// the brgemm scorer. It is stateless and shared through the params cache.
template <typename Call>
class IndexerExecutor {
public:
    virtual ~IndexerExecutor() = default;

    /// The implementation type the node reports.
    [[nodiscard]] virtual ov::intel_cpu::impl_desc_type impl_type() const = 0;
    /// Checks that the call fits its caches and sets its max_* fields. Call it before scratch_bytes() and execute().
    virtual void find_batch_max_params(Call& call) const = 0;
    /// The scratch bytes of the call, shared by all the stages.
    [[nodiscard]] virtual size_t scratch_bytes(const Call& call) const = 0;
    virtual void execute(const Call& call,
                         uint8_t* scratch,
                         const ov::intel_cpu::CpuParallelPtr& cpu_parallel) const = 0;
};

using QSAIndexerExecutor = IndexerExecutor<IndexerArgs>;
using PagedQSAIndexerExecutor = IndexerExecutor<PagedIndexerArgs>;

}  // namespace ov::Extensions::Cpu::qsa
