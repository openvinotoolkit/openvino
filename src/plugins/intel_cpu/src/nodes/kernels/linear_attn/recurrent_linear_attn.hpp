// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

#include "cpu_parallel.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

/// \brief Computes recurrent linear attention for a contiguous token batch.
///
/// The function consumes query/key/value inputs and recurrent state for the same batch,
/// optionally applies fused Q/K L2 normalization, and writes both attention output and
/// updated recurrent state.
///
/// \note `output_recurrent_state` is updated by this kernel and must have a preallocated
/// shape compatible with `recurrent_state`.
/// \param query Query tensor with shape [tokens, qk_heads, qk_head_size].
/// \param key Key tensor with shape [tokens, qk_heads, qk_head_size].
/// \param value Value tensor with shape [tokens, v_heads, v_head_size].
/// \param recurrent_state Input recurrent state tensor for current step.
/// \param gate Gate tensor with shape [tokens, v_heads].
/// \param beta Beta tensor with shape [tokens, v_heads].
/// \param q_l2_norm_eps Epsilon used for query L2 normalization.
/// \param k_l2_norm_eps Epsilon used for key L2 normalization.
/// \param use_qk_l2norm Enables Q/K L2 normalization path.
/// \param output_attn Output attention tensor [tokens, v_heads, v_head_size].
/// \param output_recurrent_state Output tensor with updated recurrent state.
/// \param temp_buffer buffer contains `num_threads * 3 * qk_head_size * sizeof(float)` bytes, 64-byte aligned.
/// \param cpu_parallel CPU parallel runtime used for threading/work split.
void recurrent_linear_attn(const ov::intel_cpu::PlainTensor& query,
                           const ov::intel_cpu::PlainTensor& key,
                           const ov::intel_cpu::PlainTensor& value,
                           const ov::intel_cpu::PlainTensor& recurrent_state,
                           const ov::intel_cpu::PlainTensor& gate,
                           const ov::intel_cpu::PlainTensor& beta,
                           float q_l2_norm_eps,
                           float k_l2_norm_eps,
                           bool use_qk_l2norm,
                           ov::intel_cpu::PlainTensor& output_attn,
                           ov::intel_cpu::PlainTensor& output_recurrent_state,
                           float* temp_buffer,
                           const ov::intel_cpu::CpuParallelPtr& cpu_parallel);

/// \brief Computes recurrent linear attention using paged state storage.
///
/// The function processes multiple variable-length subsequences packed into one token batch.
/// `subsequence_begins`, `block_indices`, `block_indices_begins`, `past_lens`, and
/// `cache_interval` describe how each subsequence maps to blocks in `recurrent_state_table`.
///
/// \note `recurrent_state_table` is updated in place according to paging metadata.
/// \param query Query tensor with shape [batch_in_tokens, qk_heads, qk_head_size].
/// \param key Key tensor with shape [batch_in_tokens, qk_heads, qk_head_size].
/// \param value Value tensor with shape [batch_in_tokens, v_heads, v_head_size].
/// \param recurrent_state_table Paged recurrent-state table updated in place.
/// \param gate Gate tensor with shape [batch_in_tokens, v_heads].
/// \param beta Beta tensor with shape [batch_in_tokens, v_heads].
/// \param subsequence_begins Prefix offsets for token ranges per sequence.
/// \param block_indices Flattened block ids used by all sequences.
/// \param block_indices_begins Prefix offsets into `block_indices` per sequence.
/// \param past_lens Number of already-processed tokens per sequence.
/// \param cache_interval Cache update interval per sequence.
/// \param q_l2_norm_eps Epsilon used for query L2 normalization.
/// \param k_l2_norm_eps Epsilon used for key L2 normalization.
/// \param use_qk_l2norm Enables Q/K L2 normalization path.
/// \param output_attn Output attention tensor [batch_in_tokens, v_heads, v_head_size].
/// \param temp_buffer buffer contains `num_threads * 3 * qk_head_size * sizeof(float)` bytes, 64-byte aligned.
/// \param cpu_parallel CPU parallel runtime used for threading/work split.
void recurrent_linear_attn_paged(const ov::intel_cpu::PlainTensor& query,
                                 const ov::intel_cpu::PlainTensor& key,
                                 const ov::intel_cpu::PlainTensor& value,
                                 ov::intel_cpu::PlainTensor& recurrent_state_table,
                                 const ov::intel_cpu::PlainTensor& gate,
                                 const ov::intel_cpu::PlainTensor& beta,
                                 const ov::intel_cpu::PlainTensor& subsequence_begins,
                                 const ov::intel_cpu::PlainTensor& block_indices,
                                 const ov::intel_cpu::PlainTensor& block_indices_begins,
                                 const ov::intel_cpu::PlainTensor& past_lens,
                                 const ov::intel_cpu::PlainTensor& cache_interval,
                                 float q_l2_norm_eps,
                                 float k_l2_norm_eps,
                                 bool use_qk_l2norm,
                                 ov::intel_cpu::PlainTensor& output_attn,
                                 float* temp_buffer,
                                 const ov::intel_cpu::CpuParallelPtr& cpu_parallel);

}  // namespace ov::Extensions::Cpu::XARCH
