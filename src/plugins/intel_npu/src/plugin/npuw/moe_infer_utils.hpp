// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <vector>

#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace npuw {
namespace moe {

/**
 * @brief Slice a single expert's weight from batched weight tensor.
 *
 * Creates a zero-copy view of expert weight from batched tensor with shape [num_experts, ...].
 * Handles both sub-byte (4-bit) and regular (>=8-bit) element types.
 *
 * @param batched_weight Source tensor with shape [num_experts, ...]
 * @param expert_id ID of the expert to slice (0 to num_experts-1)
 * @param num_experts Total number of experts in the batch
 * @return ov::Tensor Zero-copy view with shape [1, ...] for the selected expert
 */
ov::Tensor slice_expert_weight(const ov::Tensor& batched_weight, size_t expert_id, size_t num_experts);

/**
 * @brief Parse selected experts from router output tensor.
 *
 * Analyzes router output to determine which experts are selected by each token.
 * Router output has shape [num_experts, 1, token_num, 1] with non-zero weights
 * indicating expert selection.
 *
 * @param router_output Router output tensor with expert selection weights
 * @param num_experts Total number of experts
 * @param token_to_experts[out] Map from token_id to list of selected expert_ids
 * @param expert_to_tokens[out] Map from expert_id to list of tokens that selected it
 * @return Vector of selected expert IDs (sorted)
 */
std::vector<size_t> parse_selected_experts_from_router(const ov::SoPtr<ov::ITensor>& router_output,
                                                       size_t num_experts,
                                                       std::map<size_t, std::vector<size_t>>& token_to_experts,
                                                       std::map<size_t, std::vector<size_t>>& expert_to_tokens);

/**
 * @brief Optimized tensor setter for small/large tensors.
 *
 * Uses copy for small tensors (<= 11520 bytes) to avoid set_tensor overhead.
 * Uses set_tensor for large tensors (zero-copy).
 *
 * @param request Inference request
 * @param iport Input port to set tensor for
 * @param tensor_impl Tensor to set or copy
 */
void set_tensor_optimized(ov::SoPtr<ov::IAsyncInferRequest> request,
                          const ov::Output<const ov::Node>& iport,
                          const ov::SoPtr<ov::ITensor>& tensor_impl);

/**
 * @brief Gather router scores for selected tokens from router output.
 *
 * Extracts router scores for a specific expert and token range from batched router output.
 * Supports both 4D [num_experts, 1, token_num, 1] and 2D [num_experts, token_num] shapes.
 *
 * @param router_source Source router tensor with all expert scores
 * @param router_dest Destination tensor to write gathered scores
 * @param expert_id ID of the expert to gather scores for
 * @param token_ids List of token IDs to process
 * @param chunk_start Starting index in token_ids list
 * @param chunk_size Number of tokens to process
 */
void gather_router_scores(const ov::SoPtr<ov::ITensor>& router_source,
                          const ov::SoPtr<ov::ITensor>& router_dest,
                          size_t expert_id,
                          const std::vector<size_t>& token_ids,
                          size_t chunk_start,
                          size_t chunk_size);

/**
 * @brief Gather expert input embeddings for selected tokens.
 *
 * Extracts input embeddings for a range of tokens from the source tensor.
 * Supports both 2D [num_tokens, hidden_dim] and 4D [1, 1, num_tokens, hidden_dim] shapes.
 *
 * @param input_source Source tensor with all token embeddings
 * @param input_dest Destination tensor to write gathered embeddings
 * @param token_ids List of token IDs to process
 * @param chunk_start Starting index in token_ids list
 * @param chunk_size Number of tokens to process
 */
void gather_expert_inputs(const ov::SoPtr<ov::ITensor>& input_source,
                          const ov::SoPtr<ov::ITensor>& input_dest,
                          const std::vector<size_t>& token_ids,
                          size_t chunk_start,
                          size_t chunk_size);

/**
 * @brief Scatter expert outputs back to global output buffer.
 *
 * Writes expert output for processed tokens to the correct positions in the global
 * output buffer based on expert slot assignments.
 *
 * @param expert_output Expert's output tensor for current chunk
 * @param global_output_buffer Global output buffer [K, 1, num_tokens, embed_dim]
 * @param token_ids List of token IDs that were processed
 * @param chunk_start Starting index in token_ids list
 * @param chunk_size Number of tokens processed
 * @param embed_dim Embedding dimension size
 * @param input_token_count Total number of input tokens
 * @param expert_slots_for_tokens Precomputed expert slot for each token in token_ids (same size as token_ids)
 */
void scatter_expert_outputs(const ov::SoPtr<ov::ITensor>& expert_output,
                            const ov::SoPtr<ov::ITensor>& global_output_buffer,
                            const std::vector<size_t>& token_ids,
                            size_t chunk_start,
                            size_t chunk_size,
                            size_t embed_dim,
                            size_t input_token_count,
                            const std::vector<size_t>& expert_slots_for_tokens);

}  // namespace moe
}  // namespace npuw
}  // namespace ov
