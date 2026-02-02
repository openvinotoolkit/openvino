// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace npuw {

namespace moe {

/**
 * @brief MoE inference processing mode
 *
 * Determines how experts are processed based on token count and expert activation pattern.
 *
 * The mode is determined by input token count:
 * - input_token_count == 1 → EXPERT_BATCH (single token, K experts in parallel)
 * - input_token_count > 1  → EXPERT_ITERATIVE (multiple tokens, iterate through experts)
 */
enum class MoEProcessingMode {
    /**
     * @brief Expert iterative mode
     *
     * Characteristics:
     * - Expert dimension: Iterative (process one expert at a time)
     * - Token dimension: Batch/Chunked (each expert processes N tokens in chunks)
     * - Pattern: for each expert { process N tokens in chunks }
     *
     * Use cases:
     * - Traditional prompt processing (multiple tokens from user input)
     * - Speculative decoding (processing multiple candidate tokens)
     * - Any scenario with multiple tokens requiring expert processing
     */
    EXPERT_ITERATIVE,

    /**
     * @brief Expert batch mode
     *
     * Characteristics:
     * - Expert dimension: Batch (process K experts simultaneously in parallel)
     * - Token dimension: Single (1 token per iteration)
     * - Pattern: process K experts in parallel for 1 token
     *
     * Use cases:
     * - Traditional token generation (token-by-token generation in autoregressive mode)
     * - Single-token inference scenarios
     */
    EXPERT_BATCH
};

/**
 * @brief Determine processing mode based on token count
 *
 * @param input_token_count Number of input tokens
 * @return EXPERT_BATCH for single token, EXPERT_ITERATIVE for multiple tokens
 */
inline MoEProcessingMode determine_processing_mode(size_t input_token_count) {
    return (input_token_count == 1) ? MoEProcessingMode::EXPERT_BATCH : MoEProcessingMode::EXPERT_ITERATIVE;
}

/**
 * @brief Get human-readable mode name for logging
 *
 * @param mode Processing mode
 * @return String representation of the mode
 */
inline const char* get_mode_name(MoEProcessingMode mode) {
    switch (mode) {
    case MoEProcessingMode::EXPERT_ITERATIVE:
        return "EXPERT_ITERATIVE";
    case MoEProcessingMode::EXPERT_BATCH:
        return "EXPERT_BATCH";
    default:
        return "UNKNOWN";
    }
}

}  // namespace moe

/**
 * @brief MoE I/O tensors structure
 *
 * Contains input and output tensors for MoE expert inference.
 * Shared between IBaseInferRequest and MoEExecutor.
 */
struct MoEIO {
    std::vector<ov::SoPtr<ov::ITensor>> outputs;  // # of elements - # of subgraph outputs
    ov::SoPtr<ov::ITensor> router_scores;         // Expert model input: router output for expert selection
    ov::SoPtr<ov::ITensor> expert_input;          // Expert model input: token embeddings
};

}  // namespace npuw
}  // namespace ov
