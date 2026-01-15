// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "../moe_infer_utils.hpp"  // For RequestCache
#include "../util.hpp"             // For TensorPtr
#include "moe_config.hpp"

namespace ov {
namespace npuw {
namespace moe {

// Import TensorPtr from parent namespace
using ov::npuw::util::TensorPtr;

/**
 * @brief MoE runtime resources for reusable allocations across inferences
 *
 * Resources are mode-specific:
 * - Prefill mode (input_token_count > 1): Uses chunk_infer_requests + expert_output_accumulator
 * - Decoding mode (input_token_count = 1): Uses request_cache only
 *
 * Each JustInferRequest (prefill or decoding) has independent MoEResources.
 * Resources are initialized once and reused across multiple inference calls.
 */
struct MoEResources {
    // ============================================================================
    // DECODING MODE ONLY (nullptr/empty in prefill mode)
    // ============================================================================

    // Request cache for expert combination caching (decoding optimization)
    // Created only when pool_size > 0 and is_decoding() = true
    std::unique_ptr<ov::npuw::moe::RequestCache> request_cache;

    // ============================================================================
    // PREFILL MODE ONLY (nullptr/empty in decoding mode)
    // ============================================================================

    // Pre-sorted chunk sizes in descending order for greedy selection
    // Example: [256, 128, 64, 32, 16]
    // Used to select optimal chunk size for token batches
    std::vector<size_t> sorted_chunk_sizes;

    // Infer requests for different chunk sizes
    // Map: chunk_size -> infer_request
    // Reused across experts by unpacking different weights
    std::map<size_t, ov::SoPtr<ov::IAsyncInferRequest>> chunk_infer_requests;

    // Output accumulation buffer
    // Shape: [num_active_experts, 1, input_token_count, expert_hidden_dim]
    // Accumulates expert outputs before final reduction
    TensorPtr expert_output_accumulator;

    /**
     * @brief Initialize resources for prefill mode (multi-token inference)
     *
     * Initializes:
     * - sorted_chunk_sizes: extracted from config.compiled_models
     * - chunk_infer_requests: created for each chunk size model
     * - expert_output_accumulator: allocated with given shape and device
     *
     * @param config MoE configuration (must have input_token_count > 1)
     * @param allocator Memory allocation function
     * @param device Target device for buffer allocation
     */
    void initialize_for_prefill(
        const MoEConfig& config,
        std::function<TensorPtr(const ov::element::Type&, const ov::Shape&, const std::string&)> allocator,
        const std::string& device);

    /**
     * @brief Initialize resources for decoding mode (single-token inference)
     *
     * Creates request cache if pool_size > 0. Request cache enables expert
     * combination caching for improved decoding performance.
     *
     * @param config MoE configuration (must have input_token_count = 1)
     * @param allocator Memory allocation function
     * @param device Target device for buffer allocation
     * @param num_sublayers Total number of sublayers (for cache structure)
     * @param pool_size Request cache pool size (0 = cache disabled)
     */
    void initialize_for_decoding(
        const MoEConfig& config,
        std::function<TensorPtr(const ov::element::Type&, const ov::Shape&, const std::string&)> allocator,
        const std::string& device,
        size_t num_sublayers,
        size_t pool_size);

    /**
     * @brief Reset request cache (for recreation after failure)
     *
     * Clears the entire request cache. All sublayers will need re-initialization.
     */
    void reset_cache();

    /**
     * @brief Check if resources are initialized
     */
    bool is_initialized() const {
        return expert_output_accumulator != nullptr;
    }
};

}  // namespace moe
}  // namespace npuw
}  // namespace ov
