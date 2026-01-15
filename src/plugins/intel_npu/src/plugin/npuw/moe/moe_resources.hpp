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
 * Manages resources that are allocated once and reused across multiple
 * inference calls. Resources are organized into:
 * - Per-sublayer resources: Request cache (each function call has independent cache)
 * - Shared resources: Sorted chunk sizes and output buffer (shared by all function calls)
 *
 * Design rationale:
 * - request_caches is per-sublayer because each function call may have different closures
 * - sorted_chunk_sizes is shared because all function calls use the same compiled models
 * - expert_output_accumulator is shared because JustInferRequest executes synchronously (no concurrent access)
 */
struct MoEResources {
    // Single request cache managing all sublayers (decoding mode optimization)
    // Created on first initialize_cache() call with total number of sublayers
    std::unique_ptr<ov::npuw::moe::RequestCache> request_cache;

    // Pre-sorted chunk sizes in descending order for greedy selection
    // Example: [256, 128, 64, 32, 16]
    // Used in prefill mode to efficiently select chunk sizes
    // SHARED by all function calls
    std::vector<size_t> sorted_chunk_sizes;

    // Prefill mode infer requests for different chunk sizes
    // Map: chunk_size -> infer_request
    // Created once during initialization, reused across all inferences
    // SHARED by all function calls
    std::map<size_t, ov::SoPtr<ov::IAsyncInferRequest>> chunk_infer_requests;

    // Output accumulation buffer (prefill mode)
    // Shape: [num_active_experts, 1, input_token_count, expert_hidden_dim]
    // Accumulates expert outputs before final reduction
    // SHARED by all function calls
    TensorPtr expert_output_accumulator;

    /**
     * @brief Initialize shared MoE resources (called once for function body)
     *
     * Initializes resources shared by all function calls:
     * - sorted_chunk_sizes: extracted from config.compiled_models
     * - chunk_infer_requests: created for each chunk size model
     * - expert_output_accumulator: allocated with given shape and device
     *
     * @param config MoE configuration
     * @param allocator Memory allocation function
     * @param device Target device for buffer allocation
     */
    void initialize_shared(
        const MoEConfig& config,
        std::function<TensorPtr(const ov::element::Type&, const ov::Shape&, const std::string&)> allocator,
        const std::string& device);

    /**
     * @brief Initialize request cache structure
     *
     * Creates RequestCache on first call (for all sublayers).
     * Subsequent calls are no-ops. Actual layer initialization is done
     * via request_cache->initialize_layer() after creating requests.
     *
     * @param num_sublayers Total number of sublayers
     * @param pool_size Request cache pool size (0 = disabled)
     */
    void initialize_cache(size_t num_sublayers, size_t pool_size);

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
