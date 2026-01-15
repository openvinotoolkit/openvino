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
 * inference calls:
 * - Request cache: Pre-allocated infer requests for expert combinations
 * - Sorted chunk sizes: Pre-sorted list for greedy chunk selection
 * - Output buffer: Accumulated expert outputs (prefill mode)
 */
struct MoEResources {
    // Request cache for expert combinations (decoding mode optimization)
    // Caches infer requests configured for specific expert combinations
    // to avoid repeated closure unpacking
    std::unique_ptr<ov::npuw::moe::RequestCache> request_cache;

    // Pre-sorted chunk sizes in descending order for greedy selection
    // Example: [256, 128, 64, 32, 16]
    // Used in prefill mode to efficiently select chunk sizes
    std::vector<size_t> sorted_chunk_sizes;

    // Output accumulation buffer (prefill mode)
    // Shape: [num_active_experts, 1, input_token_count, expert_hidden_dim]
    // Accumulates expert outputs before final reduction
    TensorPtr output_buffer;

    /**
     * @brief Initialize MoE resources
     *
     * @param config MoE configuration
     * @param allocator Memory allocation function
     * @param pool_size Request cache pool size (0 = disabled)
     * @param device Target device for buffer allocation
     */
    void initialize(const MoEConfig& config,
                    std::function<TensorPtr(const ov::element::Type&, const ov::Shape&, const std::string&)> allocator,
                    size_t pool_size,
                    const std::string& device);

    /**
     * @brief Reset resources state (called during subrequest recreation)
     */
    void reset();

    /**
     * @brief Check if resources are initialized
     */
    bool is_initialized() const {
        return output_buffer != nullptr;
    }
};

}  // namespace moe
}  // namespace npuw
}  // namespace ov
