// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_resources.hpp"

#include <algorithm>

#include "../logging.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace npuw {
namespace moe {

void MoEResources::initialize(
    const MoEConfig& config,
    std::function<TensorPtr(const ov::element::Type&, const ov::Shape&, const std::string&)> allocator,
    size_t pool_size,
    const std::string& device) {
    LOG_DEBUG("Initializing MoE resources...");
    LOG_BLOCK();

    // Initialize request cache if pool_size > 0
    if (pool_size > 0) {
        LOG_DEBUG("Creating request cache with pool_size=" << pool_size);
        // Note: Cache will be populated later when requests are created
        // RequestCache constructor signature: RequestCache(size_t num_layers, size_t pool_size)
        // We don't create it here as it needs to be populated per submodel index
    } else {
        LOG_DEBUG("Request cache disabled (pool_size=0)");
    }

    // Pre-sort chunk sizes in descending order for greedy selection
    sorted_chunk_sizes.clear();
    sorted_chunk_sizes.reserve(config.compiled_models.size());
    for (const auto& entry : config.compiled_models) {
        sorted_chunk_sizes.push_back(entry.first);
    }
    std::sort(sorted_chunk_sizes.begin(), sorted_chunk_sizes.end(), std::greater<size_t>());

    LOG_DEBUG("Sorted chunk sizes: " << sorted_chunk_sizes.size() << " entries");

    // Allocate output buffer for prefill mode
    const size_t active_experts = config.num_active_experts;
    const size_t num_tokens = config.input_token_count;
    const size_t embed_dim = config.expert_hidden_dim;

    // Buffer shape: [num_active_experts, 1, num_tokens, embed_dim]
    ov::Shape buffer_shape = {active_experts, 1, num_tokens, embed_dim};

    // Infer element type from first compiled model output
    auto first_model = config.compiled_models.begin()->second;
    auto output_element_type = first_model->outputs()[0].get_element_type();

    output_buffer = allocator(output_element_type, buffer_shape, device);

    LOG_DEBUG("Allocated output buffer: shape=" << buffer_shape << ", type=" << output_element_type
                                                << ", device=" << device);

    LOG_DEBUG("MoE resources initialization completed");
}

void MoEResources::reset() {
    LOG_DEBUG("Resetting MoE resources...");

    // Clear request cache if exists
    if (request_cache) {
        request_cache.reset();
    }

    // Keep sorted_chunk_sizes (static data)
    // Keep output_buffer (reusable allocation)

    LOG_DEBUG("MoE resources reset completed");
}

}  // namespace moe
}  // namespace npuw
}  // namespace ov
