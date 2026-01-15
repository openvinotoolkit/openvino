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

void MoEResources::initialize_shared(
    const MoEConfig& config,
    std::function<TensorPtr(const ov::element::Type&, const ov::Shape&, const std::string&)> allocator,
    const std::string& device) {
    LOG_DEBUG("Initializing shared MoE resources...");
    LOG_BLOCK();

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

    expert_output_accumulator = allocator(output_element_type, buffer_shape, device);

    LOG_DEBUG("Allocated shared output buffer: shape=" << buffer_shape << ", type=" << output_element_type
                                                       << ", device=" << device);

    LOG_DEBUG("Shared MoE resources initialization completed");
}

void MoEResources::initialize_cache(size_t num_sublayers, size_t pool_size) {
    if (pool_size == 0) {
        LOG_DEBUG("Request cache disabled (pool_size=0)");
        return;
    }

    // Create the RequestCache once for all sublayers
    if (!request_cache) {
        LOG_DEBUG("Creating RequestCache for " << num_sublayers << " sublayers with pool_size=" << pool_size);
        request_cache = std::make_unique<ov::npuw::moe::RequestCache>(num_sublayers, pool_size);
        LOG_DEBUG("RequestCache created successfully");
    }

    // Note: Actual request pool creation (initialize_layer) is done by caller
    // after creating the inference requests for each sublayer
}

void MoEResources::reset_cache() {
    LOG_DEBUG("Resetting request cache...");
    request_cache.reset();

    // Keep sorted_chunk_sizes (static data)
    // Keep expert_output_accumulator (reusable allocation)

    LOG_DEBUG("Request cache reset completed");
}

}  // namespace moe
}  // namespace npuw
}  // namespace ov
