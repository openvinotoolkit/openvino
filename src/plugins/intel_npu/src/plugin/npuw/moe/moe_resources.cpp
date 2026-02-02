// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_resources.hpp"

#include <algorithm>

#include "../logging.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace npuw {
namespace moe {

void MoEResources::initialize_expert_iterative_mode(
    const MoEConfig& config,
    std::function<TensorPtr(const ov::element::Type&, const ov::Shape&, const std::string&)> allocator,
    const std::string& device) {
    LOG_DEBUG("Initializing MoE resources for EXPERT_ITERATIVE mode...");
    LOG_BLOCK();

    if (config.input_token_count <= 1) {
        OPENVINO_THROW("initialize_expert_iterative_mode called with input_token_count=",
                       config.input_token_count,
                       ", expected > 1. Use initialize_expert_batch_mode instead.");
    }

    // Step 1: Create chunk infer requests and collect chunk sizes
    LOG_DEBUG("Creating chunk infer requests for expert iterative mode...");
    sorted_chunk_sizes.clear();
    chunk_infer_requests.clear();
    sorted_chunk_sizes.reserve(config.compiled_models.size());

    for (const auto& entry : config.compiled_models) {
        const size_t chunk_size = entry.first;
        const auto& compiled_model = entry.second;

        if (!compiled_model) {
            LOG_WARN("Compiled model for chunk_size=" << chunk_size << " is null, skipping...");
            continue;
        }

        try {
            auto infer_request = compiled_model->create_infer_request();
            chunk_infer_requests[chunk_size] = std::move(infer_request);
            sorted_chunk_sizes.push_back(chunk_size);
            LOG_DEBUG("  Created chunk infer request for chunk_size=" << chunk_size);
        } catch (const std::exception& ex) {
            OPENVINO_THROW("MoE chunk infer request creation failed for chunk_size=", chunk_size, ": ", ex.what());
        }
    }

    // Step 2: Sort chunk sizes in descending order for greedy selection
    std::sort(sorted_chunk_sizes.begin(), sorted_chunk_sizes.end(), std::greater<size_t>());
    LOG_DEBUG("Created " << chunk_infer_requests.size() << " chunk infer requests with sorted sizes");

    // Step 3: Allocate output accumulator buffer
    const size_t active_experts = config.num_active_experts;
    const size_t num_tokens = config.input_token_count;
    const size_t embed_dim = config.expert_hidden_dim;

    ov::Shape buffer_shape = {active_experts, 1, num_tokens, embed_dim};

    // Infer element type from first compiled model output
    auto first_model = config.compiled_models.begin()->second;
    auto output_element_type = first_model->outputs()[0].get_element_type();

    expert_output_accumulator = allocator(output_element_type, buffer_shape, device);

    LOG_DEBUG("Allocated iterative mode output buffer: shape=" << buffer_shape << ", type=" << output_element_type
                                                               << ", device=" << device);

    LOG_DEBUG("Expert iterative mode initialization completed");
}

void MoEResources::initialize_expert_batch_mode(
    const MoEConfig& config,
    std::function<TensorPtr(const ov::element::Type&, const ov::Shape&, const std::string&)> allocator,
    const std::string& device,
    size_t num_sublayers,
    size_t pool_size) {
    LOG_DEBUG("Initializing MoE resources for EXPERT_BATCH mode...");
    LOG_BLOCK();

    if (config.input_token_count != 1) {
        OPENVINO_THROW("initialize_expert_batch_mode called with input_token_count=",
                       config.input_token_count,
                       ", expected = 1. Use initialize_expert_iterative_mode instead.");
    }

    // Expert batch mode resources:
    // - No chunk_infer_requests (uses batch-unrolled model via cache/subrequest)
    // - No expert_output_accumulator (expert outputs go directly to downstream via normal tensor connections)
    // - Request cache (optional, if pool_size > 0)

    if (pool_size > 0) {
        // Create the RequestCache once for all sublayers
        if (!request_cache) {
            LOG_DEBUG("Creating RequestCache for " << num_sublayers << " sublayers with pool_size=" << pool_size);
            request_cache = std::make_unique<ov::npuw::moe::RequestCache>(num_sublayers, pool_size);
            LOG_DEBUG("RequestCache created successfully");
        }
        // Note: Actual request pool creation (initialize_layer) is done by caller
        // after creating the inference requests for each sublayer
    } else {
        LOG_DEBUG("Request cache disabled (pool_size=0)");
    }

    LOG_DEBUG("Expert batch mode initialization completed");
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
