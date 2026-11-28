// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_resources.hpp"

#include <algorithm>
#include <cstring>

#include "../logging.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/memory_util.hpp"

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

    OPENVINO_ASSERT(config.is_valid(), "MoE: Invalid iterative configuration");
    const auto& first_entry = *config.compiled_models.begin();
    OPENVINO_ASSERT(first_entry.first > 0 && first_entry.second && !first_entry.second->outputs().empty(),
                    "MoE: Invalid first chunk model");
    const auto output_shape = first_entry.second->outputs()[0].get_shape();
    OPENVINO_ASSERT((output_shape.size() == 3 || output_shape.size() == 4) && output_shape[0] == 1 &&
                        output_shape.back() == config.expert_hidden_dim,
                    "MoE: Unsupported chunk output layout");
    size_t token_axis = 0;
    for (size_t axis = 1; axis + 1 < output_shape.size(); ++axis) {
        OPENVINO_ASSERT(output_shape[axis] == 1 || output_shape[axis] == first_entry.first,
                        "MoE: Unsupported chunk output layout");
        if (output_shape[axis] == first_entry.first) {
            OPENVINO_ASSERT(token_axis == 0, "MoE: Ambiguous chunk output layout");
            token_axis = axis;
        }
    }
    OPENVINO_ASSERT(token_axis != 0, "MoE: Chunk output has no token axis");
    auto buffer_shape = output_shape;
    buffer_shape[0] = config.num_active_experts;
    buffer_shape[token_axis] = config.input_token_count;
    const auto output_element_type = first_entry.second->outputs()[0].get_element_type();
    OPENVINO_ASSERT(output_element_type == ov::element::f32 || output_element_type == ov::element::f16,
                    "MoE: Unsupported chunk output element type");
    const auto buffer_bytes = ov::util::get_memory_size_safe(output_element_type, buffer_shape);
    OPENVINO_ASSERT(buffer_bytes.has_value(), "MoE: Accumulator size overflows");

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
            // Create 2 requests per chunk_size for double-buffering:
            // slot 0 and slot 1 alternate so Unpack+Gather can overlap with NPU inference.
            chunk_infer_requests[chunk_size][0] = compiled_model->create_infer_request();
            chunk_infer_requests[chunk_size][1] = compiled_model->create_infer_request();
            sorted_chunk_sizes.push_back(chunk_size);
            LOG_DEBUG("  Created 2 chunk infer requests (double-buffer) for chunk_size=" << chunk_size);
        } catch (const std::exception& ex) {
            OPENVINO_THROW("MoE chunk infer request creation failed for chunk_size=", chunk_size, ": ", ex.what());
        }
    }

    // Step 2: Sort chunk sizes in descending order for greedy selection
    std::sort(sorted_chunk_sizes.begin(), sorted_chunk_sizes.end(), std::greater<size_t>());
    LOG_DEBUG("Created " << chunk_infer_requests.size() << " chunk infer requests with sorted sizes");

    // Step 3: Allocate output accumulator buffer
    expert_output_accumulator = allocator(output_element_type, buffer_shape, device);
    OPENVINO_ASSERT(expert_output_accumulator && expert_output_accumulator->is_continuous() &&
                        expert_output_accumulator->get_element_type() == output_element_type &&
                        expert_output_accumulator->get_shape() == buffer_shape &&
                        expert_output_accumulator->get_byte_size() >= *buffer_bytes,
                    "MoE: Invalid accumulator allocation");
    std::memset(expert_output_accumulator->data(), 0, expert_output_accumulator->get_byte_size());

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
    if (request_cache) {
        request_cache.reset();
    }

    // Keep sorted_chunk_sizes (static data)
    // Keep expert_output_accumulator (reusable allocation)

    LOG_DEBUG("Request cache reset completed");
}

}  // namespace moe
}  // namespace npuw
}  // namespace ov
