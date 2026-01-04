// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_infer_utils.hpp"

#include "logging.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "util.hpp"

namespace ov {
namespace npuw {
namespace moe {

ov::Tensor slice_expert_weight(const ov::Tensor& batched_weight, size_t expert_id, size_t num_experts) {
    // Slice weight tensor from batched (num_experts, ...) to single expert (1, ...)
    auto shape = batched_weight.get_shape();
    if (shape.empty() || shape[0] != num_experts) {
        NPUW_ASSERT(false && "Expected batched weight with first dimension equal to num_experts");
    }

    // Calculate new shape: replace first dimension with 1
    ov::Shape view_shape = shape;
    view_shape[0] = 1;

    // Check if element type is sub-byte (4-bit types)
    // For sub-byte types, we can't use strides, but we can create a zero-copy view
    // by wrapping the data pointer at the correct byte offset
    auto elem_type = batched_weight.get_element_type();
    if (elem_type == ov::element::nf4 || elem_type == ov::element::u4 || elem_type == ov::element::i4) {
        // Calculate byte-level offset for this expert
        size_t total_byte_size = batched_weight.get_byte_size();
        size_t expert_byte_size = total_byte_size / num_experts;

        // Get pointer to this expert's data
        const uint8_t* base_ptr = static_cast<const uint8_t*>(batched_weight.data());
        void* expert_ptr = const_cast<uint8_t*>(base_ptr + expert_id * expert_byte_size);

        // Create zero-copy tensor wrapping the expert's data slice
        ov::Tensor expert_tensor(elem_type, view_shape, expert_ptr);

        LOG_DEBUG("Sliced expert " << expert_id << " weight (4-bit, zero-copy): " << shape << " -> " << view_shape);
        return expert_tensor;
    }

    // For >= 8-bit types, use util::view to create zero-copy strided view
    auto view_impl = ov::npuw::util::view(ov::get_tensor_impl(batched_weight), 0, expert_id, 1);
    ov::Tensor view_tensor = ov::make_tensor(view_impl);

    LOG_DEBUG("Sliced expert " << expert_id << " weight using util::view: " << shape << " -> "
                               << view_tensor.get_shape());

    return view_tensor;
}

std::vector<size_t> parse_selected_experts_from_router(const ov::SoPtr<ov::ITensor>& router_output,
                                                       size_t num_experts,
                                                       std::map<size_t, std::vector<size_t>>& token_to_experts,
                                                       std::map<size_t, std::vector<size_t>>& expert_to_tokens) {
    if (!router_output) {
        NPUW_ASSERT(false && "Router output tensor is null");
    }

    // Clear input maps
    token_to_experts.clear();
    expert_to_tokens.clear();

    // Expected router output shape: [num_experts, 1, token_num, 1]
    auto shape = router_output->get_shape();
    if (shape.size() != 4 || shape[0] != num_experts || shape[1] != 1 || shape[3] != 1) {
        NPUW_ASSERT(false && "Unexpected router output shape!");
    }

    size_t num_tokens = shape[2];  // token_num from shape

    // Parse which expert each token selects based on non-zero weights
    auto parse_experts = [&](auto* data) {
        // For each token, find which experts have non-zero weights
        for (size_t token_id = 0; token_id < num_tokens; ++token_id) {
            for (size_t expert_id = 0; expert_id < num_experts; ++expert_id) {
                // Index calculation for shape [num_experts, 1, token_num, 1]
                // data[expert_id, 0, token_id, 0]
                size_t idx = expert_id * num_tokens + token_id;

                float value = std::abs(static_cast<float>(data[idx]));
                if (value > 1e-6f) {
                    // This token selected this expert
                    token_to_experts[token_id].push_back(expert_id);
                    expert_to_tokens[expert_id].push_back(token_id);
                }
            }
        }
    };

    auto elem_type = router_output->get_element_type();
    if (elem_type == ov::element::f32) {
        parse_experts(router_output->data<float>());
    } else if (elem_type == ov::element::f16) {
        parse_experts(router_output->data<ov::float16>());
    } else {
        NPUW_ASSERT(false && "Unsupported element type in router output tensor");
    }

    // Convert expert_to_tokens keys to vector
    std::vector<size_t> selected_experts;
    selected_experts.reserve(expert_to_tokens.size());
    for (const auto& [expert_id, tokens] : expert_to_tokens) {
        selected_experts.push_back(expert_id);
    }

    return selected_experts;
}

void set_tensor_optimized(ov::SoPtr<ov::IAsyncInferRequest> request,
                          const ov::Output<const ov::Node>& iport,
                          const ov::SoPtr<ov::ITensor>& tensor_impl) {
    // Optimization: For small tensors, use copy instead of set_tensor to avoid overhead
    // Threshold: 11520 bytes (~11.25KB, typically 5760 f16 elements or 2880 f32 elements)
    // Empirically verified to have performance benefit for small tensors
    // Typical shapes: 1x2880x1 (f16: 5760 bytes), 1x5760x1 (f16: 11520 bytes)
    constexpr size_t SMALL_TENSOR_THRESHOLD_BYTES = 11520;

    size_t tensor_bytes = tensor_impl->get_byte_size();

    if (tensor_bytes <= SMALL_TENSOR_THRESHOLD_BYTES) {
        // Small tensor: direct copy to avoid set_tensor overhead (~0.65ms per call)
        // Copy is faster for small tensors due to avoiding NPU plugin overhead
        auto clparam = request->get_tensor(iport);
        tensor_impl->copy_to(clparam._ptr);
        LOG_DEBUG("Using copy for small tensor (" << tensor_bytes << " bytes)");
    } else {
        // Large tensor: use set_tensor (zero-copy)
        request->set_tensor(iport, tensor_impl);
    }
}

void gather_router_scores(const ov::SoPtr<ov::ITensor>& router_source,
                          const ov::SoPtr<ov::ITensor>& router_dest,
                          size_t expert_id,
                          const std::vector<size_t>& token_ids,
                          size_t chunk_start,
                          size_t chunk_size) {
    auto router_source_shape = router_source->get_shape();

    // Calculate expert offset in source tensor
    size_t expert_offset;
    if (router_source_shape.size() == 4) {
        expert_offset = expert_id * router_source_shape[2];  // [num_experts, 1, token_num, 1]
    } else if (router_source_shape.size() == 2) {
        expert_offset = expert_id * router_source_shape[1];  // [num_experts, token_num]
    } else {
        NPUW_ASSERT(false && "Unexpected router source shape");
    }

    // Gather router scores for chunk tokens
    if (router_source->get_element_type() == ov::element::f16) {
        const auto* src_base = router_source->data<ov::float16>() + expert_offset;
        auto* dst_base = router_dest->data<ov::float16>();
        for (size_t i = 0; i < chunk_size; ++i) {
            dst_base[i] = src_base[token_ids[chunk_start + i]];
        }
    } else if (router_source->get_element_type() == ov::element::f32) {
        const auto* src_base = router_source->data<float>() + expert_offset;
        auto* dst_base = router_dest->data<float>();
        for (size_t i = 0; i < chunk_size; ++i) {
            dst_base[i] = src_base[token_ids[chunk_start + i]];
        }
    } else {
        NPUW_ASSERT(false && "Unsupported router element type for gathering");
    }
}

void gather_expert_inputs(const ov::SoPtr<ov::ITensor>& input_source,
                          const ov::SoPtr<ov::ITensor>& input_dest,
                          const std::vector<size_t>& token_ids,
                          size_t chunk_start,
                          size_t chunk_size) {
    auto input_shape = input_source->get_shape();

    // Determine dimensions
    size_t hidden_dim;
    size_t token_stride;
    if (input_shape.size() == 2) {
        hidden_dim = input_shape[1];
        token_stride = hidden_dim;
    } else if (input_shape.size() == 4) {
        hidden_dim = input_shape[3];
        token_stride = hidden_dim;
    } else {
        NPUW_ASSERT(false && "Unexpected expert input tensor shape");
    }

    // Gather input embeddings for chunk tokens
    if (input_source->get_element_type() == ov::element::f16) {
        const auto* src_base = input_source->data<ov::float16>();
        auto* dst_base = input_dest->data<ov::float16>();
        for (size_t i = 0; i < chunk_size; ++i) {
            size_t token_id = token_ids[chunk_start + i];
            const auto* src_token = src_base + token_id * token_stride;
            auto* dst_token = dst_base + i * hidden_dim;
            std::memcpy(dst_token, src_token, hidden_dim * sizeof(ov::float16));
        }
    } else if (input_source->get_element_type() == ov::element::f32) {
        const auto* src_base = input_source->data<float>();
        auto* dst_base = input_dest->data<float>();
        for (size_t i = 0; i < chunk_size; ++i) {
            size_t token_id = token_ids[chunk_start + i];
            const auto* src_token = src_base + token_id * token_stride;
            auto* dst_token = dst_base + i * hidden_dim;
            std::memcpy(dst_token, src_token, hidden_dim * sizeof(float));
        }
    } else {
        NPUW_ASSERT(false && "Unsupported expert input element type for gathering");
    }
}

void scatter_expert_outputs(const ov::SoPtr<ov::ITensor>& expert_output,
                            const ov::SoPtr<ov::ITensor>& global_output_buffer,
                            const std::vector<size_t>& token_ids,
                            size_t chunk_start,
                            size_t chunk_size,
                            size_t embed_dim,
                            size_t input_token_count,
                            const std::vector<size_t>& expert_slots_for_tokens) {
    auto elem_type = global_output_buffer->get_element_type();

    for (size_t i = 0; i < chunk_size; ++i) {
        size_t original_token_id = token_ids[chunk_start + i];
        size_t expert_slot = expert_slots_for_tokens[chunk_start + i];

        // Calculate offsets
        size_t src_offset = i * embed_dim;
        size_t dst_offset = expert_slot * input_token_count * embed_dim + original_token_id * embed_dim;

        // Scatter output to global buffer
        if (elem_type == ov::element::f32) {
            const float* src = expert_output->data<float>() + src_offset;
            float* dst = global_output_buffer->data<float>() + dst_offset;
            std::memcpy(dst, src, embed_dim * sizeof(float));
        } else if (elem_type == ov::element::f16) {
            const ov::float16* src = expert_output->data<ov::float16>() + src_offset;
            ov::float16* dst = global_output_buffer->data<ov::float16>() + dst_offset;
            std::memcpy(dst, src, embed_dim * sizeof(ov::float16));
        } else {
            OPENVINO_THROW("MoE: Unsupported element type for chunk output relayout: ", elem_type);
        }
    }
}

}  // namespace moe
}  // namespace npuw
}  // namespace ov
