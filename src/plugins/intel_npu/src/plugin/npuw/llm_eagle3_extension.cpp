// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_eagle3_extension.hpp"

#include <algorithm>
#include <cstring>

#include "logging.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace ov {
namespace npuw {

bool matchEagle3HiddenStatesString(const std::string& input) {
    return input == Eagle3LayerNames::hidden_states;
}

bool matchEagle3InternalHiddenStatesString(const std::string& input) {
    return input == Eagle3LayerNames::internal_hidden_states;
}

void Eagle3Extension::validate_tensor(const ov::SoPtr<ov::ITensor>& tensor, const std::string& name) {
    OPENVINO_ASSERT(ov::element::f32 == tensor->get_element_type() || ov::element::f16 == tensor->get_element_type(),
                    name + " input must be float32 or float16");
    OPENVINO_ASSERT(tensor->get_shape().size() == 3,
                    name + " input must have 3 dimensions: [batch, token_length, embedding_size]");
}

}  // namespace npuw
}  // namespace ov

namespace {

void pad_hidden_state_input(const ov::SoPtr<ov::ITensor>& padded_hidden_state,
                            const ov::SoPtr<ov::ITensor>& hidden_state) {
    // Pad the token_length dimension (dimension 1) of hidden state input [batch, token_length, embedding_size]
    auto padded_shape = padded_hidden_state->get_shape();
    auto hidden_state_shape = hidden_state->get_shape();
    OPENVINO_ASSERT(hidden_state_shape.size() == 3,
                    "Hidden state input should have 3 dimensions: [batch, token_length, embedding_size]");
    OPENVINO_ASSERT(padded_shape.size() == 3,
                    "Padded hidden state should have 3 dimensions: [batch, token_length, embedding_size]");

    OPENVINO_ASSERT(hidden_state_shape[0] == 1, "Batch size must be 1 for Eagle3 hidden states");
    OPENVINO_ASSERT(padded_shape[0] == 1, "Padded batch size must be 1 for Eagle3 hidden states");

    OPENVINO_ASSERT(padded_shape[2] == hidden_state_shape[2], "Embedding size must match");
    OPENVINO_ASSERT(padded_shape[1] >= hidden_state_shape[1], "Padded token length must be >= input token length");

    // Calculate dimensions (batch size is always 1)
    const size_t input_token_len = hidden_state_shape[1];
    const size_t padded_token_len = padded_shape[1];
    const size_t embedding_size = hidden_state_shape[2];
    const size_t token_padding = padded_token_len - input_token_len;
    const size_t elem_size = hidden_state->get_element_type().size();

    // Calculate byte sizes for efficient bulk operations
    const size_t input_bytes = input_token_len * embedding_size * elem_size;
    const size_t padding_bytes = token_padding * embedding_size * elem_size;
    const size_t total_bytes = padded_hidden_state->get_byte_size();

    // Get raw data pointers
    const uint8_t* src_data = reinterpret_cast<const uint8_t*>(hidden_state->data());
    uint8_t* dst_data = reinterpret_cast<uint8_t*>(padded_hidden_state->data());

    // Use memset for efficient zero-filling (left padding)
    std::memset(dst_data, 0, total_bytes);

    // Use memcpy for efficient data copying (right-aligned after padding)
    std::memcpy(dst_data + padding_bytes, src_data, input_bytes);
}

}  // anonymous namespace

namespace ov {
namespace npuw {

void Eagle3Extension::initialize(const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
                                 const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports) {
    bool has_hidden_states_input = in_ports.find(Eagle3LayerNames::hidden_states) != in_ports.end();
    bool has_internal_hidden_states_input = in_ports.find(Eagle3LayerNames::internal_hidden_states) != in_ports.end();
    bool has_last_hidden_state_output = out_ports.find(Eagle3LayerNames::last_hidden_state) != out_ports.end();

    if (has_hidden_states_input && has_internal_hidden_states_input && has_last_hidden_state_output) {
        m_role = Eagle3ModelRole::Draft;
        LOG_INFO("Eagle3 Draft Model detected");
        LOG_DEBUG("Eagle3 draft model: inputs hidden_states, internal_hidden_states, and output last_hidden_state");
    } else if (!has_hidden_states_input && !has_internal_hidden_states_input && has_last_hidden_state_output) {
        m_role = Eagle3ModelRole::Target;
        LOG_INFO("Eagle3 Target Model detected");
        LOG_DEBUG("Eagle3 target model: only output last_hidden_state");
    } else {
        m_role = Eagle3ModelRole::None;
    }
}

void Eagle3Extension::prepare_inputs(std::shared_ptr<ov::IAsyncInferRequest> request,
                                     const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports) {
    // Only draft models need to prepare Eagle3 inputs
    if (m_role != Eagle3ModelRole::Draft) {
        return;
    }

    auto hidden_states_it = in_ports.find(Eagle3LayerNames::hidden_states);
    if (hidden_states_it != in_ports.end() && m_hidden_states) {
        auto padded_hidden_states = request->get_tensor(hidden_states_it->second);
        pad_hidden_state_input(padded_hidden_states, m_hidden_states);
        LOG_VERB("Eagle3 Draft: Set hidden_states input tensor");
    }

    auto internal_hidden_states_it = in_ports.find(Eagle3LayerNames::internal_hidden_states);
    if (internal_hidden_states_it != in_ports.end() && m_internal_hidden_states) {
        auto padded_internal_hidden_states = request->get_tensor(internal_hidden_states_it->second);
        pad_hidden_state_input(padded_internal_hidden_states, m_internal_hidden_states);
        LOG_VERB("Eagle3 Draft: Set internal_hidden_states input tensor");
    }
}

void Eagle3Extension::process_outputs(std::shared_ptr<ov::IAsyncInferRequest> request,
                                      const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports) {
    // Both draft and target models have last_hidden_state output
    if (m_role == Eagle3ModelRole::Draft || m_role == Eagle3ModelRole::Target) {
        auto last_hidden_state_it = out_ports.find(Eagle3LayerNames::last_hidden_state);
        if (last_hidden_state_it != out_ports.end()) {
            m_last_hidden_state = request->get_tensor(last_hidden_state_it->second);
            LOG_VERB("Eagle3 " << (m_role == Eagle3ModelRole::Draft ? "Draft" : "Target")
                               << ": Retrieved last_hidden_state output tensor");
        }
    }
}

void Eagle3Extension::prepare_inputs_for_chunk(
    std::shared_ptr<ov::IAsyncInferRequest> request,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
    uint32_t chunk_start_token,
    uint32_t chunk_token_count) {
    // Only draft models need chunk-specific input preparation
    if (m_role != Eagle3ModelRole::Draft) {
        return;
    }

    // Helper lambda to process a single hidden state input with chunk slicing
    auto process_hidden_state_chunk = [&](const std::string& input_name,
                                          const ov::SoPtr<ov::ITensor>& original_tensor) {
        auto input_it = in_ports.find(input_name);
        if (input_it != in_ports.end() && original_tensor) {
            auto padded_tensor = request->get_tensor(input_it->second);

            // Create chunk slice from the original tensor
            // Shape: [batch, seq_len, embedding_size] - always 3D
            const auto& original_shape = original_tensor->get_shape();
            if (original_shape.size() == 3) {
                constexpr uint32_t seq_dim = 1;  // Token length dimension is always dimension 1

                // Ensure chunk boundaries are valid
                uint32_t total_tokens = static_cast<uint32_t>(original_shape[seq_dim]);
                uint32_t chunk_end_token = std::min(chunk_start_token + chunk_token_count, total_tokens);

                if (chunk_start_token < total_tokens && chunk_token_count > 0) {
                    // Create tensor slice for current chunk
                    ov::Shape start_shape(original_shape.size(), 0);
                    ov::Shape end_shape = original_shape;

                    start_shape[seq_dim] = chunk_start_token;
                    end_shape[seq_dim] = chunk_end_token;

                    auto chunk_tensor =
                        ov::get_tensor_impl(ov::Tensor(ov::make_tensor(original_tensor), start_shape, end_shape));

                    pad_hidden_state_input(padded_tensor, chunk_tensor);
                    LOG_VERB("Eagle3 Draft: Set " << input_name << " chunk [" << chunk_start_token << ":"
                                                  << chunk_end_token << "] for chunk processing");
                }
            }
        }
    };

    // Process both hidden state inputs using the unified helper
    process_hidden_state_chunk(Eagle3LayerNames::hidden_states, m_hidden_states);
    process_hidden_state_chunk(Eagle3LayerNames::internal_hidden_states, m_internal_hidden_states);
}

}  // namespace npuw
}  // namespace ov
