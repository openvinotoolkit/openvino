// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_eagle3_extension.hpp"

#include <algorithm>

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

// Forward declaration of padding function
void pad_hidden_state_input(const ov::SoPtr<ov::ITensor>& padded_hidden_state,
                            const ov::SoPtr<ov::ITensor>& hidden_state);

template <typename T>
void fill_tensor(ov::SoPtr<ov::ITensor> tensor, T fill_val, size_t offset = 0u) {
    T* tensor_data = tensor->data<T>();
    std::fill(tensor_data + offset, tensor_data + tensor->get_size(), fill_val);
}

void fill_tensor_bytes(ov::SoPtr<ov::ITensor> tensor, uint8_t fill_val) {
    auto* tensor_data = reinterpret_cast<uint8_t*>(tensor->data());
    std::fill_n(tensor_data, tensor->get_byte_size(), fill_val);
}

void pad_hidden_state_input(const ov::SoPtr<ov::ITensor>& padded_hidden_state,
                            const ov::SoPtr<ov::ITensor>& hidden_state) {
    // Pad the token_length dimension (dimension 1) of hidden state input [batch, token_length, embedding_size]
    auto padded_shape = padded_hidden_state->get_shape();
    auto hidden_state_shape = hidden_state->get_shape();
    OPENVINO_ASSERT(hidden_state_shape.size() == 3,
                    "Hidden state input should have 3 dimensions: [batch, token_length, embedding_size]");
    OPENVINO_ASSERT(padded_shape.size() == 3,
                    "Padded hidden state should have 3 dimensions: [batch, token_length, embedding_size]");
    // Check batch size and embedding size match
    OPENVINO_ASSERT(padded_shape[0] == hidden_state_shape[0], "Batch size must match");
    OPENVINO_ASSERT(padded_shape[2] == hidden_state_shape[2], "Embedding size must match");
    // Check token length padding is valid
    OPENVINO_ASSERT(padded_shape[1] >= hidden_state_shape[1], "Padded token length must be >= input token length");

    // Calculate dimensions
    const size_t batch_size = hidden_state_shape[0];
    const size_t input_token_len = hidden_state_shape[1];
    const size_t padded_token_len = padded_shape[1];
    const size_t embedding_size = hidden_state_shape[2];
    const size_t token_padding = padded_token_len - input_token_len;

    // Get data pointers based on element type
    if (hidden_state->get_element_type() == ov::element::f32) {
        // Handle float32 data
        float* padded_data = padded_hidden_state->data<float>();
        const float* input_data = hidden_state->data<float>();

        // Fill with zeros first (left padding)
        fill_tensor<float>(padded_hidden_state, 0.0f);

        // Copy actual data to the right side of each batch
        for (size_t b = 0; b < batch_size; ++b) {
            size_t input_batch_offset = b * input_token_len * embedding_size;
            size_t padded_batch_offset = b * padded_token_len * embedding_size + token_padding * embedding_size;

            std::copy_n(input_data + input_batch_offset,
                        input_token_len * embedding_size,
                        padded_data + padded_batch_offset);
        }
    } else if (hidden_state->get_element_type() == ov::element::f16) {
        // Handle float16 data (using byte copy for simplicity)
        uint8_t* padded_data = reinterpret_cast<uint8_t*>(padded_hidden_state->data());
        const uint8_t* input_data = reinterpret_cast<const uint8_t*>(hidden_state->data());

        // Fill with zeros first
        fill_tensor_bytes(padded_hidden_state, 0);

        const size_t elem_size = hidden_state->get_element_type().size();

        // Copy actual data to the right side of each batch
        for (size_t b = 0; b < batch_size; ++b) {
            size_t input_batch_offset = b * input_token_len * embedding_size * elem_size;
            size_t padded_batch_offset =
                b * padded_token_len * embedding_size * elem_size + token_padding * embedding_size * elem_size;

            std::copy_n(input_data + input_batch_offset,
                        input_token_len * embedding_size * elem_size,
                        padded_data + padded_batch_offset);
        }
    } else {
        OPENVINO_THROW("Unsupported element type for hidden state input: ", hidden_state->get_element_type());
    }
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

void Eagle3Extension::prepare_inputs_impl(std::shared_ptr<ov::IAsyncInferRequest> request,
                                          const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports) {
    // Only draft models need to prepare Eagle3 inputs
    if (m_role != Eagle3ModelRole::Draft) {
        return;
    }

    // Set hidden_states input if available
    auto hidden_states_it = in_ports.find(Eagle3LayerNames::hidden_states);
    if (hidden_states_it != in_ports.end() && m_hidden_states) {
        auto padded_hidden_states = request->get_tensor(hidden_states_it->second);
        pad_hidden_state_input(padded_hidden_states, m_hidden_states);
        LOG_VERB("Eagle3 Draft: Set hidden_states input tensor");
    }

    // Set internal_hidden_states input if available
    auto internal_hidden_states_it = in_ports.find(Eagle3LayerNames::internal_hidden_states);
    if (internal_hidden_states_it != in_ports.end() && m_internal_hidden_states) {
        auto padded_internal_hidden_states = request->get_tensor(internal_hidden_states_it->second);
        pad_hidden_state_input(padded_internal_hidden_states, m_internal_hidden_states);
        LOG_VERB("Eagle3 Draft: Set internal_hidden_states input tensor");
    }
}

void Eagle3Extension::process_outputs_impl(
    std::shared_ptr<ov::IAsyncInferRequest> request,
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

void Eagle3Extension::prepare_inputs(std::shared_ptr<ov::IAsyncInferRequest> request,
                                     const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports) {
    // Automatically handle inputs based on model role
    if (m_role == Eagle3ModelRole::Draft) {
        prepare_inputs_impl(request, in_ports);
    }
    // Target models and non-Eagle3 models require no input preparation
}

void Eagle3Extension::process_outputs(std::shared_ptr<ov::IAsyncInferRequest> request,
                                      const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports) {
    // Automatically handle outputs based on model role
    if (m_role == Eagle3ModelRole::Target || m_role == Eagle3ModelRole::Draft) {
        process_outputs_impl(request, out_ports);
    }
    // non-Eagle3 models require no output processing
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
