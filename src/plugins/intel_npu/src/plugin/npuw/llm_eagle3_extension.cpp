// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_eagle3_extension.hpp"

#include <algorithm>
#include <cstring>

#include "infer_request_utils.hpp"
#include "logging.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "util.hpp"

namespace ov {
namespace npuw {

bool matchEagle3HiddenStatesString(const std::string& input) {
    return input == Eagle3LayerNames::hidden_states;
}

bool matchEagle3InternalHiddenStatesString(const std::string& input) {
    return input == Eagle3LayerNames::internal_hidden_states;
}

ov::PartialShape Eagle3Extension::get_static_input(const std::shared_ptr<ov::Model>& model,
                                                   const ov::Output<ov::Node>& input,
                                                   uint32_t input_size) {
    const auto& input_name = input.get_any_name();
    const auto& input_shape = input.get_partial_shape();

    // Check if this is an Eagle3 internal_hidden_states input
    if (matchEagle3InternalHiddenStatesString(input_name)) {
        OPENVINO_ASSERT(input_shape.size() == 3u,
                        "Eagle3 internal_hidden_states must have 3 dimensions: [batch, seq_len, hidden_size]");
        OPENVINO_ASSERT(input_shape[2].is_static(),
                        "Eagle3 internal_hidden_states hidden_size dimension must be static");
        return ov::PartialShape({1, input_size, input_shape[2]});
    }

    // Check if this is an Eagle3 hidden_states input
    if (matchEagle3HiddenStatesString(input_name)) {
        OPENVINO_ASSERT(input_shape.size() == 3u,
                        "Eagle3 hidden_states must have 3 dimensions: [batch, seq_len, hidden_size]");

        const auto& hidden_dim = input_shape[2];
        if (hidden_dim.is_static()) {
            return ov::PartialShape({1, input_size, hidden_dim});
        }

        // Calculate from internal_hidden_states dimension (3x relationship)
        for (const auto& model_input : model->inputs()) {
            if (matchEagle3InternalHiddenStatesString(model_input.get_any_name())) {
                const auto& internal_shape = model_input.get_partial_shape();
                if (internal_shape.size() == 3u && internal_shape[2].is_static()) {
                    int64_t internal_hidden_size = internal_shape[2].get_length();
                    return ov::PartialShape({1, input_size, internal_hidden_size * 3});
                }
            }
        }

        OPENVINO_THROW("Eagle3 hidden_states dimension is dynamic but internal_hidden_states not found or not static");
    }

    return ov::PartialShape();
}

void Eagle3Extension::validate_hidden_state_tensor(const ov::SoPtr<ov::ITensor>& tensor, const std::string& name) {
    OPENVINO_ASSERT(ov::element::f32 == tensor->get_element_type() || ov::element::f16 == tensor->get_element_type(),
                    name + " input must be float32 or float16");
    OPENVINO_ASSERT(tensor->get_shape().size() == 3,
                    name + " input must have 3 dimensions: [batch, token_length, embedding_size]");
}

void Eagle3Extension::store_hidden_state_inputs(
    const std::vector<ov::Output<const ov::Node>>& inputs,
    const std::function<ov::SoPtr<ov::ITensor>(const ov::Output<const ov::Node>&)>& get_tensor_func) {
    // Only draft models need hidden state inputs
    if (m_role != Eagle3ModelRole::Draft) {
        return;
    }

    auto hidden_states_port = util::find_port_by_name(inputs, Eagle3LayerNames::hidden_states);
    OPENVINO_ASSERT(hidden_states_port.has_value(),
                    "Eagle3 Draft model requires 'hidden_states' input to be provided by user");
    auto hidden_states_tensor = get_tensor_func(hidden_states_port.value());
    validate_hidden_state_tensor(hidden_states_tensor, "hidden_states");
    m_hidden_states = hidden_states_tensor;

    auto internal_hidden_states_port = util::find_port_by_name(inputs, Eagle3LayerNames::internal_hidden_states);
    OPENVINO_ASSERT(internal_hidden_states_port.has_value(),
                    "Eagle3 Draft model requires 'internal_hidden_states' input to be provided by user");
    auto internal_hidden_states_tensor = get_tensor_func(internal_hidden_states_port.value());
    validate_hidden_state_tensor(internal_hidden_states_tensor, "internal_hidden_states");
    m_internal_hidden_states = internal_hidden_states_tensor;
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

    // Zero-fill the padded tensor
    ov::npuw::util::fill_tensor_bytes(padded_hidden_state, 0u);

    // Copy hidden state data to the right side
    ov::npuw::util::copy_to_right(hidden_state, padded_hidden_state);
}

}  // anonymous namespace

namespace ov {
namespace npuw {

void Eagle3Extension::initialize(const ov::AnyMap& rt_info,
                                 const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
                                 const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports) {
    bool is_eagle3_model = false;
    if (auto it = rt_info.find("eagle3_mode"); it != rt_info.end()) {
        is_eagle3_model = it->second.as<bool>();
    }

    if (!is_eagle3_model) {
        m_role = Eagle3ModelRole::None;
        LOG_DEBUG("Not an Eagle3 model (eagle3_mode not found or false in rt_info)");
        return;
    }

    // It's an Eagle3 model, now determine if it's Draft or Target based on inputs/outputs
    bool has_hidden_states_input = in_ports.find(Eagle3LayerNames::hidden_states) != in_ports.end();
    bool has_internal_hidden_states_input = in_ports.find(Eagle3LayerNames::internal_hidden_states) != in_ports.end();
    bool has_last_hidden_state_output = out_ports.find(Eagle3LayerNames::last_hidden_state) != out_ports.end();

    if (has_hidden_states_input && has_internal_hidden_states_input && has_last_hidden_state_output) {
        m_role = Eagle3ModelRole::Draft;
        LOG_INFO("Eagle3 Draft Model detected");
    } else if (!has_hidden_states_input && !has_internal_hidden_states_input && has_last_hidden_state_output) {
        m_role = Eagle3ModelRole::Target;
        LOG_INFO("Eagle3 Target Model detected");
    } else {
        OPENVINO_THROW("Eagle3 model flag set in rt_info, but model structure doesn't match Draft or Target pattern. "
                       "Draft requires: hidden_states, internal_hidden_states inputs + last_hidden_state output. "
                       "Target requires: last_hidden_state output only.");
    }
}

void Eagle3Extension::prepare_inputs(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                                     const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports) {
    // Only draft models need to prepare Eagle3 inputs
    if (m_role != Eagle3ModelRole::Draft) {
        return;
    }

    auto hidden_states_it = in_ports.find(Eagle3LayerNames::hidden_states);
    OPENVINO_ASSERT(hidden_states_it != in_ports.end(), "Eagle3 Draft model must have hidden_states input port");
    OPENVINO_ASSERT(m_hidden_states, "Eagle3 Draft model requires hidden_states input tensor to be provided");
    auto padded_hidden_states = request->get_tensor(hidden_states_it->second);
    pad_hidden_state_input(padded_hidden_states, m_hidden_states);

    auto internal_hidden_states_it = in_ports.find(Eagle3LayerNames::internal_hidden_states);
    OPENVINO_ASSERT(internal_hidden_states_it != in_ports.end(),
                    "Eagle3 Draft model must have internal_hidden_states input port");
    OPENVINO_ASSERT(m_internal_hidden_states,
                    "Eagle3 Draft model requires internal_hidden_states input tensor to be provided");
    auto padded_internal_hidden_states = request->get_tensor(internal_hidden_states_it->second);
    pad_hidden_state_input(padded_internal_hidden_states, m_internal_hidden_states);
}

void Eagle3Extension::update_last_hidden_state(
    const std::shared_ptr<ov::IAsyncInferRequest>& request,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports) {
    auto last_hidden_state_it = out_ports.find(Eagle3LayerNames::last_hidden_state);
    OPENVINO_ASSERT(last_hidden_state_it != out_ports.end(), "Eagle3 model must have last_hidden_state output port");

    m_last_hidden_state = request->get_tensor(last_hidden_state_it->second);

    LOG_VERB("Eagle3 " << (m_role == Eagle3ModelRole::Draft ? "Draft" : "Target")
                       << ": Retrieved last_hidden_state output tensor");
}

void Eagle3Extension::prepare_inputs_for_chunk(
    const std::shared_ptr<ov::IAsyncInferRequest>& request,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
    uint32_t chunk_start_token,
    uint32_t chunk_token_count) {
    // Only draft models need chunk-specific input preparation
    if (m_role != Eagle3ModelRole::Draft) {
        return;
    }

    OPENVINO_ASSERT(chunk_token_count > 0, "Chunk token count must be greater than 0");

    auto process_hidden_state_chunk = [&](const std::string& input_name,
                                          const ov::SoPtr<ov::ITensor>& original_tensor) {
        auto input_it = in_ports.find(input_name);
        OPENVINO_ASSERT(input_it != in_ports.end(), "Eagle3 Draft model must have " + input_name + " input port");
        OPENVINO_ASSERT(original_tensor, "Eagle3 Draft model requires " + input_name + " input tensor to be provided");

        auto padded_tensor = request->get_tensor(input_it->second);

        // Get original tensor shape: [batch, seq_len, embedding_size]
        const auto& original_shape = original_tensor->get_shape();
        OPENVINO_ASSERT(original_shape.size() == 3,
                        input_name + " tensor must have 3 dimensions: [batch, seq_len, embedding_size]");

        constexpr uint32_t seq_dim = 1;
        uint32_t total_tokens = static_cast<uint32_t>(original_shape[seq_dim]);

        // Validate chunk boundaries
        OPENVINO_ASSERT(chunk_start_token < total_tokens,
                        "chunk_start_token (" + std::to_string(chunk_start_token) +
                            ") must be less than total_tokens (" + std::to_string(total_tokens) + ")");

        uint32_t chunk_end_token = std::min(chunk_start_token + chunk_token_count, total_tokens);

        // Create tensor slice for current chunk along the sequence dimension
        auto chunk_tensor = util::make_tensor_slice(original_tensor, seq_dim, chunk_start_token, chunk_end_token);

        pad_hidden_state_input(padded_tensor, chunk_tensor);
        LOG_VERB("Eagle3 Draft: Set " << input_name << " chunk [" << chunk_start_token << ":" << chunk_end_token
                                      << "] for chunk processing");
    };

    // Process both hidden state inputs using the unified helper
    process_hidden_state_chunk(Eagle3LayerNames::hidden_states, m_hidden_states);
    process_hidden_state_chunk(Eagle3LayerNames::internal_hidden_states, m_internal_hidden_states);
}

}  // namespace npuw
}  // namespace ov
