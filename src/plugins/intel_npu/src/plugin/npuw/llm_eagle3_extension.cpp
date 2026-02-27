// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_eagle3_extension.hpp"

#include <algorithm>
#include <cstring>
#include <regex>

#include "infer_request_utils.hpp"
#include "logging.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "util.hpp"

namespace ov {
namespace npuw {

bool matchEagle3HiddenStatesString(const std::string& input) {
    return input == Eagle3LayerNames::hidden_states;
}

bool matchEagle3TreeMaskString(const std::string& input) {
    return input == Eagle3LayerNames::eagle_tree_mask;
}

ov::PartialShape Eagle3Extension::get_static_input(const std::shared_ptr<ov::Model>& model,
                                                   const ov::Output<ov::Node>& input,
                                                   uint32_t input_size) {
    const auto& input_name = input.get_any_name();
    const auto& input_shape = input.get_partial_shape();

    // Check if this is an Eagle3 hidden_states input
    if (matchEagle3HiddenStatesString(input_name)) {
        OPENVINO_ASSERT(input_shape.size() == 3u,
                        "Eagle3 hidden_states must have 3 dimensions: [batch, seq_len, hidden_size]");

        if (input_shape[2].is_static()) {
            return ov::PartialShape({1, input_size, input_shape[2]});
        }

        for (const auto& model_output : model->outputs()) {
            if (model_output.get_any_name() == Eagle3LayerNames::last_hidden_state) {
                const auto& output_shape = model_output.get_partial_shape();
                OPENVINO_ASSERT(output_shape.size() == 3u,
                                "Eagle3 last_hidden_state must have 3 dimensions: [batch, seq_len, hidden_size]");
                if (output_shape[2].is_static()) {
                    return ov::PartialShape({1, input_size, output_shape[2]});
                }
            }
        }

        OPENVINO_THROW("Eagle3 hidden_states hidden_size dimension is dynamic and "
                       "could not be inferred from last_hidden_state output");
    }

    return ov::PartialShape();
}

void Eagle3Extension::validate_hidden_state_tensor(const ov::SoPtr<ov::ITensor>& tensor, const std::string& name) {
    OPENVINO_ASSERT(ov::element::f32 == tensor->get_element_type() || ov::element::f16 == tensor->get_element_type(),
                    name + " input must be float32 or float16");
    OPENVINO_ASSERT(tensor->get_shape().size() == 3,
                    name + " input must have 3 dimensions: [batch, token_length, embedding_size]");
}

void Eagle3Extension::validate_tree_mask_tensor(const ov::SoPtr<ov::ITensor>& tensor, const std::string& name) {
    OPENVINO_ASSERT(ov::element::f32 == tensor->get_element_type() || ov::element::f16 == tensor->get_element_type(),
                    name + " input must be float32 or float16");
    OPENVINO_ASSERT(tensor->get_shape().size() == 4,
                    name + " input must have 4 dimensions: [batch, 1, input_size, kvcache_size]");
    OPENVINO_ASSERT(tensor->get_shape()[0] == 1, name + " input batch dimension must be 1");
    OPENVINO_ASSERT(tensor->get_shape()[1] == 1, name + " input second dimension must be 1");
}

void Eagle3Extension::store_user_inputs(const ov::IInferRequest& request,
                                        const std::vector<ov::Output<const ov::Node>>& inputs) {
    // Store eagle_tree_mask for both Draft and Target models
    auto tree_mask_port = util::find_port_by_name(inputs, Eagle3LayerNames::eagle_tree_mask);
    if (tree_mask_port.has_value()) {
        auto tree_mask_tensor = request.get_tensor(tree_mask_port.value());
        validate_tree_mask_tensor(tree_mask_tensor, "eagle_tree_mask");
        m_eagle_tree_mask = tree_mask_tensor;
    }

    // Store hidden_states for Draft model only
    auto hidden_states_port = util::find_port_by_name(inputs, Eagle3LayerNames::hidden_states);
    if (hidden_states_port.has_value()) {
        auto hidden_states_tensor = request.get_tensor(hidden_states_port.value());
        validate_hidden_state_tensor(hidden_states_tensor, "hidden_states");
        m_hidden_states = hidden_states_tensor;
    }
}

}  // namespace npuw
}  // namespace ov

namespace {

void pad_hidden_state_input(const ov::SoPtr<ov::ITensor>& hidden_state,
                            const ov::SoPtr<ov::ITensor>& padded_hidden_state) {
    // Pad the token_length dimension (dimension 1) of hidden state input [batch, token_length, embedding_size]
    auto hidden_state_shape = hidden_state->get_shape();
    auto padded_shape = padded_hidden_state->get_shape();
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

void pad_tree_mask_input(const ov::SoPtr<ov::ITensor>& tree_mask, const ov::SoPtr<ov::ITensor>& padded_tree_mask) {
    // Pad the tree mask tensor [batch, 1, input_size, kvcache_size]
    // Two scenarios:
    // 1. Prefill phase: user provides small mask, model expects {1, 1, 1, 1}
    // 2. Generate phase: user provides full mask, model expects {1, 1, input_size, kvcache_size}
    // Padding is applied to dimensions 2 and 3 (seq_len dimensions)
    auto padded_shape = padded_tree_mask->get_shape();
    auto tree_mask_shape = tree_mask->get_shape();

    OPENVINO_ASSERT(tree_mask_shape.size() == 4,
                    "Tree mask input should have 4 dimensions: [batch, 1, input_size, kvcache_size]");
    OPENVINO_ASSERT(padded_shape.size() == 4,
                    "Padded tree mask should have 4 dimensions: [batch, 1, input_size, kvcache_size]");

    OPENVINO_ASSERT(tree_mask_shape[0] == 1, "Batch size must be 1 for Eagle3 tree mask");
    OPENVINO_ASSERT(padded_shape[0] == 1, "Padded batch size must be 1 for Eagle3 tree mask");
    OPENVINO_ASSERT(tree_mask_shape[1] == 1, "Second dimension must be 1 for Eagle3 tree mask");
    OPENVINO_ASSERT(padded_shape[1] == 1, "Padded second dimension must be 1 for Eagle3 tree mask");

    OPENVINO_ASSERT(padded_shape[2] >= tree_mask_shape[2], "Padded input_size must be >= original input_size");
    OPENVINO_ASSERT(padded_shape[3] >= tree_mask_shape[3], "Padded kvcache_size must be >= original kvcache_size");

    const size_t input_size = tree_mask_shape[2];
    const size_t padded_input_size = padded_shape[2];
    const size_t kvcache_size = tree_mask_shape[3];
    const size_t padded_kvcache_size = padded_shape[3];
    const size_t elem_size = tree_mask->get_element_type().size();

    if (input_size == 1 && kvcache_size == 1) {
        std::memcpy(padded_tree_mask->data(), tree_mask->data(), tree_mask->get_byte_size());
        return;
    }

    // Get raw data pointers
    const uint8_t* src_data = reinterpret_cast<const uint8_t*>(tree_mask->data());
    uint8_t* dst_data = reinterpret_cast<uint8_t*>(padded_tree_mask->data());

    // Zero-fill the entire padded tensor first
    std::memset(dst_data, 0, padded_tree_mask->get_byte_size());

    // Copy each row of the original tensor to the padded tensor
    // Left-pad both dimensions (top-left corner is padding, bottom-right is data)
    const size_t row_padding = padded_input_size - input_size;
    const size_t col_padding = padded_kvcache_size - kvcache_size;

    for (size_t i = 0; i < input_size; ++i) {
        const uint8_t* src_row = src_data + i * kvcache_size * elem_size;
        uint8_t* dst_row = dst_data + (i + row_padding) * padded_kvcache_size * elem_size + col_padding * elem_size;
        std::memcpy(dst_row, src_row, kvcache_size * elem_size);
    }
}

}  // anonymous namespace

namespace ov {
namespace npuw {

void Eagle3Extension::initialize(bool is_eagle_model,
                                 const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
                                 const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports) {
    if (!is_eagle_model) {
        m_role = Eagle3ModelRole::None;
        return;
    }

    // It's an Eagle3 model, now determine if it's Draft or Target based on inputs/outputs
    bool has_hidden_states_input = in_ports.find(Eagle3LayerNames::hidden_states) != in_ports.end();
    bool has_eagle_tree_mask_input = in_ports.find(Eagle3LayerNames::eagle_tree_mask) != in_ports.end();
    bool has_last_hidden_state_output = out_ports.find(Eagle3LayerNames::last_hidden_state) != out_ports.end();

    if (has_hidden_states_input && has_eagle_tree_mask_input && has_last_hidden_state_output) {
        m_role = Eagle3ModelRole::Draft;
        LOG_INFO("Eagle3 Draft Model detected");
    } else if (!has_hidden_states_input && has_eagle_tree_mask_input && has_last_hidden_state_output) {
        m_role = Eagle3ModelRole::Target;
        LOG_INFO("Eagle3 Target Model detected");
    } else {
        OPENVINO_THROW(
            "Eagle3 mode enabled via NPUW_EAGLE property, but model structure doesn't match Draft or Target pattern. "
            "Draft requires: hidden_states input + eagle_tree_mask input + last_hidden_state output. "
            "Target requires: eagle_tree_mask input + last_hidden_state output.");
    }

    // Create sampling state for external pipeline communication
    if (m_role != Eagle3ModelRole::None) {
        m_sampling_state = std::make_shared<Eagle3SamplingState>();
    }
}

void Eagle3Extension::prepare_inputs(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                                     const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports) {
    // Eagle3 models (both Draft and Target) MUST have eagle_tree_mask
    auto tree_mask_it = in_ports.find(Eagle3LayerNames::eagle_tree_mask);
    OPENVINO_ASSERT(tree_mask_it != in_ports.end(), "Eagle3 model must have eagle_tree_mask input port");
    OPENVINO_ASSERT(m_eagle_tree_mask, "Eagle3 model requires eagle_tree_mask tensor to be provided by user");

    auto padded_tree_mask = request->get_tensor(tree_mask_it->second);
    pad_tree_mask_input(m_eagle_tree_mask, padded_tree_mask);
    LOG_VERB("Eagle3: Set eagle_tree_mask input tensor");

    // Draft models MUST have hidden_states
    if (m_role == Eagle3ModelRole::Draft) {
        auto hidden_states_it = in_ports.find(Eagle3LayerNames::hidden_states);
        OPENVINO_ASSERT(hidden_states_it != in_ports.end(), "Eagle3 Draft model must have hidden_states input port");
        OPENVINO_ASSERT(m_hidden_states, "Eagle3 Draft model requires hidden_states tensor to be provided by user");

        auto padded_hidden_states = request->get_tensor(hidden_states_it->second);
        pad_hidden_state_input(m_hidden_states, padded_hidden_states);
        LOG_VERB("Eagle3: Set hidden_states input tensor");
    }
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

void Eagle3Extension::accumulate_chunk_last_hidden_state(
    const std::shared_ptr<ov::IAsyncInferRequest>& request,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports,
    uint32_t chunk_token_count,
    uint32_t total_seq_len) {
    if (m_role == Eagle3ModelRole::None) {
        return;
    }

    auto last_hidden_state_it = out_ports.find(Eagle3LayerNames::last_hidden_state);
    OPENVINO_ASSERT(last_hidden_state_it != out_ports.end(), "Eagle3 model must have last_hidden_state output port");

    auto chunk_output = request->get_tensor(last_hidden_state_it->second);
    const auto& chunk_shape = chunk_output->get_shape();

    OPENVINO_ASSERT(chunk_shape.size() == 3, "last_hidden_state must have 3 dimensions: [batch, seq_len, hidden_size]");

    const uint32_t batch_size = static_cast<uint32_t>(chunk_shape[0]);
    const uint32_t chunk_seq_len = static_cast<uint32_t>(chunk_shape[1]);
    const uint32_t hidden_size = static_cast<uint32_t>(chunk_shape[2]);

    OPENVINO_ASSERT(batch_size == 1, "Batch size must be 1 for Eagle3");
    OPENVINO_ASSERT(chunk_token_count <= chunk_seq_len, "chunk_token_count must be <= chunk_seq_len");

    // Pre-allocate tensor on first chunk
    if (!m_last_hidden_state) {
        m_last_hidden_state = ov::get_tensor_impl(
            ov::Tensor(chunk_output->get_element_type(), ov::Shape{batch_size, total_seq_len, hidden_size}));
        m_chunked_seq_offset = 0;

        LOG_VERB("Eagle3: Pre-allocated last_hidden_state tensor with shape=[" << batch_size << "," << total_seq_len
                                                                               << "," << hidden_size << "]");
    }

    const auto& target_shape = m_last_hidden_state->get_shape();
    const uint32_t target_total_len = static_cast<uint32_t>(target_shape[1]);

    OPENVINO_ASSERT(target_total_len == total_seq_len,
                    "Pre-allocated tensor size (" + std::to_string(target_total_len) + ") must match total_seq_len (" +
                        std::to_string(total_seq_len) + ")");

    OPENVINO_ASSERT(m_chunked_seq_offset + chunk_token_count <= target_total_len,
                    "Can't write chunk by stored chunked sequence offset and requested number of tokens, as it will "
                    "exceed pre-allocated size");

    // Extract only the rightmost chunk_token_count tokens from the output
    // The chunk_output is right-aligned with padding on the left
    constexpr uint32_t seq_dim = 1;
    const uint32_t chunk_start_offset = chunk_seq_len - chunk_token_count;

    auto chunk_output_slice = util::make_tensor_slice(chunk_output, seq_dim, chunk_start_offset, chunk_seq_len);

    auto target_slice = util::make_tensor_slice(m_last_hidden_state,
                                                seq_dim,
                                                m_chunked_seq_offset,
                                                m_chunked_seq_offset + chunk_token_count);

    chunk_output_slice->copy_to(target_slice._ptr);

    LOG_VERB("Eagle3: Copied chunk [" << chunk_start_offset << ":" << chunk_seq_len << "] to position ["
                                      << m_chunked_seq_offset << ":" << (m_chunked_seq_offset + chunk_token_count)
                                      << "], " << chunk_token_count << " tokens");

    m_chunked_seq_offset += chunk_token_count;
}

void Eagle3Extension::prepare_inputs_for_chunk(
    const std::shared_ptr<ov::IAsyncInferRequest>& request,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
    uint32_t chunk_start_token,
    uint32_t chunk_token_count) {
    // Eagle3 models (both Draft and Target) MUST have eagle_tree_mask
    auto tree_mask_it = in_ports.find(Eagle3LayerNames::eagle_tree_mask);
    OPENVINO_ASSERT(tree_mask_it != in_ports.end(), "Eagle3 model must have eagle_tree_mask input port");
    OPENVINO_ASSERT(m_eagle_tree_mask, "Eagle3 model requires eagle_tree_mask tensor to be provided by user");

    auto padded_tree_mask = request->get_tensor(tree_mask_it->second);
    pad_tree_mask_input(m_eagle_tree_mask, padded_tree_mask);
    LOG_VERB("Eagle3 Chunk: Set eagle_tree_mask input tensor (full mask, not chunked)");

    // Draft models MUST have hidden_states for chunked prefill
    if (m_role != Eagle3ModelRole::Draft) {
        return;
    }

    auto hidden_states_it = in_ports.find(Eagle3LayerNames::hidden_states);
    OPENVINO_ASSERT(hidden_states_it != in_ports.end(), "Eagle3 Draft model must have hidden_states input port");
    OPENVINO_ASSERT(m_hidden_states, "Eagle3 Draft model requires hidden_states tensor to be provided by user");
    OPENVINO_ASSERT(chunk_token_count > 0, "Chunk token count must be greater than 0");

    auto padded_tensor = request->get_tensor(hidden_states_it->second);

    // Get original tensor shape: [batch, seq_len, embedding_size]
    const auto& original_shape = m_hidden_states->get_shape();
    OPENVINO_ASSERT(original_shape.size() == 3,
                    "hidden_states tensor must have 3 dimensions: [batch, seq_len, embedding_size]");

    constexpr uint32_t seq_dim = 1;
    uint32_t total_tokens = static_cast<uint32_t>(original_shape[seq_dim]);

    // Validate chunk boundaries
    OPENVINO_ASSERT(chunk_start_token < total_tokens,
                    "chunk_start_token (" + std::to_string(chunk_start_token) + ") must be less than total_tokens (" +
                        std::to_string(total_tokens) + ")");

    uint32_t chunk_end_token = std::min(chunk_start_token + chunk_token_count, total_tokens);

    // Create tensor slice for current chunk along the sequence dimension
    auto chunk_tensor = util::make_tensor_slice(m_hidden_states, seq_dim, chunk_start_token, chunk_end_token);

    pad_hidden_state_input(chunk_tensor, padded_tensor);
    LOG_VERB("Eagle3 Draft: Set hidden_states chunk [" << chunk_start_token << ":" << chunk_end_token
                                                       << "] for chunk processing");
}

void Eagle3Extension::adjust_kvcache_before_infer(
    std::shared_ptr<ov::IAsyncInferRequest> request,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports,
    uint32_t& num_stored_tokens,
    bool v_transposed,
    uint32_t kv_dim) {
    LOG_DEBUG("Eagle3: Adjusting KV cache based on sampling result");
    LOG_BLOCK();

    auto& result = m_pending_sampling_result;

    // Check if there's a valid result to process
    if (result.num_total_generated == 0 || result.accepted_token_mask.empty()) {
        LOG_DEBUG("Eagle3: No pending sampling result, skipping KV cache adjustment");
        return;
    }

    // Validate the sampling result
    OPENVINO_ASSERT(result.accepted_token_mask.size() == result.num_total_generated,
                    "Eagle3: accepted_token_mask size must equal num_total_generated");

    // Validate num_accepted_tokens matches the mask
    uint32_t actual_accepted = 0;
    for (bool accepted : result.accepted_token_mask) {
        if (accepted)
            actual_accepted++;
    }
    OPENVINO_ASSERT(actual_accepted == result.num_accepted_tokens,
                    "Eagle3: num_accepted_tokens (" + std::to_string(result.num_accepted_tokens) +
                        ") must equal the number of true values in accepted_token_mask (" +
                        std::to_string(actual_accepted) + ")");

    // Calculate tokens to discard
    uint32_t tokens_to_discard = result.num_total_generated - result.num_accepted_tokens;

    if (tokens_to_discard == 0) {
        LOG_DEBUG("Eagle3: All generated tokens were accepted, no KV cache adjustment needed");
        // Clear the result
        result = SamplingResult();
        return;
    }

    LOG_DEBUG("Eagle3: Discarding " << tokens_to_discard << " tokens, keeping " << result.num_accepted_tokens
                                    << " tokens");

    // Fast path: If accepted tokens are contiguous from the start, simply rollback.
    // num_stored_tokens already equals the correct target value (gen_start_pos + num_accepted_tokens),
    // because it is derived from the first input position id of this round which is organized
    // by the pipeline after the previous validation. No arithmetic needed here.
    if (is_contiguous_acceptance()) {
        LOG_DEBUG("Eagle3: Accepted tokens are contiguous, using fast rollback (num_stored_tokens already correct)");
        // num_stored_tokens is already correct, nothing to do
    } else {
        // Complex path: Rearrange KV cache to keep only accepted tokens
        LOG_DEBUG("Eagle3: Accepted tokens are non-contiguous, rearranging KV cache");
        trim_kvcache_by_sampling(request, in_ports, out_ports, num_stored_tokens, v_transposed, kv_dim);
    }

    // Clear the processed sampling result
    result = SamplingResult();
    LOG_DEBUG("Eagle3: KV cache adjustment complete, new num_stored_tokens: " << num_stored_tokens);
}

void Eagle3Extension::trim_kvcache_by_sampling(
    std::shared_ptr<ov::IAsyncInferRequest> request,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports,
    uint32_t& num_stored_tokens,
    bool v_transposed,
    uint32_t kv_dim) {
    namespace uu = ov::npuw::util;

    auto& result = m_pending_sampling_result;
    auto& mask = result.accepted_token_mask;

    // Get the compiled model to iterate through outputs
    auto& compiled = request->get_compiled_model();

    // The starting position of the generated tokens in the KV cache.
    //
    // num_stored_tokens is the first input position id of this round, which equals
    // (gen_start_pos + num_accepted_tokens) because the pipeline organizes position ids
    // based on the previous validation result.
    //
    // Therefore:
    //   gen_start_pos = num_stored_tokens - num_accepted_tokens
    //
    // Example: prompt=14, draft generated 8 tokens, 2 accepted
    //   Physical KV cache length = 14 + 8 = 22
    //   num_stored_tokens        = 16  (= first position id, set by pipeline)
    //   gen_start_pos            = 16 - 2 = 14  ✓
    //
    // The old (incorrect) formula was: gen_start_pos = num_stored_tokens - num_total_generated
    //   which would give 16 - 8 = 8  ✗
    uint32_t gen_start_pos = num_stored_tokens - result.num_accepted_tokens;

    LOG_VERB("Eagle3: Rearranging KV cache, gen_start_pos=" << gen_start_pos
                                                            << ", num_stored_tokens=" << num_stored_tokens);

    // Build contiguous segments for batch copying
    // Each segment represents a continuous block of accepted tokens: [start_idx, length]
    struct Segment {
        uint32_t start_idx;  // Start index in mask
        uint32_t length;     // Length of continuous accepted tokens
    };
    std::vector<Segment> segments;

    uint32_t idx = 0;
    while (idx < mask.size()) {
        if (mask[idx]) {
            // Found start of an accepted segment
            uint32_t start = idx;
            uint32_t length = 0;
            while (idx < mask.size() && mask[idx]) {
                length++;
                idx++;
            }
            segments.push_back({start, length});
        } else {
            idx++;
        }
    }

    LOG_VERB("Eagle3: Found " << segments.size() << " contiguous segments to copy");

    // Iterate through all KV cache layers
    for (std::size_t i = 1; i < compiled->outputs().size(); ++i) {  // Start from 1 to skip logits
        const auto& output_name = compiled->outputs()[i].get_any_name();

        // Convert present layer name to past layer name
        const auto& input_name = std::regex_replace(output_name, std::regex("present"), "past_key_values");

        if (in_ports.find(input_name) == in_ports.end()) {
            continue;  // Skip if this is not a KV cache layer
        }

        auto past_kv_tensor = request->get_tensor(in_ports.at(input_name));

        // Determine the dimension for KV cache (value layers might be transposed)
        const auto& current_kv_dim = (output_name.find("value") != std::string::npos && v_transposed) ? 3u : kv_dim;

        LOG_VERB("Eagle3: Processing layer " << input_name << " on dimension " << current_kv_dim);

        // Copy each contiguous segment in batch
        uint32_t write_pos = gen_start_pos;
        for (const auto& seg : segments) {
            uint32_t read_start = gen_start_pos + seg.start_idx;
            uint32_t read_end = read_start + seg.length;

            if (write_pos != read_start) {
                // Create slices for the entire segment
                auto src_slice = uu::make_tensor_slice(past_kv_tensor, current_kv_dim, read_start, read_end);
                auto dst_slice =
                    uu::make_tensor_slice(past_kv_tensor, current_kv_dim, write_pos, write_pos + seg.length);

                // Copy the entire segment at once
                uu::copy_tensor_by_dim(src_slice, dst_slice, current_kv_dim, current_kv_dim);
                LOG_VERB("Eagle3: Copied segment [" << read_start << ":" << read_end << "] to [" << write_pos << ":"
                                                    << (write_pos + seg.length) << "]");
            } else {
                LOG_VERB("Eagle3: Segment [" << read_start << ":" << read_end
                                             << "] already in correct position, skipping");
            }

            write_pos += seg.length;
        }
    }

    // num_stored_tokens is already the correct target value (gen_start_pos + num_accepted_tokens),
    // so no update is needed here.
    LOG_VERB("Eagle3: KV cache rearrangement complete, num_stored_tokens=" << num_stored_tokens
                                                                           << " (unchanged, already correct)");
}

bool Eagle3Extension::is_contiguous_acceptance() const {
    auto& mask = m_pending_sampling_result.accepted_token_mask;

    if (mask.empty()) {
        return true;
    }

    // Check if all true values are at the beginning, followed by all false values
    bool found_false = false;
    for (bool accepted : mask) {
        if (!accepted) {
            found_false = true;
        } else if (found_false) {
            // Found a true after a false, so not contiguous
            return false;
        }
    }

    return true;
}

bool Eagle3Extension::process_sampling_result_from_state(
    std::shared_ptr<ov::IAsyncInferRequest> request,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports,
    uint32_t& num_stored_tokens,
    bool v_transposed,
    uint32_t kv_dim) {
    if (!m_sampling_state) {
        return false;
    }

    std::vector<bool> mask;
    uint32_t num_total = 0, num_accepted = 0;

    if (!m_sampling_state->extract_sampling_result(mask, num_total, num_accepted)) {
        return false;
    }

    LOG_DEBUG("Eagle3: Retrieved sampling result from VariableState: " << num_accepted << "/" << num_total
                                                                       << " tokens accepted");

    // Set the sampling result
    SamplingResult result;
    result.accepted_token_mask = std::move(mask);
    result.num_total_generated = num_total;
    result.num_accepted_tokens = num_accepted;
    set_sampling_result(result);

    // Adjust KV cache based on the sampling result
    adjust_kvcache_before_infer(request, in_ports, out_ports, num_stored_tokens, v_transposed, kv_dim);

    return true;
}

}  // namespace npuw
}  // namespace ov