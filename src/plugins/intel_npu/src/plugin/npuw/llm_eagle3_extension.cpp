// Copyright (C) 2018-2026 Intel Corporation
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

bool matchEagle3TreeMaskString(const std::string& input) {
    return input == Eagle3LayerNames::eagle_tree_mask;
}

ov::PartialShape Eagle3Extension::get_static_input(const std::shared_ptr<ov::Model>& model,
                                                   const ov::Output<ov::Node>& input,
                                                   uint32_t input_size,
                                                   uint32_t kvcache_size,
                                                   bool is_prefill) {
    const auto& input_name = input.get_any_name();
    const auto& input_shape = input.get_partial_shape();

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

    if (matchEagle3TreeMaskString(input_name)) {
        // During prefill (including chunked prefill) the tree mask degenerates to a single 1×1 entry.
        // During generation the mask covers all past KV positions for each token.
        if (is_prefill) {
            return ov::PartialShape({1, 1, 1, 1});
        }
        return ov::PartialShape({1, 1, input_size, kvcache_size});
    }

    OPENVINO_THROW("Eagle3Extension::get_static_input called with unexpected input name '",
                   input_name,
                   "'. Expected '",
                   Eagle3LayerNames::hidden_states,
                   "' or '",
                   Eagle3LayerNames::eagle_tree_mask,
                   "'.");
}

void Eagle3Extension::store_user_inputs(const ov::IInferRequest& request,
                                        const std::vector<ov::Output<const ov::Node>>& inputs) {
    // Store eagle_tree_mask for both Draft and Target models
    auto tree_mask_port = util::find_port_by_name(inputs, Eagle3LayerNames::eagle_tree_mask);
    if (tree_mask_port.has_value()) {
        auto tensor = request.get_tensor(tree_mask_port.value());
        OPENVINO_ASSERT(
            tensor->get_element_type() == ov::element::f32 || tensor->get_element_type() == ov::element::f16,
            "eagle_tree_mask input must be float32 or float16");
        OPENVINO_ASSERT(tensor->get_shape().size() == 4,
                        "eagle_tree_mask must have 4 dimensions: [batch, 1, input_size, kvcache_size]");
        OPENVINO_ASSERT(tensor->get_shape()[0] == 1, "eagle_tree_mask batch dimension must be 1");
        OPENVINO_ASSERT(tensor->get_shape()[1] == 1, "eagle_tree_mask second dimension must be 1");
        m_eagle_tree_mask = tensor;
    }

    // Store hidden_states for Draft model only
    auto hidden_states_port = util::find_port_by_name(inputs, Eagle3LayerNames::hidden_states);
    if (hidden_states_port.has_value()) {
        auto tensor = request.get_tensor(hidden_states_port.value());
        OPENVINO_ASSERT(
            tensor->get_element_type() == ov::element::f32 || tensor->get_element_type() == ov::element::f16,
            "hidden_states input must be float32 or float16");
        OPENVINO_ASSERT(tensor->get_shape().size() == 3,
                        "hidden_states must have 3 dimensions: [batch, token_length, embedding_size]");
        m_hidden_states = tensor;
    }
}

}  // namespace npuw
}  // namespace ov

namespace {

// Represents a contiguous run of accepted tokens within the speculative decoding mask
struct KVCacheSegment {
    uint32_t start_idx;  ///< Index in the acceptance mask where this run begins
    uint32_t length;     ///< Number of consecutive accepted tokens
};

void pad_hidden_state_input(const ov::SoPtr<ov::ITensor>& hidden_state,
                            const ov::SoPtr<ov::ITensor>& padded_hidden_state) {
    // Pad the token_length dimension of hidden state input [batch, token_length, embedding_size].
    // Caller guarantees both tensors are rank-3 with batch=1 (validated in store_user_inputs).
    constexpr size_t kTokenDim = 1;
    constexpr size_t kEmbedDim = 2;

    const auto& src_shape = hidden_state->get_shape();
    const auto& dst_shape = padded_hidden_state->get_shape();

    OPENVINO_ASSERT(dst_shape[kEmbedDim] == src_shape[kEmbedDim], "Embedding size must match");
    OPENVINO_ASSERT(dst_shape[kTokenDim] >= src_shape[kTokenDim], "Padded token length must be >= input token length");

    ov::npuw::util::fill_tensor_bytes(padded_hidden_state, 0u);

    // Tokens are right-aligned: copy src flush to the right end of dst.
    ov::npuw::util::copy_to_right(hidden_state, padded_hidden_state);
}

void pad_tree_mask_input(const ov::SoPtr<ov::ITensor>& tree_mask, const ov::SoPtr<ov::ITensor>& padded_tree_mask) {
    // Pad the tree mask tensor [batch, 1, input_size, kvcache_size].
    // The user-provided mask is placed in the bottom-right corner of the padded tensor;
    // the remaining entries are zero-filled.
    // This handles both prefill (padded shape {1,1,1,1}) and generate phases uniformly.
    // Caller guarantees both tensors are rank-4 with batch=1 and heads=1 (validated in store_user_inputs).
    constexpr size_t kInputSizeDim = 2;
    constexpr size_t kKVSizeDim = 3;

    const auto& src_shape = tree_mask->get_shape();
    const auto& dst_shape = padded_tree_mask->get_shape();

    OPENVINO_ASSERT(dst_shape[kInputSizeDim] >= src_shape[kInputSizeDim],
                    "Padded input_size must be >= original input_size");
    OPENVINO_ASSERT(dst_shape[kKVSizeDim] >= src_shape[kKVSizeDim],
                    "Padded kvcache_size must be >= original kvcache_size");

    const size_t input_size = src_shape[kInputSizeDim];
    const size_t padded_input_size = dst_shape[kInputSizeDim];
    const size_t kvcache_size = src_shape[kKVSizeDim];
    const size_t padded_kvcache_size = dst_shape[kKVSizeDim];
    const size_t elem_size = tree_mask->get_element_type().size();

    const uint8_t* src_data = reinterpret_cast<const uint8_t*>(tree_mask->data());
    uint8_t* dst_data = reinterpret_cast<uint8_t*>(padded_tree_mask->data());

    std::memset(dst_data, 0, padded_tree_mask->get_byte_size());

    // Place source rows in the bottom-right corner: left-pad rows by row_padding,
    // left-pad columns by col_padding. This matches the causal attention convention
    // where new query tokens attend to all prior KV positions (top-left is oldest context).
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
    m_sampling_state = std::make_shared<Eagle3SamplingState>();
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
    auto last_hidden_state_it = out_ports.find(Eagle3LayerNames::last_hidden_state);
    OPENVINO_ASSERT(last_hidden_state_it != out_ports.end(), "Eagle3 model must have last_hidden_state output port");

    auto chunk_output = request->get_tensor(last_hidden_state_it->second);
    const auto& chunk_shape = chunk_output->get_shape();

    OPENVINO_ASSERT(chunk_shape.size() == 3, "last_hidden_state must have 3 dimensions: [batch, seq_len, hidden_size]");

    constexpr size_t kBatchDim = 0;
    constexpr size_t kSeqDim = 1;
    constexpr size_t kHiddenDim = 2;

    const uint32_t batch_size = static_cast<uint32_t>(chunk_shape[kBatchDim]);
    const uint32_t chunk_seq_len = static_cast<uint32_t>(chunk_shape[kSeqDim]);
    const uint32_t hidden_size = static_cast<uint32_t>(chunk_shape[kHiddenDim]);
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
    uint32_t& num_stored_tokens,
    bool v_transposed,
    uint32_t kv_dim) {
    LOG_DEBUG("Eagle3: Adjusting KV cache based on sampling result");
    LOG_BLOCK();

    auto& result = m_pending_sampling_result;

    if (result.num_total_generated == 0 || result.accepted_token_mask.empty()) {
        LOG_DEBUG("Eagle3: No pending sampling result, skipping KV cache adjustment");
        return;
    }

    OPENVINO_ASSERT(result.accepted_token_mask.size() == result.num_total_generated,
                    "Eagle3: accepted_token_mask size must equal num_total_generated");

    // Verify the header fields are consistent with the mask contents.
    const uint32_t actual_accepted =
        static_cast<uint32_t>(std::count(result.accepted_token_mask.begin(), result.accepted_token_mask.end(), true));
    OPENVINO_ASSERT(actual_accepted == result.num_accepted_tokens,
                    "Eagle3: num_accepted_tokens (" + std::to_string(result.num_accepted_tokens) +
                        ") must equal the number of true values in accepted_token_mask (" +
                        std::to_string(actual_accepted) + ")");

    uint32_t tokens_to_discard = result.num_total_generated - result.num_accepted_tokens;

    if (tokens_to_discard == 0) {
        LOG_DEBUG("Eagle3: All generated tokens were accepted, no KV cache adjustment needed");
        result = SamplingResult();
        return;
    }

    LOG_DEBUG("Eagle3: Discarding " << tokens_to_discard << " tokens, keeping " << result.num_accepted_tokens
                                    << " tokens");

    // Fast path: accepted tokens are contiguous from the start — num_stored_tokens already
    // equals (gen_start_pos + num_accepted_tokens) as set by the pipeline, so no KV
    // rearrangement is needed; just clear the result and return.
    // Complex path: non-contiguous — physically move accepted token entries together.
    const bool contiguous =
        std::is_partitioned(result.accepted_token_mask.begin(), result.accepted_token_mask.end(), [](bool v) {
            return v;
        });
    if (!contiguous) {
        LOG_DEBUG("Eagle3: Accepted tokens are non-contiguous, rearranging KV cache");
        trim_kvcache_by_sampling(request, in_ports, num_stored_tokens, v_transposed, kv_dim);
    } else {
        LOG_DEBUG("Eagle3: Accepted tokens are contiguous, using fast rollback (num_stored_tokens already correct)");
    }

    // Clear the processed sampling result
    result = SamplingResult();
    LOG_DEBUG("Eagle3: KV cache adjustment complete, new num_stored_tokens: " << num_stored_tokens);
}

void Eagle3Extension::trim_kvcache_by_sampling(
    std::shared_ptr<ov::IAsyncInferRequest> request,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
    uint32_t& num_stored_tokens,
    bool v_transposed,
    uint32_t kv_dim) {
    namespace uu = ov::npuw::util;

    auto& result = m_pending_sampling_result;
    auto& mask = result.accepted_token_mask;

    auto& compiled = request->get_compiled_model();

    // num_stored_tokens is the first input position id for this round, set by the pipeline
    // to (prompt_len + num_accepted_tokens). Therefore the position where draft tokens begin
    // in the KV cache is:
    //   gen_start_pos = num_stored_tokens - num_accepted_tokens
    //
    // Example: prompt=14 tokens, 8 drafted, 2 accepted
    //   Physical KV cache length = 14 + 8 = 22
    //   num_stored_tokens        = 14 + 2 = 16  (set by pipeline)
    //   gen_start_pos            = 16 - 2  = 14
    uint32_t gen_start_pos = num_stored_tokens - result.num_accepted_tokens;

    LOG_VERB("Eagle3: Rearranging KV cache, gen_start_pos=" << gen_start_pos
                                                            << ", num_stored_tokens=" << num_stored_tokens);

    // Build contiguous runs of accepted tokens (segments) for batch copying.
    // Each segment maps directly to a slice copy within each KV cache layer.
    std::vector<KVCacheSegment> segments;
    for (uint32_t idx = 0; idx < static_cast<uint32_t>(mask.size()); ++idx) {
        if (!mask[idx]) {
            continue;
        }
        uint32_t start = idx;
        while (idx < static_cast<uint32_t>(mask.size()) && mask[idx]) {
            ++idx;
        }
        segments.push_back({start, idx - start});
    }

    LOG_VERB("Eagle3: Found " << segments.size() << " contiguous segments to copy");

    // Iterate through all KV cache outputs.
    // Each "present.<layer>" output corresponds to a "past_key_values.<layer>" input.
    // Non-KV outputs (e.g. logits) are filtered out by the prefix check below.
    static const std::string present_prefix = "present";
    for (const auto& output : compiled->outputs()) {
        const auto& output_name = output.get_any_name();

        // Convert present layer name to past layer name:
        // "present.<layer>" -> "past_key_values.<layer>"
        if (output_name.compare(0, present_prefix.size(), present_prefix) != 0) {
            continue;  // Not a present KV layer
        }
        const std::string input_name = "past_key_values" + output_name.substr(present_prefix.size());

        auto in_port_it = in_ports.find(input_name);
        if (in_port_it == in_ports.end()) {
            continue;  // Skip if this is not a KV cache layer
        }

        auto past_kv_tensor = request->get_tensor(in_port_it->second);

        // Determine the dimension for KV cache (value layers might be transposed)
        const auto& current_kv_dim = (output_name.find("value") != std::string::npos && v_transposed) ? 3u : kv_dim;

        LOG_VERB("Eagle3: Processing layer " << input_name << " on dimension " << current_kv_dim);

        // Write accepted segments contiguously starting at gen_start_pos,
        // compacting the KV cache in-place to remove rejected-token entries.
        uint32_t write_pos = gen_start_pos;
        for (const auto& seg : segments) {
            uint32_t read_start = gen_start_pos + seg.start_idx;
            uint32_t read_end = read_start + seg.length;

            if (write_pos != read_start) {
                auto src_slice = uu::make_tensor_slice(past_kv_tensor, current_kv_dim, read_start, read_end);
                auto dst_slice =
                    uu::make_tensor_slice(past_kv_tensor, current_kv_dim, write_pos, write_pos + seg.length);
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

    // num_stored_tokens was set by the pipeline to (gen_start_pos + num_accepted_tokens)
    // and is already the correct post-trim value — no update needed.
    LOG_VERB("Eagle3: KV cache rearrangement complete, num_stored_tokens=" << num_stored_tokens);
}

bool Eagle3Extension::process_sampling_result_from_state(
    std::shared_ptr<ov::IAsyncInferRequest> request,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
    uint32_t& num_stored_tokens,
    bool v_transposed,
    uint32_t kv_dim) {
    if (!m_sampling_state) {
        return false;
    }

    SamplingResult result;
    if (!m_sampling_state->extract_sampling_result(result.accepted_token_mask,
                                                   result.num_total_generated,
                                                   result.num_accepted_tokens)) {
        return false;
    }

    LOG_DEBUG("Eagle3: Retrieved sampling result from VariableState: "
              << result.num_accepted_tokens << "/" << result.num_total_generated << " tokens accepted");

    m_pending_sampling_result = std::move(result);
    adjust_kvcache_before_infer(request, in_ports, num_stored_tokens, v_transposed, kv_dim);
    return true;
}

}  // namespace npuw
}  // namespace ov
