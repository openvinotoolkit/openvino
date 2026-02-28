// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "infer_request_utils.hpp"
#include "openvino/core/descriptor/output.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace ov {
namespace npuw {

// Layer names for Eagle3 speculative decoding
struct Eagle3LayerNames {
    static constexpr const char* hidden_states = "hidden_states";
    static constexpr const char* last_hidden_state = "last_hidden_state";
    static constexpr const char* eagle_tree_mask = "eagle_tree_mask";
};

// Utility functions for Eagle3 layer name matching
bool matchEagle3HiddenStatesString(const std::string& input);
bool matchEagle3TreeMaskString(const std::string& input);

// Model roles for Eagle3 speculative decoding
enum class Eagle3ModelRole {
    None,    ///< Not an Eagle3 model
    Target,  ///< Target model: only outputs last_hidden_state
    Draft    ///< Draft model: has hidden_states as input, and outputs last_hidden_state
};

// Special VariableState for Eagle3 sampling result communication.
// Allows external pipelines to pass sampling results through the standard OpenVINO API.
//
// Tensor layout (element type: int64):
//   [0]               num_total_generated
//   [1]               num_accepted_tokens
//   [2 .. 2+N-1]      accepted_token_mask (1 = accepted, 0 = rejected), N = num_total_generated
//
// Usage example:
//   auto states = infer_request.query_state();
//   for (auto& state : states) {
//       if (state->get_name() == "npuw_eagle3_sampling_result") {
//           auto tensor = state->get_state();
//           auto* data = tensor.data<int64_t>();
//           data[0] = num_total_generated;
//           data[1] = num_accepted_tokens;
//           for (size_t i = 0; i < mask.size(); ++i)
//               data[Eagle3SamplingState::kHeaderSize + i] = mask[i] ? 1 : 0;
//           state->set_state(tensor);
//           break;
//       }
//   }
class Eagle3SamplingState : public ov::IVariableState {
public:
    // Maximum number of tokens in a single speculative decoding round.
    // Format: [num_total_generated, num_accepted_tokens, mask[0], mask[1], ...]
    static constexpr size_t kMaxSpecTokens = 512;
    static constexpr size_t kHeaderSize = 2;  // num_total_generated + num_accepted_tokens

    Eagle3SamplingState() : ov::IVariableState("npuw_eagle3_sampling_result") {
        auto tensor = ov::Tensor(ov::element::i64, ov::Shape{kHeaderSize + kMaxSpecTokens});
        m_state = ov::get_tensor_impl(tensor);
        reset();
    }

    void reset() override {
        std::fill_n(m_state->data<int64_t>(), m_state->get_size(), 0);
    }

    void set_state(const ov::SoPtr<ov::ITensor>& state) override {
        // Copy the caller-provided tensor into our fixed-size internal buffer.
        OPENVINO_ASSERT(state->get_element_type() == ov::element::i64, "Eagle3SamplingState expects int64 tensor");
        OPENVINO_ASSERT(state->get_size() >= kHeaderSize, "Eagle3SamplingState tensor must have at least 2 elements");
        OPENVINO_ASSERT(state->get_size() <= m_state->get_size(),
                        "Eagle3SamplingState: input tensor size (" + std::to_string(state->get_size()) +
                            ") exceeds maximum capacity (" + std::to_string(m_state->get_size()) +
                            "). Consider increasing kMaxSpecTokens.");
        std::copy_n(state->data<int64_t>(), state->get_size(), m_state->data<int64_t>());
    }

    // Extract sampling result and clear the state
    bool extract_sampling_result(std::vector<bool>& mask, uint32_t& num_total, uint32_t& num_accepted) {
        if (!has_result()) {
            return false;
        }

        auto* data = m_state->data<int64_t>();
        num_total = static_cast<uint32_t>(data[0]);
        num_accepted = static_cast<uint32_t>(data[1]);

        mask.clear();
        mask.reserve(num_total);
        for (uint32_t i = 0; i < num_total; ++i) {
            mask.push_back(data[kHeaderSize + i] != 0);
        }

        reset();
        return true;
    }

private:
    // Returns true if the state tensor contains a valid (non-empty) sampling result
    bool has_result() const {
        return m_state->data<int64_t>()[0] > 0;  // num_total_generated > 0
    }
};

// Extension for Eagle3 speculative decoding
// Handles Eagle3-specific input/output logic for draft and target models
class Eagle3Extension {
public:
    // Compute the static shape for Eagle3-specific inputs (hidden_states, eagle_tree_mask).
    // is_prefill drives eagle_tree_mask shape: prefill uses {1,1,1,1}
    // generate uses {1,1,input_size,kvcache_size}
    static ov::PartialShape get_static_input(const std::shared_ptr<ov::Model>& model,
                                             const ov::Output<ov::Node>& input,
                                             uint32_t input_size,
                                             uint32_t kvcache_size,
                                             bool is_prefill);

    // Detect Eagle3 model role (Draft/Target/None) based on is_eagle flag and inputs/outputs
    void initialize(bool is_eagle_model,
                    const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
                    const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports);

    // Returns true if model is Eagle3 draft or target
    bool is_eagle3_model() const {
        return m_role != Eagle3ModelRole::None;
    }

    // Store eagle3 specific inputs (eagle_tree_mask, hidden_states) from the inference request
    // Must be called before prepare_inputs/prepare_inputs_for_chunk
    void store_user_inputs(const ov::IInferRequest& request, const std::vector<ov::Output<const ov::Node>>& inputs);

    // Prepare Eagle3 new input tensors (hidden_states)
    void prepare_inputs(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                        const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports);

    // Prepare Eagle3 new input tensors for chunked prefill
    void prepare_inputs_for_chunk(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                                  const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
                                  uint32_t chunk_start_token,
                                  uint32_t chunk_token_count);

    // Retrieve and store last_hidden_state output tensor for draft and target models
    void update_last_hidden_state(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                                  const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports);

    // Accumulate last_hidden_state from current chunk during chunked prefill
    void accumulate_chunk_last_hidden_state(
        const std::shared_ptr<ov::IAsyncInferRequest>& request,
        const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports,
        uint32_t chunk_token_count,
        uint32_t total_seq_len);

    // Reset chunked prefill state before starting a new chunked prefill session
    // NOTE: m_last_hidden_state holds tensors of different sizes in prefill vs generation phases
    // Must reset to avoid size mismatch when starting a new prefill after previous generations
    void reset_chunked_prefill_state() {
        m_last_hidden_state = {};
        m_chunked_seq_offset = 0;
    }

    // Returns the last_hidden_state tensor from the most recent inference.
    // For chunked prefill, this is the fully-assembled tensor across all chunks.
    ov::SoPtr<ov::ITensor> get_last_hidden_state() const {
        return m_last_hidden_state;
    }

    // Returns the Eagle3SamplingState VariableState exposed via query_state().
    // External pipelines write sampling results into this state; the plugin reads
    // them back in process_sampling_result_from_state() before the next inference.
    std::shared_ptr<Eagle3SamplingState> get_sampling_state() const {
        return m_sampling_state;
    }

    // Read the sampling result from the Eagle3SamplingState VariableState and apply
    // any required KV cache adjustment (discard rejected draft tokens) before infer.
    bool process_sampling_result_from_state(std::shared_ptr<ov::IAsyncInferRequest> request,
                                            const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
                                            uint32_t& num_stored_tokens,
                                            bool v_transposed,
                                            uint32_t kv_dim);

private:
    // Internal representation of a speculative decoding sampling result
    struct SamplingResult {
        std::vector<bool> accepted_token_mask;  ///< true = token accepted, false = rejected
        uint32_t num_total_generated = 0;       ///< Total tokens generated (== accepted_token_mask.size())
        uint32_t num_accepted_tokens = 0;       ///< Count of accepted tokens (== count of true in mask)
    };

    // Adjust KV cache based on sampling result before inference
    void adjust_kvcache_before_infer(std::shared_ptr<ov::IAsyncInferRequest> request,
                                     const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
                                     uint32_t& num_stored_tokens,
                                     bool v_transposed,
                                     uint32_t kv_dim);

    // Trim KV cache by rearranging only accepted tokens
    void trim_kvcache_by_sampling(std::shared_ptr<ov::IAsyncInferRequest> request,
                                  const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
                                  uint32_t& num_stored_tokens,
                                  bool v_transposed,
                                  uint32_t kv_dim);

    Eagle3ModelRole m_role = Eagle3ModelRole::None;

    ov::SoPtr<ov::ITensor> m_hidden_states;      ///< Draft model input: hidden_states
    ov::SoPtr<ov::ITensor> m_eagle_tree_mask;    ///< Draft/Target model input: eagle_tree_mask
    ov::SoPtr<ov::ITensor> m_last_hidden_state;  ///< Draft/Target model output: last_hidden_state

    // For chunked prefill: track the write offset in the pre-allocated tensor
    uint32_t m_chunked_seq_offset = 0;

    SamplingResult m_pending_sampling_result;    ///< Pending sampling result from previous inference
    std::shared_ptr<Eagle3SamplingState> m_sampling_state;  ///< VariableState for external pipeline communication
};

}  // namespace npuw
}  // namespace ov
