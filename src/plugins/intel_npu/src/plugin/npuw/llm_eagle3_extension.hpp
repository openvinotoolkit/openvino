// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <optional>
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

// Special VariableState for Eagle3 sampling result communication
// This allows external pipelines to pass sampling results through standard OpenVINO API
//
// Usage in pipeline:
//   auto states = infer_request.query_state();
//   for (auto& state : states) {
//       if (state->get_name() == "npuw_eagle3_sampling_result") {
//           auto tensor = state->get_state();
//           auto* data = tensor.data<int64_t>();
//           data[0] = num_total_generated;
//           data[1] = num_accepted_tokens;
//           for (size_t i = 0; i < mask.size(); ++i) {
//               data[2 + i] = mask[i] ? 1 : 0;
//           }
//           state->set_state(tensor);
//           break;
//       }
//   }
class Eagle3SamplingState : public ov::IVariableState {
public:
    Eagle3SamplingState() : ov::IVariableState("npuw_eagle3_sampling_result") {
        // Create a tensor to hold sampling result
        // Format: [num_total_generated, num_accepted_tokens, mask[0], mask[1], ...]
        // Max size: 2 + max_tokens (assume max 512 tokens for speculative decoding)
        constexpr size_t max_capacity = 2 + 512;
        auto tensor = ov::Tensor(ov::element::i64, ov::Shape{max_capacity});
        m_state = ov::get_tensor_impl(tensor);
        reset();
    }

    void reset() override {
        // Clear the state
        std::fill_n(m_state->data<int64_t>(), m_state->get_size(), 0);
    }

    void set_state(const ov::SoPtr<ov::ITensor>& state) override {
        OPENVINO_ASSERT(state->get_element_type() == ov::element::i64, "Eagle3SamplingState expects int64 tensor");
        OPENVINO_ASSERT(state->get_size() >= 2, "Eagle3SamplingState tensor must have at least 2 elements");

        // Copy the state
        if (state->get_size() <= m_state->get_size()) {
            std::copy_n(state->data<int64_t>(), state->get_size(), m_state->data<int64_t>());
        } else {
            // Need to resize
            auto tensor = ov::Tensor(ov::element::i64, state->get_shape());
            m_state = ov::get_tensor_impl(tensor);
            std::copy_n(state->data<int64_t>(), state->get_size(), m_state->data<int64_t>());
        }
    }

    // Check if there's valid sampling result to process
    bool has_result() const {
        auto* data = m_state->data<int64_t>();
        return data[0] > 0;  // num_total_generated > 0
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
            mask.push_back(data[2 + i] != 0);
        }

        // Clear after extraction
        reset();
        return true;
    }
};

// Extension for Eagle3 speculative decoding
// Handles Eagle3-specific input/output logic for draft and target models
class Eagle3Extension {
public:
    // Get static shape for Eagle3 input tensors
    static ov::PartialShape get_static_input(const std::shared_ptr<ov::Model>& model,
                                             const ov::Output<ov::Node>& input,
                                             uint32_t input_size);

    // Detect Eagle3 model role (Draft/Target/None) based on is_eagle flag and inputs/outputs
    void initialize(bool is_eagle_model,
                    const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
                    const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports);

    // Returns true if model is Eagle3 draft or target
    bool is_eagle3_model() const {
        return m_role != Eagle3ModelRole::None;
    }

    Eagle3ModelRole get_role() const {
        return m_role;
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

    ov::SoPtr<ov::ITensor> get_last_hidden_state() const {
        return m_last_hidden_state;
    }

    ov::SoPtr<ov::ITensor> get_eagle_tree_mask() const {
        return m_eagle_tree_mask;
    }

    // Get Eagle3 sampling state for query_state()
    std::shared_ptr<Eagle3SamplingState> get_sampling_state() const {
        return m_sampling_state;
    }

    // Process sampling result from VariableState (called in infer_generate)
    // Returns true if sampling result was found and processed
    bool process_sampling_result_from_state(
        std::shared_ptr<ov::IAsyncInferRequest> request,
        const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
        const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports,
        uint32_t& num_stored_tokens,
        bool v_transposed,
        uint32_t kv_dim);

    // Sampling result information for KV cache adjustment
    // Usage example:
    //   SamplingResult result;
    //   result.accepted_token_mask = {true, true, false, true, false};  // Accept tokens 0,1,3
    //   result.num_total_generated = 5;
    //   result.num_accepted_tokens = 3;
    //   eagle3_ext.set_sampling_result(result);
    struct SamplingResult {
        std::vector<bool> accepted_token_mask;  ///< Mask indicating which tokens are accepted (true) or rejected (false)
        uint32_t num_total_generated = 0;       ///< Total number of tokens generated (should equal accepted_token_mask.size())
        uint32_t num_accepted_tokens = 0;       ///< Number of accepted tokens (should equal count of true in accepted_token_mask)
    };

    // Set sampling result from pipeline (to be processed in next infer)
    void set_sampling_result(const SamplingResult& result) {
        m_pending_sampling_result = result;
    }

    // Adjust KV cache based on sampling result before inference
    // Should be called at the beginning of infer_generate
    void adjust_kvcache_before_infer(std::shared_ptr<ov::IAsyncInferRequest> request,
                                     const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
                                     const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports,
                                     uint32_t& num_stored_tokens,
                                     bool v_transposed,
                                     uint32_t kv_dim);

private:
    void validate_hidden_state_tensor(const ov::SoPtr<ov::ITensor>& tensor, const std::string& name);
    void validate_tree_mask_tensor(const ov::SoPtr<ov::ITensor>& tensor, const std::string& name);

    // Trim KV cache by rearranging only accepted tokens
    void trim_kvcache_by_sampling(std::shared_ptr<ov::IAsyncInferRequest> request,
                                  const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
                                  const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports,
                                  uint32_t& num_stored_tokens,
                                  bool v_transposed,
                                  uint32_t kv_dim);

    // Check if accepted tokens are contiguous from the start (e.g., [true, true, true, false, false])
    bool is_contiguous_acceptance() const;

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
