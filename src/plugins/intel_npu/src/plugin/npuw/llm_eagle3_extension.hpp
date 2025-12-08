// Copyright (C) 2025 Intel Corporation
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

namespace ov {
namespace npuw {

// Layer names for Eagle3 speculative decoding
struct Eagle3LayerNames {
    static constexpr const char* hidden_states = "hidden_states";
    static constexpr const char* last_hidden_state = "last_hidden_state";
};

// Utility functions for Eagle3 layer name matching
bool matchEagle3HiddenStatesString(const std::string& input);

// Model roles for Eagle3 speculative decoding
enum class Eagle3ModelRole {
    None,    ///< Not an Eagle3 model
    Target,  ///< Target model: only outputs last_hidden_state
    Draft    ///< Draft model: has hidden_states as input, and outputs last_hidden_state
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

    // Store hidden state inputs from user request (must be called before prepare_inputs/prepare_inputs_for_chunk)
    void store_hidden_state_inputs(const ov::IInferRequest& request,
                                   const std::vector<ov::Output<const ov::Node>>& inputs);

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

    ov::SoPtr<ov::ITensor> get_hidden_states() const {
        return m_hidden_states;
    }

    ov::SoPtr<ov::ITensor> get_last_hidden_state() const {
        return m_last_hidden_state;
    }

private:
    void validate_hidden_state_tensor(const ov::SoPtr<ov::ITensor>& tensor, const std::string& name);

    Eagle3ModelRole m_role = Eagle3ModelRole::None;

    ov::SoPtr<ov::ITensor> m_hidden_states;      ///< Draft model input: hidden_states
    ov::SoPtr<ov::ITensor> m_last_hidden_state;  ///< Draft/Target model output: last_hidden_state
};

}  // namespace npuw
}  // namespace ov
