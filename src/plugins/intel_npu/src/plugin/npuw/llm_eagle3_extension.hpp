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

#include "openvino/core/descriptor/output.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/itensor.hpp"

namespace ov {
namespace npuw {

/// Eagle3 layer names used in speculative decoding
struct Eagle3LayerNames {
    static constexpr const char* hidden_states = "hidden_states";
    static constexpr const char* internal_hidden_states = "internal_hidden_states";
    static constexpr const char* last_hidden_state = "last_hidden_state";
};

/// Eagle3 string matching utilities
bool matchEagle3HiddenStatesString(const std::string& input);
bool matchEagle3InternalHiddenStatesString(const std::string& input);

/// Eagle3 model roles in speculative decoding pipeline
enum class Eagle3ModelRole {
    None,    ///< Not an Eagle3 model
    Target,  ///< Target model: only outputs last_hidden_state
    Draft    ///< Draft model: has hidden_states/internal_hidden_states as inputs, and outputs last_hidden_state
};

/**
 * @brief Eagle3 Extension for speculative decoding support
 *
 * Encapsulates all Eagle3-specific logic for both target and draft models.
 * Target models only produce last_hidden_state output.
 * Draft models consume hidden_states and internal_hidden_states inputs, and also produce last_hidden_state output.
 */
class Eagle3Extension {
public:
    /// Initialize Eagle3 support and detect model role (Draft/Target/None) based on model inputs/outputs
    void initialize(const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
                    const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports);

    /// Returns true if this model is an Eagle3 draft or target model
    bool is_enabled() const {
        return m_role != Eagle3ModelRole::None;
    }

    /// Get the detected Eagle3 model role (None, Draft, Target)
    Eagle3ModelRole get_role() const {
        return m_role;
    }

    /// Store user-provided Eagle3 input tensors (only for draft models)
    template <typename GetTensorFunc>
    bool store_user_tensors(const std::vector<ov::Output<const ov::Node>>& inputs, GetTensorFunc get_tensor_func) {
        if (m_role != Eagle3ModelRole::Draft) {
            return false;
        }

        bool processed_any = false;

        // Process internal_hidden_states input
        auto internal_hidden_port = find_port_by_name(inputs, Eagle3LayerNames::internal_hidden_states);
        if (internal_hidden_port.has_value()) {
            auto tensor = get_tensor_func(internal_hidden_port.value());
            validate_tensor(tensor, "internal_hidden_states");
            m_internal_hidden_states = tensor;
            processed_any = true;
        }

        // Process hidden_states input
        auto hidden_port = find_port_by_name(inputs, Eagle3LayerNames::hidden_states);
        if (hidden_port.has_value()) {
            auto tensor = get_tensor_func(hidden_port.value());
            validate_tensor(tensor, "hidden_states");
            m_hidden_states = tensor;
            processed_any = true;
        }

        return processed_any;
    }

    /// Prepare Eagle3 inputs automatically based on model role (draft only)
    void prepare_inputs(std::shared_ptr<ov::IAsyncInferRequest> request,
                        const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports);

    /// Prepare Eagle3 inputs for chunked prefill (draft only), using a specific token range
    void prepare_inputs_for_chunk(std::shared_ptr<ov::IAsyncInferRequest> request,
                                  const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
                                  uint32_t chunk_start_token,
                                  uint32_t chunk_token_count);

    /// Process Eagle3 outputs automatically based on model role (both draft and target)
    void process_outputs(std::shared_ptr<ov::IAsyncInferRequest> request,
                         const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports);

    /// Get stored Eagle3 input/output tensors
    ov::SoPtr<ov::ITensor> get_hidden_states() const {
        return m_hidden_states;
    }

    ov::SoPtr<ov::ITensor> get_internal_hidden_states() const {
        return m_internal_hidden_states;
    }

    ov::SoPtr<ov::ITensor> get_last_hidden_state() const {
        return m_last_hidden_state;
    }

private:
    // Utility: find port by name
    static std::optional<ov::Output<const ov::Node>> find_port_by_name(
        const std::vector<ov::Output<const ov::Node>>& ports,
        const std::string& name) {
        auto it = std::find_if(ports.begin(), ports.end(), [&](const auto& port) {
            return port.get_names().count(name) != 0;
        });
        return (it != ports.end()) ? std::make_optional(*it) : std::nullopt;
    }

    // Validate tensor properties (type and shape)
    void validate_tensor(const ov::SoPtr<ov::ITensor>& tensor, const std::string& name);

    // Internal methods for role-specific processing
    void prepare_inputs_impl(std::shared_ptr<ov::IAsyncInferRequest> request,
                             const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports);
    void process_outputs_impl(std::shared_ptr<ov::IAsyncInferRequest> request,
                              const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports);

    Eagle3ModelRole m_role = Eagle3ModelRole::None;

    ov::SoPtr<ov::ITensor> m_hidden_states;           ///< Draft model input: hidden_states
    ov::SoPtr<ov::ITensor> m_internal_hidden_states;  ///< Draft model input: internal_hidden_states
    ov::SoPtr<ov::ITensor> m_last_hidden_state;       ///< Draft/Target model output: last_hidden_state
};

}  // namespace npuw
}  // namespace ov
