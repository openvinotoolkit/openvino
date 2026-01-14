// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace npuw {

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
    Eagle3SamplingState() : m_name("npuw_eagle3_sampling_result") {
        // Create a tensor to hold sampling result
        // Format: [num_total_generated, num_accepted_tokens, mask[0], mask[1], ...]
        // Max size: 2 + max_tokens (assume max 512 tokens for speculative decoding)
        constexpr size_t max_capacity = 2 + 512;
        m_state_tensor = ov::Tensor(ov::element::i64, ov::Shape{max_capacity});
        reset();
    }

    void reset() override {
        // Clear the state
        std::fill_n(m_state_tensor.data<int64_t>(), m_state_tensor.get_size(), 0);
    }

    std::string get_name() const override {
        return m_name;
    }

    ov::Tensor get_state() const override {
        return m_state_tensor;
    }

    void set_state(const ov::Tensor& state) override {
        OPENVINO_ASSERT(state.get_element_type() == ov::element::i64,
                        "Eagle3SamplingState expects int64 tensor");
        OPENVINO_ASSERT(state.get_size() >= 2, "Eagle3SamplingState tensor must have at least 2 elements");
        
        // Copy the state
        if (state.get_size() <= m_state_tensor.get_size()) {
            std::copy_n(state.data<int64_t>(), state.get_size(), m_state_tensor.data<int64_t>());
        } else {
            // Need to resize
            m_state_tensor = ov::Tensor(ov::element::i64, state.get_shape());
            std::copy_n(state.data<int64_t>(), state.get_size(), m_state_tensor.data<int64_t>());
        }
    }

    // Check if there's valid sampling result to process
    bool has_result() const {
        auto* data = m_state_tensor.data<int64_t>();
        return data[0] > 0;  // num_total_generated > 0
    }

    // Extract sampling result and clear the state
    bool extract_sampling_result(std::vector<bool>& mask, uint32_t& num_total, uint32_t& num_accepted) {
        if (!has_result()) {
            return false;
        }

        auto* data = m_state_tensor.data<int64_t>();
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

private:
    std::string m_name;
    ov::Tensor m_state_tensor;
};

}  // namespace npuw
}  // namespace ov
