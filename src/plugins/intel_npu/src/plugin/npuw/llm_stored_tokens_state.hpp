// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace ov {
namespace npuw {

// Special VariableState for already processed/stored tokens in LLM.
// Allows external pipelines to call reset on that state where needed.
class StoredTokensState : public ov::IVariableState {
public:
    friend class ov::npuw::LLMInferRequest;
    StoredTokensState() : ov::IVariableState("npuw_stored_tokens_state") {
        auto tensor = ov::Tensor(ov::element::i64, ov::Shape{1});
        m_state = ov::get_tensor_impl(tensor);
        reset();
    }

    void reset() override {
        m_state->data<int64_t>()[0] = 0;
    }

    void set_state(const ov::SoPtr<ov::ITensor>&) override {
        OPENVINO_THROW("StoredTokensState::set_state() should not be called!");
    }

    // Returns a copy of state to prevent external modfication.
    ov::SoPtr<ov::ITensor> get_state() const override {
        const auto* state_data = m_state->data<int64_t>();
        auto result = ov::Tensor(ov::element::i64, ov::Shape{1});
        result.data<int64_t>()[0] = state_data[0];
        return ov::get_tensor_impl(result);
    }

private:
    int64_t get_num_stored_tokens() const {
        return m_state->data<int64_t>()[0];
    }

    void set_num_stored_tokens(int64_t num_tokens) {
        m_state->data<int64_t>()[0] = num_tokens;
    }
};

}  // namespace npuw
}  // namespace ov
