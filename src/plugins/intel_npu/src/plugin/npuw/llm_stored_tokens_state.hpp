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
    StoredTokensState() : ov::IVariableState("npuw_stored_tokens_state") {
        auto tensor = ov::Tensor(ov::element::i64, ov::Shape{1});
        m_state = ov::get_tensor_impl(tensor);
        reset();
    }

    int64_t get_num_stored_tokens() const {
        return m_state->data<int64_t>()[0];
    }

    void set_num_stored_tokens(int64_t num_tokens) {
        m_state->data<int64_t>()[0] = num_tokens;
    }

    void reset() override {
        m_state->data<int64_t>()[0] = 0;
    }

    void set_state(const ov::SoPtr<ov::ITensor>& state) override {
        OPENVINO_ASSERT(state->get_element_type() == ov::element::i64, "StoredTokensState expects int64 tensor");
        OPENVINO_ASSERT(state->get_size() == 1, "StoredTokensState tensor must have exactly 1 element");
        m_state->data<int64_t>()[0] = state->data<int64_t>()[0];
    }
};

}  // namespace npuw
}  // namespace ov
