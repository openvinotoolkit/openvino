// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "llm_continuation.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace ov {
namespace npuw {

class LLMInferRequest;

// Special VariableState for already processed/stored tokens in LLM.
// Allows external pipelines to call reset on that state where needed.
//
// When continuous prefill is enabled for the owning request, this state also
// becomes the propose/grant negotiation channel. A set_state(N) call is a
// proposal that the first N tokens are unchanged. The plugin clamps N by every
// limit it owns and stores the result as a pending command. A get_state() call
// returns the grant, which is the prefix the plugin will honour. reset() is a
// control operation that discards any pending keep and requires the next
// inference to be a full prefill at cache position zero.
//
// A write is not idempotent. The caller writes X and reads back Y <= X, and
// must slice its input at Y, never at X.
//
// Without continuous prefill the legacy contract is unchanged. get_state()
// returns the stored token count, reset() zeroes it and set_state() throws.
class StoredTokensState : public ov::IVariableState {
public:
    friend class ov::npuw::LLMInferRequest;
    StoredTokensState() : ov::IVariableState("npuw_stored_tokens_state") {
        auto tensor = ov::Tensor(ov::element::i64, ov::Shape{1});
        m_state = ov::get_tensor_impl(tensor);
        m_state->data<int64_t>()[0] = 0;
    }

    void reset() override {
        if (has_continuation()) {
            m_coordinator->request_reset();
        }
        m_state->data<int64_t>()[0] = 0;
    }

    void set_state(const ov::SoPtr<ov::ITensor>& new_state) override {
        if (!has_continuation()) {
            OPENVINO_THROW("npuw_stored_tokens_state: set_state() is a continuous prefill proposal and requires "
                           "NPUW_LLM_ENABLE_CONTINUOUS_PREFILL on a model where "
                           "NPUW_LLM_CONTINUOUS_PREFILL_SUPPORTED reports true.");
        }
        OPENVINO_ASSERT(new_state, "npuw_stored_tokens_state: set_state() received a null tensor.");
        OPENVINO_ASSERT(new_state->get_element_type() == ov::element::i64,
                        "npuw_stored_tokens_state: proposal tensor must be i64, got ",
                        new_state->get_element_type());
        OPENVINO_ASSERT(new_state->get_size() == 1u,
                        "npuw_stored_tokens_state: proposal tensor must be a scalar (one element), got ",
                        new_state->get_size(),
                        " elements.");
        m_coordinator->propose(new_state->data<int64_t>()[0]);
    }

    // Returns a copy of state to prevent external modification.
    ov::SoPtr<ov::ITensor> get_state() const override {
        auto result = ov::Tensor(ov::element::i64, ov::Shape{1});
        result.data<int64_t>()[0] = has_continuation() ? m_coordinator->query() : m_state->data<int64_t>()[0];
        return ov::get_tensor_impl(result);
    }

private:
    // Wire the negotiation channel. The coordinator is owned by LLMInferRequest,
    // which also owns (and outlives) this state.
    void attach_continuation(ContinuationCoordinator* coordinator) {
        m_coordinator = coordinator;
    }

    bool has_continuation() const {
        return m_coordinator != nullptr && m_coordinator->enabled();
    }

    int64_t get_num_stored_tokens() const {
        return m_state->data<int64_t>()[0];
    }

    void set_num_stored_tokens(int64_t num_tokens) {
        m_state->data<int64_t>()[0] = num_tokens;
    }

    ContinuationCoordinator* m_coordinator = nullptr;
};

}  // namespace npuw
}  // namespace ov
