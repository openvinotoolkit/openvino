// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "openvino/core/except.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace npuw {

// A variable state over a tensor that is bound for the lifetime of the model
// (e.g. LoRA adapter weights), not per-inference data. It deliberately
// implements no reset(): such a state has no meaningful "initial" value to
// return to, so resetting it is an error.
class VariableState final : public ov::IVariableState {
public:
    VariableState(const std::string& name, const ov::SoPtr<ov::ITensor>& tensor) : ov::IVariableState(name) {
        m_state = tensor;
        clear_state_updated();
    }

    void set_state(const ov::SoPtr<ov::ITensor>& newState) override {
        m_state = newState;
        m_state_updapted = true;
    }

    void reset() override {
        OPENVINO_THROW("VariableState::reset() is not implemented");
    }

    ~VariableState() override = default;

    bool is_state_updated() const {
        return m_state_updapted;
    }

    void clear_state_updated() {
        m_state_updapted = false;
    }

private:
    bool m_state_updapted;
};

// Out-of-band channel for the number of tokens already stored in the KV cache before the
// next infer() (the "history"). Dynamic attention needs it to pick the past KV / mask shape,
// and a caller that only holds a public ov::InferRequest has no other way to provide it:
//   for (auto& state : request.query_state()) {
//       if (state.get_name() == ov::npuw::HistorySizeState::name) {
//           ov::Tensor t(ov::element::i64, ov::Shape{1});
//           t.data<int64_t>()[0] = history;
//           state.set_state(t);
//       }
//   }
// The value persists across infer() calls and must be set before every prefill chunk.
// It starts as kUnset (-1), and so does it after reset(); a prefill with dynamic attention
// fails if it still reads kUnset. Decode does not use it.
class HistorySizeState final : public ov::IVariableState {
public:
    static constexpr const char* name = "npuw_history_size";
    static constexpr int64_t kUnset = -1;

    HistorySizeState() : ov::IVariableState(name) {
        m_state = ov::get_tensor_impl(ov::Tensor(ov::element::i64, ov::Shape{1}));
        reset();
    }

    void reset() override {
        set(kUnset);
    }

    // A copy, so mutating the returned tensor cannot change the history behind the request's back.
    ov::SoPtr<ov::ITensor> get_state() const override {
        ov::Tensor copy(ov::element::i64, ov::Shape{1});
        copy.data<int64_t>()[0] = get();
        return ov::get_tensor_impl(copy);
    }

    void set_state(const ov::SoPtr<ov::ITensor>& newState) override {
        OPENVINO_ASSERT(newState, "HistorySizeState expects a non-null tensor");
        OPENVINO_ASSERT(newState->get_element_type() == ov::element::i64, "HistorySizeState expects an int64 tensor");
        OPENVINO_ASSERT(newState->get_size() == 1, "HistorySizeState expects a single-element tensor");
        const auto value = newState->data<int64_t>()[0];
        OPENVINO_ASSERT(value >= 0, "HistorySizeState expects a non-negative history size");
        set(value);
    }

    int64_t get() const {
        return m_state->data<int64_t>()[0];
    }

    void set(int64_t value) {
        m_state->data<int64_t>()[0] = value;
    }
};

}  // namespace npuw
}  // namespace ov
