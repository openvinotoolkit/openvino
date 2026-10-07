// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/ivariable_state.hpp"

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

}  // namespace npuw
}  // namespace ov
