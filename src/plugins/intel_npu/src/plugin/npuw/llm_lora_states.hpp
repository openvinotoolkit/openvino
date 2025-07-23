// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/ivariable_state.hpp"

namespace ov {
namespace npuw {

class VariableState final : public ov::IVariableState {
public:
    explicit VariableState(const std::string& name, const ov::SoPtr<ov::ITensor>& tensor) : ov::IVariableState(name) {
        m_state = tensor;
    }

    virtual void set_state(const ov::SoPtr<ov::ITensor>& newState) override {
        m_state = newState;
    }

    virtual void reset() override {
        std::memset(m_state->data(), 0, m_state->get_byte_size());
    }

    ~VariableState() override = default;
};

}  // namespace npuw
}  // namespace ov
