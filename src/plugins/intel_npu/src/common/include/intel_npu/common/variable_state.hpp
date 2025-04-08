// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/ivariable_state.hpp"

namespace intel_npu {

class VariableState final : public ov::IVariableState {
public:
    explicit VariableState(const std::string& name, const ov::SoPtr<ov::ITensor>& tensor) : ov::IVariableState(name) {
        m_state = tensor;
    }

    virtual void set_state(const ov::SoPtr<ov::ITensor>& newState) override {
        if (newState->get_byte_size() != m_state->get_byte_size()) {
            OPENVINO_THROW("Byte size mismatch");
        }

        std::memcpy(m_state->data(), newState->data(), newState->get_byte_size());
    }

    virtual void reset() override {
        std::memset(m_state->data(), 0, m_state->get_byte_size());
    }

    ~VariableState() override = default;
};

}  // namespace intel_npu
