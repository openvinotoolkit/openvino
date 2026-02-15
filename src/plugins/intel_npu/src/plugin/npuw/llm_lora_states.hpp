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

struct LoRANames {
    static constexpr const char* MatMul_A = "MatMul\\.A";
    static constexpr const char* MatMul_B = "MatMul\\.B";
    static constexpr const char* MatMul_alpha = "MatMul\\.alpha";
};

}  // namespace npuw
}  // namespace ov
