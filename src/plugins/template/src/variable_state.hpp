// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace template_plugin {

class VariableState : public ov::IVariableState {
public:
    VariableState(const std::string& name, const ov::SoPtr<ov::ITensor>& tensor) : ov::IVariableState(name) {
        m_state = tensor;
    }
    void set_state(const ov::SoPtr<ov::ITensor>& state) override {
        OPENVINO_ASSERT(state->get_shape() == m_state->get_shape(), "Wrong tensor shape.");
        OPENVINO_ASSERT(state->get_element_type() == state->get_element_type(), "Wrong tensor type.");
        OPENVINO_ASSERT(state->get_byte_size() == state->get_byte_size(), "Blob size of tensors are not equal.");
        std::memcpy(m_state->data(), state->data(), state->get_byte_size());
    }

    void reset() override {
        std::memset(m_state->data(), 0, m_state->get_byte_size());
    }

    ~VariableState() override = default;
};

}  // namespace template_plugin
}  // namespace ov
