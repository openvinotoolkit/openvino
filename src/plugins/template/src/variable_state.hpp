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
    VariableState(const std::string& name, const std::shared_ptr<op::util::VariableValue>& variable_value)
        : ov::IVariableState(name),
          m_variable_value(variable_value) {
        m_state = get_tensor_impl(variable_value->get_state());
    }
    void set_state(const ov::SoPtr<ov::ITensor>& state) override {
        OPENVINO_ASSERT(state->get_shape() == m_state->get_shape(), "Wrong tensor shape.");
        OPENVINO_ASSERT(state->get_element_type() == state->get_element_type(), "Wrong tensor type.");
        OPENVINO_ASSERT(state->get_byte_size() == state->get_byte_size(), "Blob size of tensors are not equal.");
        std::memcpy(m_state->data(), state->data(), state->get_byte_size());
    }

    void reset() override {
        std::memset(m_state->data(), 0, m_state->get_byte_size());
        m_variable_value->set_reset(true);
    }

    ~VariableState() override = default;

private:
    std::shared_ptr<op::util::VariableValue> m_variable_value;
};
}  // namespace template_plugin
}  // namespace ov
