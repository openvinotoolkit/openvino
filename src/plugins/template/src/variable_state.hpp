// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/op/util/variable.hpp"

namespace ov {
namespace template_plugin {

class VariableState : public ov::IVariableState {
public:
    VariableState(const ov::op::util::VariableInfo& variable_info,
                  const std::shared_ptr<op::util::VariableValue>& variable_value)
        : ov::IVariableState(variable_info.variable_id),
          m_data_shape(variable_info.data_shape),
          m_data_type(variable_info.data_type),
          m_variable_value(variable_value) {
        m_state = get_tensor_impl(variable_value->get_state());
    }
    void set_state(const ov::SoPtr<ov::ITensor>& state) override {
        OPENVINO_ASSERT(m_data_shape.compatible(state->get_shape()), "Wrong tensor shape");
        OPENVINO_ASSERT(m_data_type.compatible(state->get_element_type()), "Wrong tensor type.");
        m_state->set_shape(state->get_shape());
        OPENVINO_ASSERT(state->get_byte_size() == m_state->get_byte_size(), "Blob size of tensors are not equal.");
        std::memcpy(m_state->data(), state->data(), state->get_byte_size());
        m_variable_value->set_reset(false);
    }

    void reset() override {
        std::memset(m_state->data(), 0, m_state->get_byte_size());
        m_variable_value->set_reset(true);
    }

    ~VariableState() override = default;

private:
    PartialShape m_data_shape; //original shape
    element::Type m_data_type; //original type
    std::shared_ptr<op::util::VariableValue> m_variable_value;
};
}  // namespace template_plugin
}  // namespace ov
