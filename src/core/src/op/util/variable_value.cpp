// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/variable_value.hpp"

#include <memory>

#include "openvino/core/shape.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/runtime/allocator.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/tensor.hpp"

void ov::op::util::VariableValue::set_reset(bool reset) {
    m_reset = reset;
}

bool ov::op::util::VariableValue::get_reset() const {
    return m_reset;
}

ov::op::util::VariableValue::VariableValue(const ov::Tensor& value) : m_value(value) {}

ov::op::util::VariableValue::VariableValue(const ov::Tensor& value, bool reset) : m_reset(reset), m_value(value) {}

const ov::Tensor& ov::op::util::VariableValue::get_state() const {
    return m_value;
}

void ov::op::util::VariableValue::set_state(const ov::Tensor& value) {
    m_value = value;
}
