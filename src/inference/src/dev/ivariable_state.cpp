// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/ivariable_state.hpp"

#include "openvino/core/except.hpp"

ov::IVariableState::IVariableState(const std::string& name) : m_name(name) {}

ov::IVariableState::~IVariableState() = default;

const std::string& ov::IVariableState::get_name() const {
    return m_name;
}

void ov::IVariableState::reset() {
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::IVariableState::set_state(const ov::Tensor& state) {
    m_state = state;
}

const ov::Tensor& ov::IVariableState::get_state() const {
    return m_state;
}
