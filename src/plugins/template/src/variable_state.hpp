// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/ivariable_state.hpp"

namespace ov {
namespace template_plugin {

class VariableState : public ov::IVariableState {
public:
    VariableState(const std::string& name, const ov::Tensor& tensor) : ov::IVariableState(name) {
        set_state(tensor);
    }
    void reset() override {
        std::memset(m_state.data(), 0, m_state.get_byte_size());
    }

    ~VariableState() override = default;
};

}  // namespace template_plugin
}  // namespace ov
