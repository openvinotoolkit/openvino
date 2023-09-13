// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/runtime/ivariable_state.hpp"
#include "intel_gpu/plugin/graph.hpp"
#include <functional>

namespace ov {
namespace intel_gpu {

class VariableState : public ov::IVariableState {
public:
    VariableState(const std::string& name, cldnn::network::VariableState::Ptr states, cldnn::engine& engine);

    void reset() override;
    void set_state(const ov::SoPtr<ov::ITensor>& state) override;
    const ov::SoPtr<ov::ITensor>& get_state() const override;

private:
    cldnn::network::VariableState::Ptr m_variable_state;
    cldnn::engine& m_engine;
};

}  // namespace intel_gpu
}  // namespace ov
