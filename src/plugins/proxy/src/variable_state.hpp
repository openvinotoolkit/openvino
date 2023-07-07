
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/runtime/ivariable_state.hpp"

namespace ov {
namespace proxy {
/**
 * @brief Simple wrapper for hardware variable states which holds plugin shared object
 */
class VariableState : public ov::IVariableState {
    std::shared_ptr<ov::IVariableState> m_state;
    std::shared_ptr<void> m_so;

public:
    /**
     * @brief Constructor of proxy VariableState
     *
     * @param state hardware state
     * @param so shared object
     */
    VariableState(const std::shared_ptr<ov::IVariableState>& state, const std::shared_ptr<void>& so)
        : IVariableState(""),
          m_state(state),
          m_so(so) {
        OPENVINO_ASSERT(m_state);
    }
    const std::string& get_name() const override {
        return m_state->get_name();
    }

    void reset() override {
        m_state->reset();
    }

    void set_state(const ov::Tensor& state) override {
        m_state->set_state(state);
    }

    const ov::Tensor& get_state() const override {
        return m_state->get_state();
    }
};

}  // namespace proxy
}  // namespace ov
