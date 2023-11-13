// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/make_tensor.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/runtime/layout.hpp"

#include <memory>

namespace ov {
namespace intel_gpu {

VariableState::VariableState(const std::string &name, cldnn::network::VariableState::Ptr state, cldnn::engine& engine)
    : ov::IVariableState {name}
    , m_variable_state(state)
    , m_engine(engine) {
    auto internal_memory = m_variable_state->memory;
    auto internal_layout = internal_memory->get_layout();
    auto shape = internal_layout.get_shape();
    m_state = ov::make_tensor(internal_layout.data_type, shape);
}

void VariableState::reset() {
    m_variable_state->is_set = false;
}

void VariableState::set_state(const ov::SoPtr<ov::ITensor>& state) {
    const bool blocking = true;
    auto remote_ptr = std::dynamic_pointer_cast<RemoteTensorImpl>(state._ptr);
    if (remote_ptr != nullptr) {
        auto user_memory = remote_ptr->get_memory();
        cldnn::mem_lock<uint8_t> lock(user_memory, m_engine.get_service_stream());
        m_variable_state->memory->copy_from(m_engine.get_service_stream(), lock.data(), blocking);
    } else {
        auto data = state->data();
        m_variable_state->memory->copy_from(m_engine.get_service_stream(), data, blocking);
    }
    m_variable_state->is_set = true;
}

ov::SoPtr<ov::ITensor> VariableState::get_state() const {
    auto internal_memory = m_variable_state->memory;
    const bool blocking = true;
    internal_memory->copy_to(m_engine.get_service_stream(), m_state->data(), blocking);

    return m_state;
}

}  // namespace intel_gpu
}  // namespace ov
