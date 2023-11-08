// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/runtime/layout.hpp"

#include <memory>

namespace ov {
namespace intel_gpu {

VariableState::VariableState(const VariableStateInfo& info, cldnn::engine& engine)
    : ov::IVariableState {info.m_id}
    , m_layout(info.m_layout)
    , m_engine(engine) {
    m_state = ov::make_tensor(m_layout.data_type, get_tensor_shape(m_layout.get_partial_shape()));
    update_device_buffer();
}

void VariableState::reset() {
    m_is_set = false;
}

cldnn::memory::ptr VariableState::get_memory() const {
    OPENVINO_ASSERT(m_memory != nullptr);
    return m_memory;
}

const cldnn::layout& VariableState::get_layout() const {
    return m_layout;
}

bool VariableState::is_set() const {
    return m_is_set;
}
void VariableState::set() {
    m_is_set = true;
}

void VariableState::set_layout(const cldnn::layout& new_layout) {
    m_layout = new_layout;
    update_device_buffer();
}

void VariableState::set_state(const ov::SoPtr<ov::ITensor>& state) {
    const bool blocking = true;
    auto remote_ptr = std::dynamic_pointer_cast<RemoteTensorImpl>(state._ptr);
    m_layout.set_partial_shape(state->get_shape());
    update_device_buffer();
    if (remote_ptr != nullptr) {
        auto user_memory = remote_ptr->get_memory();
        cldnn::mem_lock<uint8_t> lock(user_memory, m_engine.get_service_stream());
        m_memory->copy_from(m_engine.get_service_stream(), lock.data(), blocking);
    } else {
        auto data = state->data();
        m_memory->copy_from(m_engine.get_service_stream(), data, blocking);
    }
    m_is_set = true;
}

void VariableState::update_device_buffer() {
    if (m_layout.is_dynamic() || m_layout.bytes_count() == 0)
        return;

    if (actual_size < m_layout.bytes_count()) {
        const auto alloc_type = m_engine.use_unified_shared_memory() ? cldnn::allocation_type::usm_device : cldnn::allocation_type::cl_mem;
        m_memory = m_engine.allocate_memory(m_layout, alloc_type, false);
        actual_size = std::max(actual_size, m_memory->size());
    } else {
        m_memory = m_engine.reinterpret_buffer(*m_memory, m_layout);
    }
}

ov::SoPtr<ov::ITensor> VariableState::get_state() const {
    const bool blocking = true;
    m_state->set_shape(m_memory->get_layout().get_shape());
    m_memory->copy_to(m_engine.get_service_stream(), m_state->data(), blocking);

    return m_state;
}

}  // namespace intel_gpu
}  // namespace ov
