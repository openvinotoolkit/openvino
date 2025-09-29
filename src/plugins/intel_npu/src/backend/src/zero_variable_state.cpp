// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_variable_state.hpp"

#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_host_tensor.hpp"
#include "intel_npu/utils/zero/zero_remote_tensor.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

namespace intel_npu {

ZeroVariableState::ZeroVariableState(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                     const std::string& name,
                                     const std::shared_ptr<ZeroTensor>& zero_tensor,
                                     size_t tensor_index,
                                     size_t related_tensor_index,
                                     const Config& config)
    : ov::IVariableState(name),
      _init_structs(init_structs),
      _tensor_index(tensor_index),
      _related_tensor_index(related_tensor_index),
      _zero_state(zero_tensor),
      _config(config),
      _logger("ZeroVariableState", _config.get<LOG_LEVEL>()) {
    m_state = _zero_state;
}

void ZeroVariableState::set_state(const ov::SoPtr<ov::ITensor>& new_state) {
    if (m_state._ptr == new_state._ptr) {
        // set_tensor called with the same tensor object; no action needed
        _logger.debug("ZeroVariableState::set_state - got the same state, do nothing");
        return;
    }

    m_state = new_state;
    _is_state_updated = true;

    try {
        _logger.debug("ZeroVariableState::set_state - create zero tensor");
        // Try to use the user tensor directly if its underlying data is already allocated in the same Level Zero
        // context.
        _zero_state = std::make_shared<ZeroTensor>(_init_structs, m_state, _config);
        _is_zero_state_update_needed = true;
    } catch (const ZeroTensorException&) {
        // Check if the current Level Zero tensor was previously shared with the user. If so, it cannot be reused;
        // allocate a new tensor to back up the user tensor (which cannot be imported or used directly).
        if (_zero_state == nullptr || !_zero_state->can_be_reused()) {
            _logger.debug("ZeroVariableState::set_state - allocate locally L0 tensor");
            _zero_state = std::make_shared<ZeroTensor>(_init_structs,
                                                       _config,
                                                       m_state->get_element_type(),
                                                       m_state->get_shape(),
                                                       false);
            _is_zero_state_update_needed = true;
        } else {
            _logger.debug("ZeroVariableState::set_state - reusing the level zero tensor since it is not shared "
                          "with the user");
        }
    }
}

ov::SoPtr<ov::ITensor> ZeroVariableState::get_state() const {
    auto zero_tensor = std::dynamic_pointer_cast<ZeroTensor>(m_state._ptr);
    if (zero_tensor != nullptr) {
        zero_tensor->prevent_reuse();
    }

    return m_state;
}

ov::SoPtr<ov::ITensor> ZeroVariableState::get_user_state() const {
    return m_state;
}

std::shared_ptr<ZeroTensor> ZeroVariableState::get_zero_state() const {
    return _zero_state;
}

void ZeroVariableState::reset() {
    auto remote_tensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(m_state._ptr);

    void* user_buffer = !remote_tensor ? m_state->data() : remote_tensor->get_original_memory();
    std::memset(user_buffer, 0, m_state->get_byte_size());
}

size_t ZeroVariableState::get_tensor_index() const {
    return _tensor_index;
}

size_t ZeroVariableState::get_related_tensor_index() const {
    return _related_tensor_index;
}

bool ZeroVariableState::state_update_pending() const {
    return _is_state_updated;
}

void ZeroVariableState::clear_state_update_pending() {
    _is_state_updated = false;
}

bool ZeroVariableState::zero_state_update_pending() const {
    return _is_zero_state_update_needed;
}

void ZeroVariableState::clear_zero_state_update_pending() {
    _is_zero_state_update_needed = false;
}

}  // namespace intel_npu
