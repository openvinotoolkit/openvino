// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_variable_state.hpp"

#include "intel_npu/config/common.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "zero_remote_tensor.hpp"

namespace intel_npu {

ZeroVariableState::ZeroVariableState(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                     const std::string& name,
                                     const ov::SoPtr<ov::ITensor>& tensor,
                                     size_t tensor_index,
                                     size_t related_tensor_index,
                                     const Config& config)
    : ov::IVariableState(name),
      _init_structs(init_structs),
      _tensor_index(tensor_index),
      _related_tensor_index(related_tensor_index),
      _logger("ZeroVariableState", config.get<LOG_LEVEL>()) {
    m_state = tensor;
}

void ZeroVariableState::set_state(const ov::SoPtr<ov::ITensor>& new_state) {
    m_state = new_state;
    _tensor_updated = true;

    if (_init_structs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0)) {
        if (!is_remote_tensor(new_state._ptr)) {
            if (zeroUtils::memory_was_allocated_in_the_same_l0_context(_init_structs->getContext(),
                                                                       new_state->data())) {
                _logger.debug("ZeroVariableState::set_state - tensor was created in the same L0 context");
                _zero_tensor_updated = true;
            }

            return;
        }

        _zero_tensor_updated = true;
    }
}

void ZeroVariableState::reset() {
    auto remoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(m_state._ptr);

    void* userBuffer = !remoteTensor ? m_state->data() : remoteTensor->get_original_memory();

    std::memset(userBuffer, 0, m_state->get_byte_size());
}

size_t ZeroVariableState::get_tensor_index() const {
    return _tensor_index;
}

size_t ZeroVariableState::get_related_tensor_index() const {
    return _related_tensor_index;
}

bool ZeroVariableState::tensor_was_updated() const {
    return _tensor_updated;
}

void ZeroVariableState::reset_tensor_updated_flag() {
    _tensor_updated = false;
}

bool ZeroVariableState::zero_tensor_should_be_updated() const {
    return _zero_tensor_updated;
}

void ZeroVariableState::reset_zero_tensor_updated_flag() {
    _zero_tensor_updated = false;
}

}  // namespace intel_npu
