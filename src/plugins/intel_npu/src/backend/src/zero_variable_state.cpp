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
                                     const ov::SoPtr<ov::ITensor>& tensor,
                                     size_t tensor_index,
                                     size_t related_tensor_index,
                                     const Config& config)
    : ov::IVariableState(name),
      _init_structs(init_structs),
      _tensor_index(tensor_index),
      _related_tensor_index(related_tensor_index),
      _config(config),
      _logger("ZeroVariableState", _config.get<LOG_LEVEL>()) {
    m_state = tensor;
}

void ZeroVariableState::set_state(const ov::SoPtr<ov::ITensor>& new_state) {
    m_state = new_state;
    _tensor_updated = true;

    if (_init_structs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0)) {
        try {
            _logger.debug("ZeroVariableState::set_state - create zero tensor");
            // Try to use the user tensor directly if its underlying data is already allocated in the same Level Zero
            // context.
            _zero_state = std::make_shared<ZeroTensor>(_init_structs, m_state, _config);
            _zero_tensor_updated = true;
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
                _zero_tensor_updated = true;
            }
        }
        // If command list updates are not supported, fallback to copying tensors every time.
    }
}

ov::SoPtr<ov::ITensor> ZeroVariableState::get_state() const {
    auto zeroTensor = std::dynamic_pointer_cast<ZeroTensor>(m_state._ptr);
    if (zeroTensor != nullptr) {
        zeroTensor->prevent_reuse();
    }

    return m_state;
}

std::shared_ptr<ZeroTensor> ZeroVariableState::get_zero_state() const {
    return _zero_state;
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
