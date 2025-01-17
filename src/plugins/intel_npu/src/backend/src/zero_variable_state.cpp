// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_variable_state.hpp"

#include "intel_npu/config/common.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "zero_remote_tensor.hpp"

namespace {

template <typename Type>
Type extract_object(const ov::AnyMap& params, const ov::Property<Type>& p) {
    auto itrHandle = params.find(p.name());
    ov::Any res = nullptr;
    if (itrHandle == params.end()) {
        OPENVINO_THROW("No parameter ", p.name(), " found in parameters map");
    }
    res = itrHandle->second;
    return res.as<Type>();
}

bool memory_was_allocated_in_the_same_l0_context(ze_context_handle_t hContext, const void* ptr) {
    ze_memory_allocation_properties_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    auto res = intel_npu::zeMemGetAllocProperties(hContext, ptr, &desc, nullptr);
    if (res == ZE_RESULT_SUCCESS) {
        if (desc.id) {
            if ((desc.type & ZE_MEMORY_TYPE_HOST) || (desc.type & ZE_MEMORY_TYPE_DEVICE) ||
                (desc.type & ZE_MEMORY_TYPE_SHARED)) {
                return true;
            }
        }
    }

    return false;
}

}  // namespace

namespace intel_npu {

ZeroVariableState::ZeroVariableState(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                     const IODescriptor& descriptor,
                                     const ov::SoPtr<ov::ITensor>& tensor,
                                     size_t index,
                                     const Config& config)
    : VariableState(descriptor.nameFromCompiler, tensor),
      _init_structs(init_structs),
      _descriptor(descriptor),
      _index(index),
      _logger("ZeroVariableState", config.get<LOG_LEVEL>()) {}

void ZeroVariableState::set_state(const ov::SoPtr<ov::ITensor>& new_state) {
    m_state = new_state;
    _tensor_updated = true;

    if (_init_structs->getMutableCommandListVersion()) {
        if (!is_remote_tensor(new_state._ptr)) {
            if (memory_was_allocated_in_the_same_l0_context(_init_structs->getContext(), new_state->data())) {
                _logger.debug("ZeroInferRequest::set_tensor_data - tensor was created in the same L0 context");
                _zero_tensor_updated = true;
            }

            return;
        }

        _zero_tensor_updated = true;
    }
}

void ZeroVariableState::reset() {
    auto remoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(m_state._ptr);

    void* userBuffer =
        !remoteTensor ? m_state->data() : extract_object(remoteTensor->get_properties(), ov::intel_npu::mem_handle);

    std::memset(userBuffer, 0, m_state->get_byte_size());
}

size_t ZeroVariableState::get_index() const {
    return _index;
}

const IODescriptor& ZeroVariableState::get_descriptor() const {
    return _descriptor;
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
