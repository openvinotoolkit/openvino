// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/usm_host_tensor.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include <memory>

namespace ov::intel_gpu {

USMHostTensor::USMHostTensor(std::shared_ptr<RemoteContextImpl> context, const element::Type element_type, const Shape& shape)
    : m_impl(std::make_shared<RemoteTensorImpl>(context, shape, element_type, TensorType::BT_USM_HOST_INTERNAL)) {}

USMHostTensor::USMHostTensor(std::shared_ptr<RemoteTensorImpl> tensor)
    : m_impl(tensor) {}

void* USMHostTensor::data(const element::Type& element_type) const {
    return m_impl->get_original_memory()->buffer_ptr();
}

const element::Type& USMHostTensor::get_element_type() const {
    return m_impl->get_element_type();
}

const Shape& USMHostTensor::get_shape() const {
    return m_impl->get_shape();
}

const Strides& USMHostTensor::get_strides() const {
    return m_impl->get_strides();
}

void USMHostTensor::set_shape(ov::Shape new_shape) {
    m_impl->set_shape(new_shape);
}

void USMHostTensor::set_memory(std::shared_ptr<RemoteTensorImpl> tensor) {
    OPENVINO_ASSERT(tensor->get_original_memory()->get_allocation_type() == cldnn::allocation_type::usm_host, "[GPU] Unexpected allocation type");
    m_impl = tensor;
}

std::shared_ptr<RemoteTensorImpl> USMHostTensor::get_impl() const {
    return m_impl;
}

}  // namespace ov::intel_gpu
