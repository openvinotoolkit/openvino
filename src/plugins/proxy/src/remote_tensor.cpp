// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_tensor.hpp"

#include <memory>

#include "openvino/proxy/plugin.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace {
std::shared_ptr<ov::IRemoteTensor> cast_tensor(const ov::SoPtr<ov::ITensor>& tensor) {
    auto rem_tensor = std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr);
    OPENVINO_ASSERT(rem_tensor);
    return rem_tensor;
}
}  // namespace

ov::proxy::RemoteTensor::RemoteTensor(const ov::SoPtr<ov::ITensor>& tensor, const std::string& dev_name)
    : m_name(dev_name),
      m_tensor(tensor) {
    OPENVINO_ASSERT(std::dynamic_pointer_cast<ov::IRemoteTensor>(m_tensor._ptr));
}
ov::proxy::RemoteTensor::RemoteTensor(ov::SoPtr<ov::ITensor>&& tensor, const std::string& dev_name)
    : m_name(dev_name),
      m_tensor(std::move(tensor)) {
    OPENVINO_ASSERT(m_tensor);
}

const ov::AnyMap& ov::proxy::RemoteTensor::get_properties() const {
    return cast_tensor(m_tensor)->get_properties();
}

const std::string& ov::proxy::RemoteTensor::get_device_name() const {
    return m_name;
}

void ov::proxy::RemoteTensor::set_shape(ov::Shape shape) {
    m_tensor->set_shape(shape);
}

const ov::element::Type& ov::proxy::RemoteTensor::get_element_type() const {
    return m_tensor->get_element_type();
}

const ov::Shape& ov::proxy::RemoteTensor::get_shape() const {
    return m_tensor->get_shape();
}

size_t ov::proxy::RemoteTensor::get_size() const {
    return m_tensor->get_size();
}

size_t ov::proxy::RemoteTensor::get_byte_size() const {
    return m_tensor->get_byte_size();
}

const ov::Strides& ov::proxy::RemoteTensor::get_strides() const {
    return m_tensor->get_strides();
}

ov::SoPtr<ov::ITensor> ov::proxy::RemoteTensor::get_hardware_tensor(const ov::SoPtr<ov::ITensor>& tensor, bool unwrap) {
    ov::SoPtr<ov::ITensor> hw_tensor = tensor;
    if (auto remote_tensor = std::dynamic_pointer_cast<ov::proxy::RemoteTensor>(tensor._ptr))
        hw_tensor = remote_tensor->m_tensor;

    return hw_tensor;
}

ov::SoPtr<ov::ITensor> ov::proxy::get_hardware_tensor(const ov::SoPtr<ov::ITensor>& tensor, bool unwrap) {
    return ov::proxy::RemoteTensor::get_hardware_tensor(tensor, unwrap);
}
