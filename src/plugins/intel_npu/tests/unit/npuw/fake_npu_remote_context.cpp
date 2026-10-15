// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fake_npu_remote_context.hpp"

#include "openvino/runtime/intel_npu/remote_properties.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace ov::test::npuw {

FakeRemoteTensor::FakeRemoteTensor(std::shared_ptr<ov::ITensor> host, std::string device_name)
    : m_host(std::move(host)),
      m_device_name(std::move(device_name)) {}

void FakeRemoteTensor::set_shape(ov::Shape shape) {
    m_host->set_shape(std::move(shape));
}

const ov::element::Type& FakeRemoteTensor::get_element_type() const {
    return m_host->get_element_type();
}

const ov::Shape& FakeRemoteTensor::get_shape() const {
    return m_host->get_shape();
}

const ov::Strides& FakeRemoteTensor::get_strides() const {
    return m_host->get_strides();
}

const ov::AnyMap& FakeRemoteTensor::get_properties() const {
    return m_properties;
}

const std::string& FakeRemoteTensor::get_device_name() const {
    return m_device_name;
}

void FakeRemoteTensor::copy_to(const std::shared_ptr<ov::ITensor>& dst, size_t, size_t, const ov::Shape&) const {
    m_host->copy_to(dst);
}

void FakeRemoteTensor::copy_from(const std::shared_ptr<const ov::ITensor>& src, size_t, size_t, const ov::Shape&) {
    src->copy_to(m_host);
}

const void* FakeRemoteTensor::host_data() const {
    return m_host->data();
}

FakeNpuRemoteTensor::FakeNpuRemoteTensor(std::shared_ptr<ov::ITensor> host)
    : FakeRemoteTensor(std::move(host), "NPU") {}

FakeRemoteContext::FakeRemoteContext(std::string device_name) : m_device_name(std::move(device_name)) {}

const std::string& FakeRemoteContext::get_device_name() const {
    return m_device_name;
}

const ov::AnyMap& FakeRemoteContext::get_property() const {
    return m_properties;
}


std::shared_ptr<FakeRemoteTensor> FakeRemoteContext::wrap_host_tensor(std::shared_ptr<ov::ITensor> host) const {
    return std::make_shared<FakeRemoteTensor>(std::move(host), m_device_name);
}

FakeNpuRemoteContext::FakeNpuRemoteContext() : FakeRemoteContext("NPU") {}

std::shared_ptr<FakeRemoteTensor> FakeNpuRemoteContext::wrap_host_tensor(std::shared_ptr<ov::ITensor> host) const {
    return std::make_shared<FakeNpuRemoteTensor>(std::move(host));
}

ov::SoPtr<ov::IRemoteTensor> FakeNpuRemoteContext::create_tensor(const ov::element::Type& type,
                                                              const ov::Shape& shape,
                                                              const ov::AnyMap& params) {
    auto it = params.find(ov::intel_npu::mem_handle.name());
    std::shared_ptr<ov::ITensor> host;
    if (it != params.end()) {
        host = ov::make_tensor(type, shape, it->second.as<void*>());
        increment_imported_remote_tensors();
    } else {
        host = ov::make_tensor(type, shape);
        increment_allocated_remote_tensors();
    }
    return {wrap_host_tensor(std::move(host)), nullptr};
}

const void* tensor_data_ptr(const ov::Tensor& tensor) {
    auto impl = ov::get_tensor_impl(tensor)._ptr;
    if (auto remote = std::dynamic_pointer_cast<FakeRemoteTensor>(impl)) {
        return remote->host_data();
    }
    return tensor.data();
}

}  // namespace ov::test::npuw
