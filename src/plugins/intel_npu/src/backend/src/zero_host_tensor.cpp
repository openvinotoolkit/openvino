// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_host_tensor.hpp"

#include "openvino/runtime/intel_npu/remote_properties.hpp"

namespace intel_npu {

ZeroHostTensor::ZeroHostTensor(std::shared_ptr<ov::IRemoteContext> context,
                               std::shared_ptr<ZeroInitStructsHolder> init_structs,
                               const ov::element::Type element_type,
                               const ov::Shape& shape,
                               const Config& config)
    : m_impl(std::make_shared<ZeroRemoteTensor>(context,
                                                init_structs,
                                                element_type,
                                                shape,
                                                config,
                                                ov::intel_npu::TensorType::BINDED,
                                                ov::intel_npu::MemType::L0_INTERNAL_BUF)) {}

void* ZeroHostTensor::data(const ov::element::Type&) const {
    auto itrHandle = m_impl->get_properties().find(ov::intel_npu::mem_handle.name());
    if (itrHandle == m_impl->get_properties().end()) {
        OPENVINO_THROW("No parameter ", ov::intel_npu::mem_handle.name(), " found in parameters map");
    }

    return ov::Any(itrHandle->second).as<void*>();
}

const ov::element::Type& ZeroHostTensor::get_element_type() const {
    return m_impl->get_element_type();
}

const ov::Shape& ZeroHostTensor::get_shape() const {
    return m_impl->get_shape();
}

const ov::Strides& ZeroHostTensor::get_strides() const {
    return m_impl->get_strides();
}

void ZeroHostTensor::set_shape(ov::Shape new_shape) {
    m_impl->set_shape(new_shape);
}

std::shared_ptr<ZeroRemoteTensor> ZeroHostTensor::get_impl() const {
    return m_impl;
}

}  // namespace intel_npu
