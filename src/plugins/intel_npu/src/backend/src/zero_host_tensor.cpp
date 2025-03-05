// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_host_tensor.hpp"

#include "openvino/runtime/intel_npu/remote_properties.hpp"

namespace intel_npu {

ZeroHostTensor::ZeroHostTensor(const std::shared_ptr<ov::IRemoteContext>& context,
                               const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                               const ze_device_properties_t& device_properties,
                               const ov::element::Type element_type,
                               const ov::Shape& shape,
                               const Config& config,
                               ov::intel_npu::TensorType tensor_type)
    : _impl(std::make_shared<ZeroRemoteTensor>(context,
                                               init_structs,
                                               device_properties,
                                               element_type,
                                               shape,
                                               config,
                                               tensor_type,
                                               ov::intel_npu::MemType::L0_INTERNAL_BUF)) {}

void* ZeroHostTensor::data(const ov::element::Type&) const {
    return _impl->get_original_memory();
}

const ov::element::Type& ZeroHostTensor::get_element_type() const {
    return _impl->get_element_type();
}

const ov::Shape& ZeroHostTensor::get_shape() const {
    return _impl->get_shape();
}

const ov::Strides& ZeroHostTensor::get_strides() const {
    return _impl->get_strides();
}

void ZeroHostTensor::set_shape(ov::Shape new_shape) {
    _impl->set_shape(new_shape);
}

std::shared_ptr<ZeroRemoteTensor> ZeroHostTensor::get_impl() const {
    return _impl;
}

}  // namespace intel_npu
