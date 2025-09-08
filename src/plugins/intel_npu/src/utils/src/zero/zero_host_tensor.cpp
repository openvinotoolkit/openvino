// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_host_tensor.hpp"

#include "openvino/runtime/intel_npu/remote_properties.hpp"

namespace intel_npu {

ZeroHostTensor::ZeroHostTensor(const std::shared_ptr<ov::IRemoteContext>& context,
                               const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                               const ov::element::Type element_type,
                               const ov::Shape& shape,
                               ov::intel_npu::TensorType tensor_type)
    : _impl(std::make_shared<ZeroRemoteTensor>(context,
                                               init_structs,
                                               element_type,
                                               shape,
                                               tensor_type,
                                               ov::intel_npu::MemType::L0_INTERNAL_BUF)) {}

// Note: Override data() members to not used OpenVINO library code to improve performance
void* ZeroHostTensor::data() {
    return _impl->get_original_memory();
}

void* ZeroHostTensor::data(const ov::element::Type&) {
    return _impl->get_original_memory();
}

const void* ZeroHostTensor::data() const {
    return _impl->get_original_memory();
}

const void* ZeroHostTensor::data(const ov::element::Type&) const {
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
