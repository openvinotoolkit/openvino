// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/itensor.hpp"

#include "dev/make_tensor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/allocator.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {

ITensor::~ITensor() = default;

size_t ITensor::get_size() const {
    return shape_size(get_shape());
}

size_t ITensor::get_byte_size() const {
    return (get_size() * get_element_type().bitwidth() + 8 - 1) / 8;
}

}  // namespace ov
