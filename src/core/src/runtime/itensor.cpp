// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/itensor.hpp"

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

bool ITensor::is_continuous() const {
    if (get_element_type().bitwidth() < 8)
        // OpenVINO doesn't support strides for lp types
        return true;
    const auto& shape = get_shape();
    const auto& type = get_element_type();
    std::vector<size_t> strides(shape.size());
    if (!shape.empty()) {
        strides[shape.size() - 1] = 1;
    }
    auto size = shape.size();
    for (size_t i = 1; i < size; i++) {
        strides[size - i - 1] = strides[size - i] * shape[size - i];
    }

    ov::Strides byte_strides(strides.size());
    for (size_t i = 0; i < strides.size(); ++i)
        byte_strides[i] = strides[i] * type.size();
    return byte_strides == get_strides();
}

}  // namespace ov
