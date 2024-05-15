// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/constant.hpp"

namespace ov {
namespace util {

OPENVINO_API Tensor ge(const ov::Tensor& lhs, const ov::Tensor& rhs);
template <typename T>
OPENVINO_API Tensor ge(const ov::Tensor& lhs, const T& element);
OPENVINO_API bool all(const ov::Tensor& t);
template <typename T>
OPENVINO_API std::vector<T> to_vector(const ov::Tensor& t);

template <typename T>
Tensor make_tensor_of_value(const element::Type_t& et, const T& value) {
    auto c = op::v0::Constant(et, Shape{}, value);
    auto t = Tensor(et, Shape{});
    std::memcpy(t.data(), c.get_data_ptr(), t.get_byte_size());
    return t;
}

template <typename T>
Tensor make_tensor_of_value(const element::Type_t& et, const Shape& shape, const std::vector<T>& value) {
    auto c = op::v0::Constant(et, shape, value);
    auto t = Tensor(et, Shape{});
    std::memcpy(t.data(), c.get_data_ptr(), t.get_byte_size());
    return t;
}

template <typename T>
Tensor ge(const ov::Tensor& lhs, const T& element) {
    const auto& other = make_tensor_of_value(lhs.get_element_type(), element);
    return ge(lhs, other);
}

template <typename T>
std::vector<T> to_vector(const ov::Tensor& t) {
    return t ? ov::op::v0::Constant(t).cast_vector<T>() : std::vector<T>{};
}
}  // namespace util
}  // namespace ov
