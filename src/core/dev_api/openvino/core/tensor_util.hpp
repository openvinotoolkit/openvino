// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/constant.hpp"
#include "ov_optional.hpp"

namespace ov {
namespace util {

OPENVINO_API Tensor greater_equal(const ov::Tensor& lhs, const ov::Tensor& rhs);
template <typename T>
OPENVINO_API Tensor greater_equal(const ov::Tensor& lhs, const T& element);
OPENVINO_API bool reduce_and(const ov::Tensor& t);
template <typename T>
OPENVINO_API ov::optional<std::vector<T>> to_vector(const ov::Tensor& t);

template <typename T>
Tensor make_tensor_of_value(const element::Type_t& et, const T& value, Shape shape = {}) {
    auto c = op::v0::Constant(et, shape, value);
    auto t = Tensor(et, shape);
    std::memcpy(t.data(), c.get_data_ptr(), t.get_byte_size());
    return t;
}

template <typename T>
Tensor greater_equal(const ov::Tensor& lhs, const T& element) {
    if (!lhs)
        return {};
    const auto& other = make_tensor_of_value(lhs.get_element_type(), element);
    return greater_equal(lhs, other);
}

template <typename T>
ov::optional<std::vector<T>> to_vector(const ov::Tensor& t) {
    ov::optional<std::vector<T>> result;
    if (t)
        result = ov::op::v0::Constant(t).cast_vector<T>();
    return result;
}
}  // namespace util
}  // namespace ov
