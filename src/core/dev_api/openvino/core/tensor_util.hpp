// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "openvino/op/constant.hpp"

namespace ov {
namespace util {

OPENVINO_API Tensor greater_equal(const ov::Tensor& lhs, const ov::Tensor& rhs);
template <typename T>
OPENVINO_API Tensor greater_equal(const ov::Tensor& lhs, const T& element);
OPENVINO_API bool reduce_and(const ov::Tensor& t);
template <typename T>
OPENVINO_API std::optional<std::vector<T>> to_vector(const ov::Tensor& t);

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
std::optional<std::vector<T>> to_vector(const ov::Tensor& t) {
    std::optional<std::vector<T>> result;
    if (t)
        result = ov::op::v0::Constant(t).cast_vector<T>();
    return result;
}

/// \brief Read a tensor content from a file. Only raw data is loaded.
/// \param file_name Path to file to read.
/// \param element_type Element type, when not specified the it is assumed as element::u8.
/// \param shape Shape for resulting tensor. If provided shape is static, specified number of elements is read only.
///              File should contain enough bytes, an exception is raised otherwise.
///              One of the dimensions can be dynamic. In this case it will be determined automatically based on the
///              length of the file content and `offset`. Default value is [?].
/// \param offset_in_bytes Read file starting from specified offset. Default is 0. The remining size of the file should
/// be compatible with shape.
Tensor read_tensor_data(const std::filesystem::path& file_name,
                             const element::Type& element_type = element::u8,
                             const PartialShape& shape = PartialShape::dynamic(1),
                             std::size_t offset_in_bytes = 0);

}  // namespace util
}  // namespace ov
