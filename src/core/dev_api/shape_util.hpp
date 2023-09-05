// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"

namespace ov {

template <class TContainer, class TAxes>
TContainer reduce_container(const TContainer& input, const TAxes& axes) {
    TContainer result;
    const auto input_size = input.size();
    result.reserve(input_size);

    for (size_t axis = 0; axis < input_size; ++axis) {
        if (std::find(axes.begin(), axes.end(), axis) == axes.end()) {
            result.emplace_back(input[axis]);
        }
    }

    return result;
}

template <class TContainer, class TAxes>
TContainer replace_container(const TContainer& input, const TAxes& axes) {
    auto result = input;
    for (auto&& axis : axes) {
        result[axis] = 1;
    }
    return result;
}

namespace util {
/**
 * @brief Makes spacial version of 2D ov::Shape which is recognize as dynamic.
 *
 * This is special case used for tensor <-> host tensor conversion to indicate that tensor got dynamic shape.
 *
 * @return 2-D shape with {0, SIZE_MAX}
 */
OPENVINO_DEPRECATED("This function is deprecated and will be removed soon.")
OPENVINO_API Shape make_dynamic_shape();

/**
 * @brief Check if Shape is marked as dynamic.
 *
 * @param s  Shape for check.
 * @return True if shape is dynamic otherwise false.
 */
OPENVINO_DEPRECATED("This function is deprecated and will be removed soon.")
OPENVINO_API bool is_dynamic_shape(const Shape& s);

OPENVINO_API Shape reduce(const Shape& input, const AxisSet& axes);
OPENVINO_API Shape reduce(const Shape& input, const AxisSet& axes, const bool keep_dims);
OPENVINO_API std::vector<size_t> reduce(const std::vector<size_t>& input, const AxisSet& axes);

OPENVINO_API Shape reduce_keep_dims(const Shape& input, const AxisSet& axes);
}  // namespace util
}  // namespace ov
