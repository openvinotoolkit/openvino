// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/axis_vector.hpp"
#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {

/**
 * @brief Basic reshape operation, without axes reorder.
 *
 * @param in         Pointer to input data.
 * @param out        Pointer to output data.
 * @param in_shape   Input data shape.
 * @param out_shape  Output data shape.
 * @param elem_size  Single data element size im bytes.
 */
inline void reshape(const char* in, char* out, const Shape& in_shape, size_t elem_size) {
    std::memcpy(out, in, shape_size(in_shape) * elem_size);
}

/**
 * @brief Basic reshape operation for string element type.
 *        Input data are simply copied to the output.
 *        Target shape should be set on the Tensor.
 *
 * @param in         Pointer to input data.
 * @param out        Pointer to output data.
 * @param in_shape   Input data shape.
 */
inline void reshape(const std::string* in, std::string* out, const Shape& in_shape) {
    std::copy_n(in, shape_size(in_shape), out);
}

/**
 * @brief Permutes data shape and axes.
 *
 * @param in            Pointer to input data.
 * @param out           Pointer to output data.
 * @param in_shape      Input data shape.
 * @param axes_order    Axes order.
 * @param out_shape     Output data shape.
 * @param elem_size     Single data element size im bytes.
 */
void reshape(const char* in,
             char* out,
             const Shape& in_shape,
             const AxisVector& axes_order,
             const Shape& out_shape,
             size_t elem_size);
}  // namespace reference
}  // namespace ov
