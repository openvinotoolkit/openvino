// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace reference {

/**
 * @brief Reference implementation of Split operator.
 *
 * @param data        Pointer to input data.
 * @param data_shape  Input data shape.
 * @param elem_size   Size of single element type.
 * @param axis        Axis used for split input data.
 * @param num_splits  Number of splits
 * @param out_data    Pointer to output data pointers (must have size of num_splits)
 */
void split(const char* data,
           const Shape& data_shape,
           size_t elem_size,
           int64_t axis,
           size_t num_splits,
           char** out_data);

/**
 * @brief Reference implementation of Split for packed sub-byte element types (e.g. u1, u2, u3, u4, i4, u6, nf4,
 * f4e2m1).
 *
 * Elements are copied as raw bits, so outputs keep the input element type and packing, also when split boundaries
 * are not byte aligned.
 *
 * @param data           Pointer to packed input data.
 * @param data_shape     Input data shape.
 * @param element_type   Packed element type of input and outputs.
 * @param axis           Normalized axis used for split input data.
 * @param split_lengths  Length of each output along axis (sum must be equal to data_shape[axis]).
 * @param out_data       Pointers to packed output data (must have size of split_lengths).
 */
void split_packed(const uint8_t* data,
                  const Shape& data_shape,
                  const element::Type& element_type,
                  size_t axis,
                  const std::vector<size_t>& split_lengths,
                  uint8_t** out_data);
}  // namespace reference
}  // namespace ov
