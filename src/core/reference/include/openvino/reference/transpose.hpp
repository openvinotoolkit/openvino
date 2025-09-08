// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cfenv>
#include <cmath>
#include <numeric>
#include <vector>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {

/**
 * @brief Reference implementation of Transpose operator.
 *
 * @param data          Pointer to input data.
 * @param out           Pointer to output data.
 * @param data_shape    Input data shape.
 * @param element_size  Element size in bytes for input and output.
 * @param axes_order    Transpose order.
 * @param out_shape     Output data shape.
 */
void transpose(const char* data,
               char* out,
               const Shape& data_shape,
               size_t element_size,
               const std::vector<int64_t>& axes_order,
               const Shape& out_shape);
}  // namespace reference
}  // namespace ov
