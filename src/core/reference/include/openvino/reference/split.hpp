// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/core/shape.hpp"

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
}  // namespace reference
}  // namespace ov
