// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
namespace identity {

/**
 * @brief Identity operation computes the identity of the input tensor.
 *
 * @param input Input matrix (matrices) pointer.
 * @param output Output matrix (matrices) pointer.
 * @param copy Boolean that determines whether to return the input as output or
 * copy the input to a new memory address.
 **/
template <typename T>
void identity(const T** input, T** output, const Shape& shape, const bool copy) {
    const auto total_elements = shape_size<Shape>(shape);

    if (!copy) {
        *output = *input;
    } else {
        std::memcpy(*output, *input, total_elements * sizeof(T));
    }
}
}  // namespace identity
}  // namespace reference
}  // namespace ov
