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
 **/
void identity(const char* input, char* output, const size_t size_in_bytes) {
    std::memcpy(output, input, size_in_bytes);
}
}  // namespace identity
}  // namespace reference
}  // namespace ov
