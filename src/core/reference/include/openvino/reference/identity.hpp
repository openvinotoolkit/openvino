// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>

namespace ov {
namespace reference {
/**
 * @brief Identity operation computes the identity of the input tensor.
 *
 * @param input Input matrix (matrices) pointer.
 * @param output Output matrix (matrices) pointer.
 * @param size_in_bytes Size of the input tensor in bytes.
 **/
static inline void identity(const char* input, char* output, const size_t size_in_bytes) {
    if (input == output) {
        return;
    } else {
        std::memcpy(output, input, size_in_bytes);
    }
}
}  // namespace reference
}  // namespace ov
