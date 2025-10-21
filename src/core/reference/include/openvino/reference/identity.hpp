// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>
#include <string>

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

/**
 * @brief Identity operation computes the identity of the input tensor.
 *
 * @param input Input matrix (matrices) pointer.
 * @param output Output matrix (matrices) pointer.
 * @param shape_size Size of the input tensor shape.
 **/
static inline void identity(const std::string* input, std::string* output, const size_t shape_size) {
    if (input == output) {
        return;
    } else {
        for (size_t i = 0; i < shape_size; i++) {
            output[i] = input[i];
        }
        return;
    }
}

}  // namespace reference
}  // namespace ov
