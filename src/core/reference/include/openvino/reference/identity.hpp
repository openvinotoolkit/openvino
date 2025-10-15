// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>
#include <openvino/core/type/element_type.hpp>

namespace ov {
namespace reference {
/**
 * @brief Identity operation computes the identity of the input tensor.
 *
 * @param input Input matrix (matrices) pointer.
 * @param output Output matrix (matrices) pointer.
 * @param size_in_bytes Size of the input tensor in bytes.
 **/
static inline void identity(const void* input,
                            void* output,
                            const size_t size_in_bytes,
                            const ov::element::Type& type) {
    if (input == output) {
        return;
    } else {
        if (type == ov::element::string) {
            const std::string* str_input = static_cast<const std::string*>(input);
            std::string* str_output = static_cast<std::string*>(output);
            // Assign string values one by one
            auto elem_num = size_in_bytes / sizeof(std::string);
            for (size_t i = 0; i < elem_num; i++) {
                str_output[i] = str_input[i];
            }
            return;
        } else {
            std::memcpy(output, input, size_in_bytes);
        }
    }
}
}  // namespace reference
}  // namespace ov
