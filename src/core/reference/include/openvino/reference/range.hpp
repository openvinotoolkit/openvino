// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <type_traits>

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
// Return type is `void`, only enabled if `T` is a built-in FP
// type, or OpenVINO's `bfloat16` or `float16` type.
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value || std::is_same<T, bfloat16>::value ||
                        std::is_same<T, float16>::value>::type
range(const T* start, const T* step, const size_t& num_elem, T* out) {
    for (size_t i = 0; i < num_elem; i++) {
        out[i] = *start + (static_cast<T>(i) * (*step));
    }
}

// Return type is `void`, only enabled if `T` is `is_integral`.
template <typename T>
typename std::enable_if<std::is_integral<T>::value>::type range(const T* start,
                                                                const T* step,
                                                                const size_t& num_elem,
                                                                T* out) {
    T val = *start;

    for (size_t i = 0; i < num_elem; i++) {
        out[i] = val;
        val += *step;
    }
}
}  // namespace reference
}  // namespace ov
