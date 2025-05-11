// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace ov {
namespace reference {
template <typename T, typename U>
typename std::enable_if<std::is_floating_point<T>::value, void>::type is_nan(const T* input, U* output, size_t count) {
    std::transform(input, input + count, output, [](T x) {
        return std::isnan(static_cast<float>(x));
    });
}

// used for float16 and bfloat 16 datatypes
template <typename T, typename U>
typename std::enable_if<std::is_class<T>::value, void>::type is_nan(const T* input, U* output, size_t count) {
    std::transform(input, input + count, output, [](T x) -> U {
        return std::isnan(static_cast<float>(x));
    });
}
}  // namespace reference
}  // namespace ov
