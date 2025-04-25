// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "openvino/op/is_inf.hpp"

namespace ov {
namespace reference {
template <typename T, typename U>
typename std::enable_if<std::is_floating_point<T>::value, void>::type
is_inf(const T* input, U* output, size_t count, const ov::op::v10::IsInf::Attributes& attributes) {
    if (attributes.detect_negative && attributes.detect_positive) {
        std::transform(input, input + count, output, [](T x) -> U {
            return std::isinf(x);
        });
    } else if (!attributes.detect_negative && attributes.detect_positive) {
        std::transform(input, input + count, output, [](T x) -> U {
            return (x == std::numeric_limits<T>::infinity());
        });
    } else if (attributes.detect_negative && !attributes.detect_positive) {
        std::transform(input, input + count, output, [](T x) -> U {
            return (x == -std::numeric_limits<T>::infinity());
        });
    } else {
        std::memset(output, 0, count);
    }
}

// used for float16 and bfloat 16 datatypes
template <typename T, typename U>
typename std::enable_if<std::is_class<T>::value, void>::type is_inf(const T* input,
                                                                    U* output,
                                                                    size_t count,
                                                                    const ov::op::v10::IsInf::Attributes& attributes) {
    if (attributes.detect_negative && attributes.detect_positive) {
        std::transform(input, input + count, output, [](T x) -> U {
            return std::isinf(static_cast<float>(x));
        });
    } else if (!attributes.detect_negative && attributes.detect_positive) {
        std::transform(input, input + count, output, [](T x) -> U {
            return (static_cast<float>(x) == std::numeric_limits<float>::infinity());
        });
    } else if (attributes.detect_negative && !attributes.detect_positive) {
        std::transform(input, input + count, output, [](T x) -> U {
            return (static_cast<float>(x) == -std::numeric_limits<float>::infinity());
        });
    } else {
        std::memset(output, 0, count);
    }
}

}  // namespace reference
}  // namespace ov
