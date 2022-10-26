// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

namespace ngraph {
namespace runtime {
namespace reference {
template <typename T, typename U>
void is_inf(const T* input, U* output, size_t count, const ov::op::v10::IsInf::Attributes& attributes) {
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
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
