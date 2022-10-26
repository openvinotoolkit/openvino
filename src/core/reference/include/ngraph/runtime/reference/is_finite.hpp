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
void is_finite(const T* input, U* output, size_t count) {
    std::transform(input, input + count, output, [](T x) -> U {
        return std::isfinite(x);
    });
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
