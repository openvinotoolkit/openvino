// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <iostream>

namespace ov {
namespace intel_gna {
namespace op {
namespace reference {
template <typename T, typename A>
size_t range_search(const T& arg, const A* knots, size_t knots_number) {
    if (arg < knots[0])
        return 0;

    OPENVINO_ASSERT(knots_number >= 2, "The number of knots is less than 2.");
    if (knots[knots_number - 1] < arg)
        return knots_number - 2;

    size_t left = 0;
    size_t right = knots_number - 2;
    size_t mid = (right - left) / 2;
    while (left < right && (arg < knots[mid] || knots[mid + 1] < arg)) {
        if (arg < knots[mid]) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }

        mid = left + (right - left) / 2;
    }

    return mid;
}

template <typename T, typename A>
void pwl(const T* args, T* out, size_t count, const A* m, const A* b, const A* knots, size_t segments_number) {
    for (size_t i = 0; i < count; i++) {
        // knots is one more than segments
        size_t segment_index = range_search(args[i], knots, segments_number + 1);
        out[i] = static_cast<T>(m[segment_index] * args[i] + b[segment_index]);
    }
}
}  // namespace reference
}  // namespace op
}  // namespace intel_gna
}  // namespace ov
