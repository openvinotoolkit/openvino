// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <iostream>

namespace ngraph {
namespace runtime {
namespace reference {
template <typename T>
size_t range_search(const T& arg, const double* knots, size_t knots_number) {
    OPENVINO_ASSERT(knots_number >= 2, "The number of knots is less than 2.");
    size_t left = 0;
    size_t right = knots_number - 1;
    size_t mid = (right - left) / 2;
    while (left < right && (arg < knots[mid] || knots[mid + 1] < arg)) {
        if (arg < knots[mid]) {
            right = mid;
        } else {
            left = mid;
        }

        mid = left + (right - left) / 2;
    }

    return mid;
}


template <typename T>
void pwl(const T* args, T* out, size_t count, const double* m, const double* b, size_t segments_number,
    const double* knots, size_t knots_number) {
    OPENVINO_ASSERT(segments_number + 1 == knots_number, "The number of knots have to be equal the number of segments plus one.");
    for (size_t i = 0; i < count; i++) {
        size_t segment_index = 0;
        if (args[i] < knots[0]) {
            ;
        } else if (knots[knots_number - 1] < args[i]) {
            segment_index = segments_number - 1;
        } else {
            auto index = range_search(args[i], knots, knots_number);
            segment_index = index == knots_number - 1 ? index - 1 : index;
        }
            
        out[i] = m[segment_index] * args[i] + b[segment_index];
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
