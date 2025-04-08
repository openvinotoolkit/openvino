// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
template <typename T, typename B, typename P>
void bucketize(const T* data,
               const B* buckets,
               P* out,
               const Shape& data_shape,
               const Shape& buckets_shape,
               bool with_right_bound) {
    size_t data_size = shape_size(data_shape);
    size_t buckets_size = shape_size(buckets_shape);

    // if buckets is empty, bucket index for all elements
    // in output is equal to 0
    if (buckets_size == 0) {
        std::fill_n(out, data_size, static_cast<P>(0));
        return;
    }

    for (size_t i = 0; i < data_size; i++) {
        const T val = data[i];
        const B* bound = nullptr;

        bound = with_right_bound ? std::lower_bound(buckets, buckets + buckets_size, val)
                                 : std::upper_bound(buckets, buckets + buckets_size, val);

        out[i] = static_cast<P>(bound - buckets);
    }
}

}  // namespace reference
}  // namespace ov
