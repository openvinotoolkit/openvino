// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <iterator>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

#include "ngraph/runtime/reference/transpose.hpp"
#include "utils/span.hpp"

namespace ngraph {
namespace runtime {
namespace reference {

template <typename T, typename P>
void cumsum(const T* arg,
            const P* axis_tensor,
            T* out,
            const Shape& tensor_shape,
            const bool exclusive,
            const bool reverse) {
    const auto rank = tensor_shape.size();
    const auto axis = axis_tensor[0] >= 0 ? axis_tensor[0] : rank + axis_tensor[0];

    const auto slices_count = shape_size(Shape(tensor_shape.begin(), tensor_shape.begin() + axis));
    const auto offset = shape_size(Shape(tensor_shape.begin() + axis + 1, tensor_shape.end()));

    const auto axis_dim = tensor_shape[axis];
    for (auto i = 0; i < slices_count; ++i) {
        auto shift = exclusive ? offset : 0;

        for (auto o = 0; o < offset; ++o) {
            auto sequence_start_idx = i * axis_dim * offset + o;
            out[sequence_start_idx] = exclusive ? static_cast<T>(0) : arg[sequence_start_idx];
            for (auto j = 1; j < axis_dim; ++j) {
                auto element_idx = sequence_start_idx + j * offset;
                auto in_idx = element_idx - shift;
                out[element_idx] = out[element_idx - offset] + arg[in_idx];
            }
        }
    }
}

}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
