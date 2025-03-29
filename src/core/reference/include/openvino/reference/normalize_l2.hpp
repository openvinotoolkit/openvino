// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/reference/autobroadcast_binop.hpp"
#include "openvino/reference/reduce_sum.hpp"

namespace ov {
namespace reference {
template <typename T>
void normalize_l2(const T* data,
                  T* out,
                  const Shape& data_shape,
                  const AxisSet& reduction_axes,
                  float eps,
                  op::EpsMode eps_mode) {
    if (reduction_axes.empty()) {
        // When axes is an empty list, then each `data` element is divided by itself
        // resulting value 1 for all non-zero elements
        for (size_t i = 0; i < shape_size(data_shape); ++i) {
            out[i] = data[i] == 0 ? T(0) : T(1);
        }
        return;
    }

    std::vector<T> sqr_data(shape_size(data_shape));
    for (size_t i = 0; i < shape_size(data_shape); ++i) {
        sqr_data[i] = data[i] * data[i];
    }

    Shape reduce_shape = data_shape;
    for (auto axis : reduction_axes) {
        reduce_shape[axis] = 1;
    }

    std::vector<T> sum_data(shape_size(reduce_shape));
    reduce_sum(sqr_data.data(), sum_data.data(), data_shape, reduction_axes);
    autobroadcast_binop(data,
                        sum_data.data(),
                        out,
                        data_shape,
                        reduce_shape,
                        op::AutoBroadcastSpec(op::AutoBroadcastType::NUMPY),
                        [&eps, &eps_mode](T x, T y) -> T {
                            T arg = (eps_mode == op::EpsMode::ADD) ? y + static_cast<T>(eps)
                                                                   : std::max(y, static_cast<T>(eps));
                            return x / static_cast<T>(std::sqrt(arg));
                        });
}
}  // namespace reference
}  // namespace ov
