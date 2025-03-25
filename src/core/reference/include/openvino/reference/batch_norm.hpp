// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
template <typename T>
static inline T norm(T val, T mean, T var, T eps) {
    return ((val - mean) / static_cast<T>(std::sqrt(var + eps)));
}

template <typename T>
void batch_norm_inference(float eps,
                          const T* in,
                          const T* gamma,
                          const T* beta,
                          const T* mean,
                          const T* variance,
                          T* out,
                          const Shape& in_shape) {
    auto eps_casted = static_cast<T>(eps);

    size_t in_idx = 0;
    const CoordinateTransformBasic in_transform{in_shape};
    for (Coordinate in_coord : in_transform) {
        auto ch_num = in_coord[1];
        auto ch_gamma = gamma[ch_num];
        auto ch_beta = beta[ch_num];
        auto ch_mean = mean[ch_num];
        auto ch_var = variance[ch_num];

        auto normalized = norm(in[in_idx], ch_mean, ch_var, eps_casted);
        out[in_idx] = normalized * ch_gamma + ch_beta;
        in_idx++;
    }
}
}  // namespace reference
}  // namespace ov
