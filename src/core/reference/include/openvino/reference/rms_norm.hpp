// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>

#include "openvino/reference/add.hpp"
#include "openvino/reference/convert.hpp"
#include "openvino/reference/divide.hpp"
#include "openvino/reference/multiply.hpp"
#include "openvino/reference/power.hpp"
#include "openvino/reference/reduce_mean.hpp"
#include "openvino/reference/sqrt.hpp"

namespace ov {
namespace reference {
/**
 * @brief Reference implementation of RMSNorm operator.
 *
 *  Math Formula: (x / Sqrt(ReduceMean(x^2, axes) + eps))
 *
 * @param in           Input pointer to data.
 * @param axes         Axes for reduce mean calculation
 * @param out          Output pointer to results.
 * @param in_shape     Shape of the input Tensor
 * @param eps          Epsilon for not dividing by zero while normalizing the value
 *
 */
template <class T>
void rms_norm(const T* in, const AxisSet& axes, T* out, const Shape& in_shape, double eps) {
    {
        const auto in_elem_count = shape_size(in_shape);
        const std::vector<T> deg_value(in_elem_count, 2);
        power(in, deg_value.data(), out, in_elem_count);
    }
    const auto reduced_shape = util::reduce_keep_dims(in_shape, axes);
    const auto reduced_elements_count = shape_size(reduced_shape);
    std::vector<T> root_mean_square(reduced_elements_count);
    reduce_mean(out, root_mean_square.data(), in_shape, axes);
    {
        const std::vector<T> eps_broadcasted(reduced_elements_count, static_cast<T>(eps));
        add(root_mean_square.data(), eps_broadcasted.data(), root_mean_square.data(), reduced_elements_count);
    }
    sqrt(root_mean_square.data(), root_mean_square.data(), root_mean_square.size());
    divide(in, root_mean_square.data(), out, in_shape, reduced_shape, op::AutoBroadcastType::NUMPY, false);
}

/**
 * @brief Reference implementation of RMSNorm operator.
 *
 *  Math Formula: (x / Sqrt(ReduceMean(x^2, axes) + eps)) * scale
 *
 * @param in           Input pointer to data.
 * @param axes         Axes for reduce mean calculation
 * @param out          Output pointer to results.
 * @param in_shape     Shape of the input Tensor
 * @param eps          Epsilon for not dividing by zero while normalizing the value
 * @param scale_shape  Shape of the scale Tensor (optional)
 * @param scale        Input pointer to scale (optional).
 *
 */
template <class T>
void rms_norm(const T* in,
              const AxisSet& axes,
              T* out,
              const Shape& in_shape,
              double eps,
              const Shape& scale_shape,
              const T* scale) {
    rms_norm(in, axes, out, in_shape, eps);
    multiply(out, scale, out, in_shape, scale_shape, op::AutoBroadcastType::NUMPY);
}

/**
 * @brief Reference implementation of RMS operator with output type conversion
 *
 *  Math Formula: Convert((x / Sqrt(ReduceMean(x^2, axes) + eps)) * scale), T_OUT)
 *
 * @param in           Input pointer to data
 * @param axes         Axes for reduce mean calculation
 * @param out          Output pointer to results
 * @param in_shape     Shape of the input Tensor
 * @param eps          Epsilon for not dividing by zero while normalizing the value
 * @param scale_shape  Shape of the scale Tensor
 * @param scale        Input pointer to scale
 *
 */
template <class T_IN, class T_OUT>
void rms_norm_mul_convert_out(const T_IN* in,
                              const AxisSet& axes,
                              T_OUT* out,
                              const Shape& in_shape,
                              double eps,
                              const Shape& scale_shape,
                              const T_IN* scale) {
    std::vector<T_IN> tmp_out(shape_size(in_shape));
    rms_norm(in, axes, tmp_out.data(), in_shape, eps, scale_shape, scale);
    convert(tmp_out.data(), out, tmp_out.size());
}

}  // namespace reference
}  // namespace ov
