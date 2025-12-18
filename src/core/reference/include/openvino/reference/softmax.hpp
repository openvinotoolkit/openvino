// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <execution>
#include <vector>

#include "openvino/core/shape_util.hpp"
#include "openvino/reference/reduce_max.hpp"
#include "openvino/reference/reduce_sum.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
template <typename T>
void softmax(const T* arg, T* out, const Shape& shape, const AxisSet& axes) {
    const auto temp_shape = util::reduce_keep_dims(shape, axes);
    const auto temp_elements = shape_size(temp_shape);
    auto temp_storage = std::vector<T>(temp_elements);
    const auto temp_ptr = temp_storage.data();

    reduce_max(arg, temp_ptr, shape, axes);

    const CoordinateTransformBasic transform{shape};
    for (const auto& coord : transform) {
        const Coordinate temp_coord = util::reduce_keep_dims(coord, axes);
        const auto out_index = coordinate_index(coord, shape);
        const auto temp_index = coordinate_index(temp_coord, temp_shape);
        out[out_index] = std::exp(arg[out_index] - temp_ptr[temp_index]);
    }

    reduce_sum(out, temp_ptr, shape, axes);

    for (const auto& coord : transform) {
        const Coordinate temp_coord = util::reduce_keep_dims(coord, axes);
        const auto out_index = coordinate_index(coord, shape);
        const auto temp_index = coordinate_index(temp_coord, temp_shape);
        out[out_index] /= temp_ptr[temp_index];
    }
}

/**
 * @brief Optimized implementation of Softmax operator for reduction over the last dimension.
 *
 * @param arg   Input pointer to data.
 * @param out   Output pointer to results.
 * @param shape Input shape.
 * @param axes  Axes on which softmax is applied.
 */
template <typename T>
void softmax_lastdim(const T* arg, T* out, const Shape& shape, const AxisSet& axes) {
    // Assume reduction is always over the last dimension
    const size_t last_dim = shape.back();
    const size_t outer_size = shape_size(shape) / last_dim;

    std::vector<T> max_values(outer_size, std::numeric_limits<T>::lowest());
    std::vector<T> sum_values(outer_size, T{0});

    // Compute max for each outer dimension
    for (size_t i = 0; i < outer_size; ++i) {
        for (size_t j = 0; j < last_dim; ++j) {
            const size_t idx = i * last_dim + j;
            max_values[i] = std::max(max_values[i], arg[idx]);
        }
    }

    // Compute exponentials and sum for normalization
    for (size_t i = 0; i < outer_size; ++i) {
        for (size_t j = 0; j < last_dim; ++j) {
            const size_t idx = i * last_dim + j;
            out[idx] = std::exp(arg[idx] - max_values[i]);
            sum_values[i] += out[idx];
        }
    }

    // Normalize the results
    for (size_t i = 0; i < outer_size; ++i) {
        for (size_t j = 0; j < last_dim; ++j) {
            const size_t idx = i * last_dim + j;
            out[idx] /= sum_values[i];
        }
    }
}

/**
 * @brief Optimized implementation of Softmax operator for reduction over the last dimension using parallelism.
 *
 * @param arg   Input pointer to data.
 * @param out   Output pointer to results.
 * @param shape Input shape.
 * @param axes  Axes on which softmax is applied.
 */
template <typename T>
void softmax_lastdim_par(const T* arg, T* out, const Shape& shape, const AxisSet& axes) {
    // Assume reduction is always over the last dimension
    const size_t last_dim = shape.back();
    const size_t outer_size = shape_size(shape) / last_dim;

    std::vector<T> max_values(outer_size, std::numeric_limits<T>::lowest());
    std::vector<T> sum_values(outer_size, T{0});

    // Compute max for each outer dimension in parallel
    std::for_each(std::execution::par, max_values.begin(), max_values.end(), [&](T& max_val) {
        size_t i = &max_val - max_values.data();
        for (size_t j = 0; j < last_dim; ++j) {
            const size_t idx = i * last_dim + j;
            max_val = std::max(max_val, arg[idx]);
        }
    });

    // Compute exponentials and sum for normalization in parallel
    std::for_each(std::execution::par, sum_values.begin(), sum_values.end(), [&](T& sum_val) {
        size_t i = &sum_val - sum_values.data();
        for (size_t j = 0; j < last_dim; ++j) {
            const size_t idx = i * last_dim + j;
            out[idx] = std::exp(arg[idx] - max_values[i]);
            sum_val += out[idx];
        }
    });

    // Normalize the results in parallel
    std::for_each(std::execution::par, out, out + shape_size(shape), [&](T& val) {
        size_t idx = &val - out;
        size_t i = idx / last_dim;
        val /= sum_values[i];
    });
}

}  // namespace reference
}  // namespace ov
