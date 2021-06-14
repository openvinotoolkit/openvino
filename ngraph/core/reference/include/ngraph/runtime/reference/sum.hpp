// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <numeric>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"
#include "ngraph/type/bfloat16.hpp"
#include "ngraph/type/float16.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            // Windows doesn't seem to like it if we directly use std::isfinite on integer types,
            // so we will roll our own thing here.
            template <typename T>
            typename std::enable_if<std::is_floating_point<T>::value, bool>::type is_finite(T x)
            {
                return std::isfinite(x);
            }

            template <typename T>
            typename std::enable_if<std::is_same<T, bfloat16>::value ||
                                        std::is_same<T, float16>::value,
                                    bool>::type
                is_finite(T x)
            {
                return std::isfinite(static_cast<float>(x));
            }

            template <typename T>
            typename std::enable_if<std::is_integral<T>::value, bool>::type is_finite(T /* x */)
            {
                return true;
            }

            template <typename T>
            void sum(const T* arg, T* out, const Shape& in_shape, const AxisSet& reduction_axes)
            {
                constexpr bool dont_keep_dims_in_output = false;
                const auto out_shape = reduce(in_shape, reduction_axes, dont_keep_dims_in_output);

                std::vector<T> cs(shape_size(out_shape), 0);
                std::fill(out, out + shape_size(out_shape), 0);

                const auto in_strides = row_major_strides(in_shape);
                const auto out_strides = row_major_strides(out_shape);

                CoordinateTransformBasic input_transform(in_shape);
                for (const Coordinate& input_coord : input_transform)
                {
                    const Coordinate output_coord =
                        reduce(input_coord, reduction_axes, dont_keep_dims_in_output);

                    const size_t in_idx = std::inner_product(
                        input_coord.begin(), input_coord.end(), in_strides.begin(), 0);
                    const size_t out_idx = std::inner_product(
                        output_coord.begin(), output_coord.end(), out_strides.begin(), 0);

                    T x = arg[in_idx];
                    T& z = out[out_idx];

                    if (is_finite(x) && is_finite(z))
                    {
                        T& c = cs[out_idx];
                        T t = z + (x - c);
                        c = (t - z) - (x - c);
                        z = t;
                    }
                    else
                    {
                        z = z + x;
                    }
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
