// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

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
            void sum(const T* arg,
                     T* out,
                     const Shape& in_shape,
                     const AxisSet& reduction_axes,
                     bool keep_dims)
            {
                auto out_shape = reduce(in_shape, reduction_axes, keep_dims);
                CoordinateTransform output_transform(out_shape);
                std::vector<T> cs(shape_size(out_shape));

                for (const Coordinate& output_coord : output_transform)
                {
                    out[output_transform.index(output_coord)] = 0;
                    cs[output_transform.index(output_coord)] = 0;
                }

                CoordinateTransform input_transform(in_shape);

                for (const Coordinate& input_coord : input_transform)
                {
                    Coordinate output_coord = reduce(input_coord, reduction_axes, keep_dims);

                    T x = arg[input_transform.index(input_coord)];
                    T& z = out[output_transform.index(output_coord)];

                    if (is_finite(x) && is_finite(z))
                    {
                        T& c = cs[output_transform.index(output_coord)];
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
