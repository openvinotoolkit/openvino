// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <limits>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void max(const T* arg,
                     T* out,
                     const Shape& in_shape,
                     const AxisSet& reduction_axes,
                     bool keep_dims)
            {
                T minval = std::numeric_limits<T>::has_infinity
                               ? T(-std::numeric_limits<T>::infinity())
                               : std::numeric_limits<T>::min();

                auto out_shape = reduce(in_shape, reduction_axes, keep_dims);
                CoordinateTransform output_transform(out_shape);

                for (const Coordinate& output_coord : output_transform)
                {
                    out[output_transform.index(output_coord)] = minval;
                }

                CoordinateTransform input_transform(in_shape);

                for (const Coordinate& input_coord : input_transform)
                {
                    Coordinate output_coord = reduce(input_coord, reduction_axes, keep_dims);

                    T x = arg[input_transform.index(input_coord)];
                    T max = out[output_transform.index(output_coord)];
                    if (x > max)
                    {
                        out[output_transform.index(output_coord)] = x;
                    }
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
