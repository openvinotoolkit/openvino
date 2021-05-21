// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void reduce_l2(const T* arg,
                           T* out,
                           const Shape& in_shape,
                           const AxisSet& reduction_axes,
                           bool keep_dims)
            {
                auto out_shape = reduce(in_shape, reduction_axes, keep_dims);
                CoordinateTransform output_transform(out_shape);

                for (const Coordinate& output_coord : output_transform)
                {
                    out[output_transform.index(output_coord)] = 0;
                }

                CoordinateTransform input_transform(in_shape);

                for (const Coordinate& input_coord : input_transform)
                {
                    Coordinate output_coord = reduce(input_coord, reduction_axes, keep_dims);

                    size_t output_index = output_transform.index(output_coord);

                    out[output_index] =
                        out[output_index] + arg[input_transform.index(input_coord)] *
                                                arg[input_transform.index(input_coord)];
                }
                for (const Coordinate& output_coord : output_transform)
                {
                    out[output_transform.index(output_coord)] =
                        sqrt(out[output_transform.index(output_coord)]);
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
