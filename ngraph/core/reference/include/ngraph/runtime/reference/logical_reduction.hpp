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
            static inline void reduce_logical_and(const char* arg,
                                                  char* out,
                                                  const Shape& input_shape,
                                                  const AxisSet& reduction_axes,
                                                  bool keep_dims)
            {
                CoordinateTransform output_transform(
                    reduce(input_shape, reduction_axes, keep_dims));

                for (const Coordinate& output_coord : output_transform)
                {
                    out[output_transform.index(output_coord)] = 1;
                }

                CoordinateTransform input_transform(input_shape);

                for (const Coordinate& input_coord : input_transform)
                {
                    Coordinate output_coord = reduce(input_coord, reduction_axes, keep_dims);
                    out[output_transform.index(output_coord)] =
                        out[output_transform.index(output_coord)] &&
                        arg[input_transform.index(input_coord)];
                }
            }

            static inline void reduce_logical_or(const char* arg,
                                                 char* out,
                                                 const Shape& input_shape,
                                                 const AxisSet& reduction_axes,
                                                 bool keep_dims)
            {
                CoordinateTransform output_transform(
                    reduce(input_shape, reduction_axes, keep_dims));

                for (const Coordinate& output_coord : output_transform)
                {
                    out[output_transform.index(output_coord)] = 0;
                }

                CoordinateTransform input_transform(input_shape);

                for (const Coordinate& input_coord : input_transform)
                {
                    Coordinate output_coord = reduce(input_coord, reduction_axes, keep_dims);
                    out[output_transform.index(output_coord)] =
                        out[output_transform.index(output_coord)] ||
                        arg[input_transform.index(input_coord)];
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
