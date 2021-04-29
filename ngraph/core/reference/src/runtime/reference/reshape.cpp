// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <stdio.h>

#include "ngraph/check.hpp"
#include "ngraph/runtime/reference/reshape.hpp"

using namespace ngraph;

void runtime::reference::reshape(const char* arg,
                                 char* out,
                                 const Shape& in_shape,
                                 const AxisVector& in_axis_order,
                                 const Shape& out_shape,
                                 size_t elem_size)
{
    // Unfortunately we don't yet have a constructor for CoordinateTransform that lets
    // us pass only source_space_shape
    // and source_axis_order so we have to construct the defaults here.
    Shape in_start_corner(in_shape.size(), 0); // (0,...0)
    Strides in_strides(in_shape.size(), 1);    // (1,...,1)

    CoordinateTransform input_transform(
        in_shape, in_start_corner, in_shape, in_strides, in_axis_order);
    CoordinateTransform output_transform(out_shape);

    NGRAPH_CHECK(shape_size(input_transform.get_target_shape()) ==
                 shape_size(output_transform.get_target_shape()));

    CoordinateTransform::Iterator output_it = output_transform.begin();

    for (const Coordinate& input_coord : input_transform)
    {
        if (output_it == output_transform.end())
            break;
        const Coordinate& output_coord = *output_it;

        memcpy(out + output_transform.index(output_coord) * elem_size,
               arg + input_transform.index(input_coord) * elem_size,
               elem_size);

        ++output_it;
    }
}
