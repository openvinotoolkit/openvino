// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/reference/max.hpp"
#include "ngraph/runtime/reference/sum.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void softmax(const T* arg, T* out, const Shape& shape, const AxisSet& axes)
            {
                auto temp_shape = reduce(shape, axes, true);
                auto temp_elements = shape_size(temp_shape);
                auto temp_ptr = new T[temp_elements];

                max(arg, temp_ptr, shape, axes, true);

                CoordinateTransform transform(shape);
                CoordinateTransform temp_transform(temp_shape);
                for (const Coordinate& coord : transform)
                {
                    Coordinate temp_coord = reduce(coord, axes, true);
                    out[transform.index(coord)] = std::exp(
                        arg[transform.index(coord)] - temp_ptr[temp_transform.index(temp_coord)]);
                }

                sum(out, temp_ptr, shape, axes, true);

                for (const Coordinate& coord : transform)
                {
                    Coordinate temp_coord = reduce(coord, axes, true);
                    out[transform.index(coord)] /= temp_ptr[temp_transform.index(temp_coord)];
                }

                delete[] temp_ptr;
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
