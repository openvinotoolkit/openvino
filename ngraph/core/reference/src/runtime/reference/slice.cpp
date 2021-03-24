// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/slice.hpp"

#include <cstring>

#include "ngraph/check.hpp"
#include "ngraph/coordinate_range.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void slice(const char* arg,
                       char* out,
                       const Shape& arg_shape,
                       const Coordinate& lower_bounds,
                       const Coordinate& upper_bounds,
                       const Strides& strides,
                       const Shape& out_shape,
                       size_t elem_size)
            {
                const CoordinateTransform input_transform(
                    arg_shape, lower_bounds, upper_bounds, strides);

                const CoordinateTransform output_transform(out_shape);

                NGRAPH_CHECK(shape_size(input_transform.get_target_shape()) ==
                             shape_size(output_transform.get_target_shape()));

                auto dst_mem = out;

                for (auto range :
                     coordinates::slice(arg_shape, lower_bounds, upper_bounds, strides))
                {
                    auto src_index = range.begin_index;
                    for (size_t i = 0; i < range.element_number; src_index += range.step, ++i)
                    {
                        const auto src_mem = arg + src_index * elem_size;
                        std::memcpy(dst_mem, src_mem, elem_size);
                        std::advance(dst_mem, elem_size);
                    }
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
