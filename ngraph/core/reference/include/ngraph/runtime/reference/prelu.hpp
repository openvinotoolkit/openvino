// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <ngraph/runtime/reference/autobroadcast_binop.hpp>
#include <ngraph/shape.hpp>

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            bool shapes_match(const Shape& input, const Shape& output)
            {
                if (input.size() > output.size())
                {
                    return false;
                }
                const auto matching_size = [](size_t in_size, size_t out_size) {
                    return in_size == out_size || in_size == 1;
                };
                return std::equal(input.rbegin(), input.rend(), output.rbegin(), matching_size);
            }

            template <typename T>
            void prelu(const T* arg,
                       const T* slope,
                       T* out,
                       const Shape& arg_shape,
                       const Shape& slope_shape,
                       const Shape& out_shape)
            {
                NGRAPH_CHECK(shapes_match(arg_shape, out_shape) &&
                                 shapes_match(slope_shape, out_shape),
                             "PReLU has invalid input/output dims configuration.");

                autobroadcast_binop(arg,
                                    slope,
                                    out,
                                    arg_shape,
                                    slope_shape,
                                    op::AutoBroadcastType::NUMPY,
                                    [](T x, T y) -> T { return x < T(0) ? T(x * y) : x; });
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
