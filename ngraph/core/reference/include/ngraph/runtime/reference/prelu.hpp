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
            inline Shape broadcast_shape(const Shape& arg_shape, const Shape& slope_shape)
            {
                auto new_shape = slope_shape;
                if (slope_shape.size() > arg_shape.size())
                {
                    const auto found = std::search(++begin(arg_shape),
                                                   end(arg_shape),
                                                   begin(slope_shape),
                                                   end(slope_shape),
                                                   [](size_t s1, size_t s2)
                                                   { return s1 == s2 || s1 == 1 || s2 == 1; });
                    NGRAPH_CHECK(found != end(arg_shape), "something is wrong");
                    const auto axis_diff =
                        std::distance(std::next(found, slope_shape.size()), end(arg_shape));
                    NGRAPH_CHECK(axis_diff >= 0, "something is wrong");
                    new_shape.insert(new_shape.end(), axis_diff, 1);
                }
                else if (slope_shape.size() == 1 && slope_shape.front() > 1)
                {
                    if (arg_shape.size() < 2)
                    {
                        new_shape.insert(new_shape.end(), 2, 1);
                    }
                    else
                    {
                        const auto found = std::find(
                            std::next(begin(arg_shape), 1), end(arg_shape), slope_shape.front());
                        NGRAPH_CHECK(found != end(arg_shape), "something is wrong");
                        const auto axis_diff =
                            std::distance(std::next(found, slope_shape.size()), end(arg_shape));
                        NGRAPH_CHECK(axis_diff >= 0, "something is wrong");
                        new_shape.insert(new_shape.end(), axis_diff, 1);
                    }
                }
                return new_shape;
            }
            template <typename T>
            void prelu(const T* arg,
                       const T* slope,
                       T* out,
                       const Shape& arg_shape,
                       const Shape& slope_shape,
                       const Shape& out_shape)
            {
                const auto broadcastable_slope_shape = broadcast_shape(arg_shape, slope_shape);
                autobroadcast_binop(arg,
                                    slope,
                                    out,
                                    arg_shape,
                                    broadcastable_slope_shape,
                                    op::AutoBroadcastType::NUMPY,
                                    [](T x, T y) -> T { return x < T(0) ? T(x * y) : x; });
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
