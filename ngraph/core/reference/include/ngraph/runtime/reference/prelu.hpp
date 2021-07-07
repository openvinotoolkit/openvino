// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <ngraph/op/util/attr_types.hpp>
#include <ngraph/shape.hpp>

#include "ngraph/runtime/reference/autobroadcast_binop.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void prelu(const T* arg,
                       const T* slope,
                       T* out,
                       const Shape& arg_shape,
                       const Shape& slope_shape)
            {
                Shape slope_shape_tmp = slope_shape;
                const auto channel_dim_idx = arg_shape.size() > 1 ? 1 : 0;
                if (slope_shape.size() == 1 && arg_shape[channel_dim_idx] == slope_shape[0])
                {
                    Shape channel_slope_shape(arg_shape.size(), 1);
                    channel_slope_shape[channel_dim_idx] = slope_shape[0];
                    std::swap(slope_shape_tmp, channel_slope_shape);
                }
                autobroadcast_binop(arg,
                                    slope,
                                    out,
                                    arg_shape,
                                    slope_shape_tmp,
                                    ngraph::op::AutoBroadcastType::NUMPY,
                                    [](T x, T y) -> T { return x < T(0) ? T(x * y) : x; });
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
