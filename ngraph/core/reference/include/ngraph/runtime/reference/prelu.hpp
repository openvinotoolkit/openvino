// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <ngraph/op/util/attr_types.hpp>
#include <ngraph/shape.hpp>
#include <numeric>

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
                Shape broadcasted_slope(slope_shape.begin(), slope_shape.end());
                broadcasted_slope.insert(
                    broadcasted_slope.begin(), arg_shape.size() - broadcasted_slope.size(), 1);

                for (int i = arg_shape.size() - 1; i >= 0; i--)
                {
                    NGRAPH_CHECK(broadcasted_slope[i] == arg_shape[i] || broadcasted_slope[i] == 1,
                                 "Invalid slope shape");
                }
                NGRAPH_CHECK(shape_size(arg_shape) >= shape_size(slope_shape),
                             "Slope shape has to be equal or smaller than first input shape");
                autobroadcast_binop(arg,
                                    slope,
                                    out,
                                    arg_shape,
                                    slope_shape,
                                    ngraph::op::AutoBroadcastType::NUMPY,
                                    [](T x, T y) -> T { return x < T(0) ? T(x * y) : x; });
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
