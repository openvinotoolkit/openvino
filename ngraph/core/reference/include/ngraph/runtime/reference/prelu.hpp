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
                if (arg_shape.size() >= 3) {
                    broadcasted_slope.insert(
                    broadcasted_slope.end(), arg_shape.size() - 2, 1);
                }
                
                autobroadcast_binop(arg,
                                    slope,
                                    out,
                                    arg_shape,
                                    broadcasted_slope,
                                    ngraph::op::AutoBroadcastType::NUMPY,
                                    [](T x, T y) -> T { return x < T(0) ? T(x * y) : x; });
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
