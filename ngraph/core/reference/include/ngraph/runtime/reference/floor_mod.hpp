// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

#include "ngraph/runtime/reference/autobroadcast_binop.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void floor_mod(const T* arg0,
                           const T* arg1,
                           T* out,
                           const Shape& arg0_shape,
                           const Shape& arg1_shape,
                           const op::AutoBroadcastSpec& broadcast_spec)
            {
                autobroadcast_binop(
                    arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec, [](T x, T y) -> T {
                        // Cast to double is needed for integer input,
                        // otherwise std::floor will act like std::trunc
                        const double divisor = static_cast<double>(y);
                        return x - y * std::floor(x / divisor);
                    });
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
