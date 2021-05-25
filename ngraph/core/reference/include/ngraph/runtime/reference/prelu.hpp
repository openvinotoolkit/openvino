//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

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
