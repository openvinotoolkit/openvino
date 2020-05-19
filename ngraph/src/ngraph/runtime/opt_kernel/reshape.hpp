//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/axis_vector.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace opt_kernel
        {
            template <typename T>
            void reshape_in0(const T* in,
                             T* out,
                             const Shape& in_shape,
                             const AxisVector& in_axis_order,
                             const Shape& out_shape)
            {
                *out = *in;
            }

            template <typename T>
            void reshape_in1(const T* in,
                             T* out,
                             const Shape& in_shape,
                             const AxisVector& in_axis_order,
                             const Shape& out_shape)
            {
                size_t size[1];
                size_t in_index[1];
                size_t* map_index[1];
                for (size_t i = 0; i < 1; i++)
                {
                    size[i] = in_shape[in_axis_order[i]];
                    map_index[in_axis_order[i]] = &in_index[i];
                }
                for (in_index[0] = 0; in_index[0] < size[0]; ++in_index[0])
                {
                    *out++ = in[*map_index[0]];
                }
            }

            template <typename T>
            void reshape_in2(const T* in,
                             T* out,
                             const Shape& in_shape,
                             const AxisVector& in_axis_order,
                             const Shape& out_shape)
            {
                size_t size[2];
                size_t in_index[2];
                size_t* map_index[2];
                for (size_t i = 0; i < 2; i++)
                {
                    size[i] = in_shape[in_axis_order[i]];
                    map_index[in_axis_order[i]] = &in_index[i];
                }
                for (in_index[0] = 0; in_index[0] < size[0]; ++in_index[0])
                {
                    for (in_index[1] = 0; in_index[1] < size[1]; ++in_index[1])
                    {
                        // clang-format off
                        *out++ = in[*map_index[0] * in_shape[1] +
                                    *map_index[1]];
                        // clang-format on
                    }
                }
            }

            template <typename T>
            void reshape_in3(const T* in,
                             T* out,
                             const Shape& in_shape,
                             const AxisVector& in_axis_order,
                             const Shape& out_shape)
            {
                size_t size[3];
                size_t in_index[3];
                size_t* map_index[3];
                for (size_t i = 0; i < 3; i++)
                {
                    size[i] = in_shape[in_axis_order[i]];
                    map_index[in_axis_order[i]] = &in_index[i];
                }
                for (in_index[0] = 0; in_index[0] < size[0]; ++in_index[0])
                {
                    for (in_index[1] = 0; in_index[1] < size[1]; ++in_index[1])
                    {
                        for (in_index[2] = 0; in_index[2] < size[2]; ++in_index[2])
                        {
                            // clang-format off
                            *out++ = in[*map_index[0] * in_shape[1] * in_shape[2] +
                                        *map_index[1] * in_shape[2] +
                                        *map_index[2]];
                            // clang-format on
                        }
                    }
                }
            }

            template <typename T>
            void reshape_in4(const T* in,
                             T* out,
                             const Shape& in_shape,
                             const AxisVector& in_axis_order,
                             const Shape& out_shape)
            {
                size_t size[4];
                size_t in_index[4];
                size_t* map_index[4];
                for (size_t i = 0; i < 4; i++)
                {
                    size[i] = in_shape[in_axis_order[i]];
                    map_index[in_axis_order[i]] = &in_index[i];
                }
                for (in_index[0] = 0; in_index[0] < size[0]; ++in_index[0])
                {
                    for (in_index[1] = 0; in_index[1] < size[1]; ++in_index[1])
                    {
                        for (in_index[2] = 0; in_index[2] < size[2]; ++in_index[2])
                        {
                            for (in_index[3] = 0; in_index[3] < size[3]; ++in_index[3])
                            {
                                // clang-format off
                                *out++ =
                                    in[*map_index[0] * in_shape[1] * in_shape[2] * in_shape[3] +
                                       *map_index[1] * in_shape[2] * in_shape[3] +
                                       *map_index[2] * in_shape[3] +
                                       *map_index[3]];
                                // clang-format on
                            }
                        }
                    }
                }
            }

            template <typename T>
            void reshape_in5(const T* in,
                             T* out,
                             const Shape& in_shape,
                             const AxisVector& in_axis_order,
                             const Shape& out_shape)
            {
                size_t size[5];
                size_t in_index[5];
                size_t* map_index[5];
                for (size_t i = 0; i < 5; i++)
                {
                    size[i] = in_shape[in_axis_order[i]];
                    map_index[in_axis_order[i]] = &in_index[i];
                }
                for (in_index[0] = 0; in_index[0] < size[0]; ++in_index[0])
                {
                    for (in_index[1] = 0; in_index[1] < size[1]; ++in_index[1])
                    {
                        for (in_index[2] = 0; in_index[2] < size[2]; ++in_index[2])
                        {
                            for (in_index[3] = 0; in_index[3] < size[3]; ++in_index[3])
                            {
                                for (in_index[4] = 0; in_index[4] < size[4]; ++in_index[4])
                                {
                                    // clang-format off
                                    *out++ =
                                        in[*map_index[0] * in_shape[1] * in_shape[2] * in_shape[3] * in_shape[4] +
                                           *map_index[1] * in_shape[2] * in_shape[3] * in_shape[4] +
                                           *map_index[2] * in_shape[3] * in_shape[4] +
                                           *map_index[3] * in_shape[4] +
                                           *map_index[4]];
                                    // clang-format on
                                }
                            }
                        }
                    }
                }
            }

            template <typename T>
            void reshape_in6(const T* in,
                             T* out,
                             const Shape& in_shape,
                             const AxisVector& in_axis_order,
                             const Shape& out_shape)
            {
                size_t size[6];
                size_t in_index[6];
                size_t* map_index[6];
                for (size_t i = 0; i < 6; i++)
                {
                    size[i] = in_shape[in_axis_order[i]];
                    map_index[in_axis_order[i]] = &in_index[i];
                }
                for (in_index[0] = 0; in_index[0] < size[0]; ++in_index[0])
                {
                    for (in_index[1] = 0; in_index[1] < size[1]; ++in_index[1])
                    {
                        for (in_index[2] = 0; in_index[2] < size[2]; ++in_index[2])
                        {
                            for (in_index[3] = 0; in_index[3] < size[3]; ++in_index[3])
                            {
                                for (in_index[4] = 0; in_index[4] < size[4]; ++in_index[4])
                                {
                                    for (in_index[5] = 0; in_index[5] < size[5]; ++in_index[5])
                                    {
                                        // clang-format off
                                        *out++ = in[*map_index[0] * in_shape[1] * in_shape[2] * in_shape[3] * in_shape[4] * in_shape[5] +
                                                    *map_index[1] * in_shape[2] * in_shape[3] * in_shape[4] * in_shape[5] +
                                                    *map_index[2] * in_shape[3] * in_shape[4] * in_shape[5] +
                                                    *map_index[3] * in_shape[4] * in_shape[5] +
                                                    *map_index[4] * in_shape[5] +
                                                    *map_index[5]];
                                        // clang-format on
                                    }
                                }
                            }
                        }
                    }
                }
            }
            template <typename T>
            void reshape(const T* in,
                         T* out,
                         const Shape& in_shape,
                         const AxisVector& in_axis_order,
                         const Shape& out_shape)
            {
                switch (in_shape.size())
                {
                case 0: reshape_in0<T>(in, out, in_shape, in_axis_order, out_shape); break;
                case 1: reshape_in1<T>(in, out, in_shape, in_axis_order, out_shape); break;
                case 2: reshape_in2<T>(in, out, in_shape, in_axis_order, out_shape); break;
                case 3: reshape_in3<T>(in, out, in_shape, in_axis_order, out_shape); break;
                case 4: reshape_in4<T>(in, out, in_shape, in_axis_order, out_shape); break;
                case 5: reshape_in5<T>(in, out, in_shape, in_axis_order, out_shape); break;
                case 6: reshape_in6<T>(in, out, in_shape, in_axis_order, out_shape); break;
                default: reference::reshape(in, out, in_shape, in_axis_order, out_shape); break;
                }
            }
        }
    }
}
