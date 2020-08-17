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

#include <cmath>
#include <utility>

#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/shape_util.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace opt_kernel
        {
            template <typename T>
            void broadcast_2d(
                const T* in, T* out, const Shape& in_shape, const Shape& out_shape, size_t out_axis)
            {
                size_t index[2];
                size_t& in_index = index[out_axis];
                auto out_strides = row_major_strides(out_shape);
                for (index[0] = 0; index[0] < out_shape[0]; ++index[0])
                {
                    for (index[1] = 0; index[1] < out_shape[1]; ++index[1])
                    {
                        // clang-format off
                        out[index[0] * out_strides[0] +
                            index[1]] =
                                in[in_index];
                        // clang-format on
                    }
                }
            }

            // #define PARALLEL
            template <typename T>
            void broadcast_3d(
                const T* in, T* out, const Shape& in_shape, const Shape& out_shape, size_t out_axis)
            {
                size_t index[3];
                size_t& in_index = index[out_axis];
                auto out_strides = row_major_strides(out_shape);
                for (index[0] = 0; index[0] < out_shape[0]; ++index[0])
                {
                    for (index[1] = 0; index[1] < out_shape[1]; ++index[1])
                    {
                        for (index[2] = 0; index[2] < out_shape[2]; ++index[2])
                        {
                            // clang-format off
                            out[index[0] * out_strides[0] +
                                index[1] * out_strides[1] +
                                index[2]] =
                                    in[in_index];
                            // clang-format on
                        }
                    }
                }
            }

            template <typename T>
            void broadcast_4d(
                const T* in, T* out, const Shape& in_shape, const Shape& out_shape, size_t out_axis)
            {
                size_t index[4];
                size_t& in_index = index[out_axis];
                auto out_strides = row_major_strides(out_shape);
                for (index[0] = 0; index[0] < out_shape[0]; ++index[0])
                {
                    for (index[1] = 0; index[1] < out_shape[1]; ++index[1])
                    {
                        for (index[2] = 0; index[2] < out_shape[2]; ++index[2])
                        {
                            for (index[3] = 0; index[3] < out_shape[3]; ++index[3])
                            {
                                // clang-format off
                                out[index[0] * out_strides[0] +
                                    index[1] * out_strides[1] +
                                    index[2] * out_strides[2] +
                                    index[3]] =
                                        in[in_index];
                                // clang-format on
                            }
                        }
                    }
                }
            }

            template <typename T>
            void broadcast_5d(
                const T* in, T* out, const Shape& in_shape, const Shape& out_shape, size_t out_axis)
            {
                size_t index[5];
                size_t& in_index = index[out_axis];
                auto out_strides = row_major_strides(out_shape);
                for (index[0] = 0; index[0] < out_shape[0]; ++index[0])
                {
                    for (index[1] = 0; index[1] < out_shape[1]; ++index[1])
                    {
                        for (index[2] = 0; index[2] < out_shape[2]; ++index[2])
                        {
                            for (index[3] = 0; index[3] < out_shape[3]; ++index[3])
                            {
                                for (index[4] = 0; index[4] < out_shape[4]; ++index[4])
                                {
                                    // clang-format off
                                    out[index[0] * out_strides[0] +
                                        index[1] * out_strides[1] +
                                        index[2] * out_strides[2] +
                                        index[3] * out_strides[3] +
                                        index[4]] =
                                            in[in_index];
                                    // clang-format on
                                }
                            }
                        }
                    }
                }
            }

            template <typename T>
            void broadcast_6d(
                const T* in, T* out, const Shape& in_shape, const Shape& out_shape, size_t out_axis)
            {
                size_t index[6];
                size_t& in_index = index[out_axis];
                auto out_strides = row_major_strides(out_shape);
                for (index[0] = 0; index[0] < out_shape[0]; ++index[0])
                {
                    for (index[1] = 0; index[1] < out_shape[1]; ++index[1])
                    {
                        for (index[2] = 0; index[2] < out_shape[2]; ++index[2])
                        {
                            for (index[3] = 0; index[3] < out_shape[3]; ++index[3])
                            {
                                for (index[4] = 0; index[4] < out_shape[4]; ++index[4])
                                {
                                    for (index[5] = 0; index[5] < out_shape[5]; ++index[5])
                                    {
                                        // clang-format off
                                        out[index[0] * out_strides[0] +
                                            index[1] * out_strides[1] +
                                            index[2] * out_strides[2] +
                                            index[3] * out_strides[3] +
                                            index[4] * out_strides[4] +
                                            index[5]] =
                                                in[in_index];
                                        // clang-format on
                                    }
                                }
                            }
                        }
                    }
                }
            }

            template <typename T>
            void broadcast(const T* in,
                           T* out,
                           const Shape& in_shape,
                           const Shape& out_shape,
                           const AxisSet& broadcast_axes)
            {
                if (is_scalar(in_shape))
                {
                    for (size_t i = 0; i < shape_size(out_shape); ++i)
                    {
                        out[i] = in[0];
                    }
                }
                else if (in_shape.size() == 1)
                {
                    size_t output_axis = 0;
                    for (size_t i = 0; i < out_shape.size(); i++)
                    {
                        if (broadcast_axes.count(i) == 0)
                        {
                            output_axis = i;
                            break;
                        }
                    }
                    switch (out_shape.size())
                    {
                    case 2: broadcast_2d<T>(in, out, in_shape, out_shape, output_axis); break;
                    case 3: broadcast_3d<T>(in, out, in_shape, out_shape, output_axis); break;
                    case 4: broadcast_4d<T>(in, out, in_shape, out_shape, output_axis); break;
                    case 5: broadcast_5d<T>(in, out, in_shape, out_shape, output_axis); break;
                    case 6: broadcast_6d<T>(in, out, in_shape, out_shape, output_axis); break;
                    default:
                        runtime::reference::broadcast<T>(
                            in, out, in_shape, out_shape, broadcast_axes);
                        break;
                    }
                }
                else
                {
                    runtime::reference::broadcast<T>(in, out, in_shape, out_shape, broadcast_axes);
                }
            }
        }
    }
}
