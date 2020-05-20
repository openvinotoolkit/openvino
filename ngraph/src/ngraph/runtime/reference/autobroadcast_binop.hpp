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

#include <cstddef>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            /// \brief Helper function to implement autobroadcasting elementwise binop references.
            ///
            /// \tparam T Element type of the input tensors.
            /// \tparam U Element type of the output tensor.
            /// \tparam Functor Type of the functor for the elementwise operation. Must support
            ///                 operator()(T,T), and operator()(T,T) must return a value of type
            ///                 U.
            ///
            /// \param arg0 Pointer to the buffer for left operand input tensor.
            /// \param arg1 Pointer to the buffer for right operand input tensor.
            /// \param out Pointer to the buffer for output tensor. This must be pre-allocated by
            ///            the caller, and must be large enough to hold a tensor of the correct
            ///            shape.
            /// \param broadcast_spec Specification of the auto-broadcasting scheme.
            /// \param elementwise_functor Functor implementing the elementwise operation to be
            ///                            applied across the input tensors. Must accept two
            ///                            arguments of type T, and return a value of type U.
            template <typename T, typename U, typename Functor>
            void autobroadcast_binop(const T* arg0,
                                     const T* arg1,
                                     U* out,
                                     const Shape& arg0_shape,
                                     const Shape& arg1_shape,
                                     const op::AutoBroadcastSpec& broadcast_spec,
                                     Functor elementwise_functor)
            {
                switch (broadcast_spec.m_type)
                {
                case op::AutoBroadcastType::NONE:
                    for (size_t i = 0; i < shape_size(arg0_shape); i++)
                    {
                        out[i] = elementwise_functor(arg0[i], arg1[i]);
                    }
                    break;
                case op::AutoBroadcastType::NUMPY:
                    // We'll be using CoordinateTransform to handle the broadcasting. The general
                    // procedure is as follows:
                    //
                    // (1) Left pad the shorter of the two shapes with ones.
                    // (2) Squeeze (remove ones from) both shapes, and record the squeezed axis
                    //     indices.
                    // (3) Using CoordinateTransform, broadcast both args to the final output
                    //     shape. The "broadcasted axes" will be those that were squeezed in step
                    //     2.
                    //
                    // Example:
                    //
                    //    Input shape->Padded shape->Squeezed Shape/Squeezed Axes
                    //    -----------  ------------  ----------------------------
                    // a: [ 3, 2, 1]   [ 3, 2, 1]    [ 3, 2   ]     {2}
                    // b: [    1, 6]   [ 1, 1, 6]    [       6]     {0,1}
                    //                   |  |  |
                    //                   v  v  v
                    //                 Output shape
                    //                 ------------
                    //                 [ 3, 2, 6]
                    {
                        Shape arg0_padded_shape = arg0_shape;
                        Shape arg1_padded_shape = arg1_shape;

                        while (arg0_padded_shape.size() < arg1_padded_shape.size())
                        {
                            arg0_padded_shape.insert(arg0_padded_shape.begin(), 1);
                        }

                        while (arg1_padded_shape.size() < arg0_padded_shape.size())
                        {
                            arg1_padded_shape.insert(arg1_padded_shape.begin(), 1);
                        }

                        Shape arg0_squeezed_shape;
                        Shape arg1_squeezed_shape;
                        AxisSet arg0_squeezed_axes;
                        AxisSet arg1_squeezed_axes;
                        Shape output_shape;

                        for (size_t i = 0; i < arg0_padded_shape.size(); i++)
                        {
                            if (arg0_padded_shape[i] == 1)
                            {
                                arg0_squeezed_axes.insert(i);
                            }
                            else
                            {
                                arg0_squeezed_shape.push_back(arg0_padded_shape[i]);
                            }

                            if (arg1_padded_shape[i] == 1)
                            {
                                arg1_squeezed_axes.insert(i);
                            }
                            else
                            {
                                arg1_squeezed_shape.push_back(arg1_padded_shape[i]);
                            }

                            output_shape.push_back(arg0_padded_shape[i] == 1
                                                       ? arg1_padded_shape[i]
                                                       : arg0_padded_shape[i]);
                        }

                        CoordinateTransform arg0_transform(arg0_squeezed_shape);
                        CoordinateTransform arg1_transform(arg1_squeezed_shape);
                        CoordinateTransform output_transform(output_shape);

                        for (const Coordinate& output_coord : output_transform)
                        {
                            Coordinate arg0_coord = reduce(output_coord, arg0_squeezed_axes);
                            Coordinate arg1_coord = reduce(output_coord, arg1_squeezed_axes);
                            out[output_transform.index(output_coord)] =
                                elementwise_functor(arg0[arg0_transform.index(arg0_coord)],
                                                    arg1[arg1_transform.index(arg1_coord)]);
                        }
                    }
                    break;
                case op::AutoBroadcastType::PDPD:
                    // We'll be using CoordinateTransform to handle the broadcasting. No need to
                    // process arg0 and output shape will be the same as arg0. We need to process
                    // arg1 and the general procedure is as follows:
                    //
                    // (1) Trim trailing ones from arg1 shape.
                    // (2) Left and right pad arg1 to match arg0 shape. Axis is the index start
                    //     to align between arg0 and arg1.
                    // (3) Squeeze (remove ones from) arg1 shape, and record the squeezed axis
                    //     indices.
                    // (3) Using CoordinateTransform, broadcast arg1 to the final output
                    //     shape. The "broadcasted axes" will be those that were squeezed in step
                    //     23.
                    //
                    // Example:
                    //
                    //    Input shape->   Padded shape->   Squeezed Shape/Squeezed Axes
                    //    -----------  ------------  ----------------------------
                    // a: [ 3, 4, 5, 6]   [ 3, 4, 5, 6]    [ 3, 4, 5, 6]
                    // b: [    4, 5,  ]   [ 1, 4, 5, 1]    [    4, 5   ]     {0,3}
                    //                      |  |  |
                    //                      v  v  v
                    //                     Output shape
                    //                     ------------
                    //                    [ 3, 4, 5, 6]
                    {
                        int64_t axis = broadcast_spec.m_axis;
                        if (axis == -1)
                        {
                            axis = arg0_shape.size() - arg1_shape.size();
                        }

                        Shape arg1_padded_shape = arg1_shape;
                        // Trim trailing ones
                        while (arg1_padded_shape.size() > 0 && arg1_padded_shape.back() == 1)
                        {
                            arg1_padded_shape.pop_back();
                        }

                        for (int64_t i = 0; i < axis; ++i)
                        {
                            arg1_padded_shape.insert(arg1_padded_shape.begin(), 1);
                        }

                        while (arg1_padded_shape.size() < arg0_shape.size())
                        {
                            arg1_padded_shape.insert(arg1_padded_shape.end(), 1);
                        }

                        Shape arg1_squeezed_shape;
                        AxisSet arg1_squeezed_axes;

                        for (size_t i = 0; i < arg0_shape.size(); i++)
                        {
                            if (arg1_padded_shape[i] == 1)
                            {
                                arg1_squeezed_axes.insert(i);
                            }
                            else
                            {
                                arg1_squeezed_shape.push_back(arg1_padded_shape[i]);
                            }
                        }

                        CoordinateTransform arg0_transform(arg0_shape);
                        CoordinateTransform arg1_transform(arg1_squeezed_shape);
                        CoordinateTransform output_transform(arg0_shape);

                        for (const Coordinate& output_coord : output_transform)
                        {
                            Coordinate arg1_coord = reduce(output_coord, arg1_squeezed_axes);
                            out[output_transform.index(output_coord)] =
                                elementwise_functor(arg0[arg0_transform.index(output_coord)],
                                                    arg1[arg1_transform.index(arg1_coord)]);
                        }
                    }
                }
            }

            /// \brief Helper function to implement autobroadcasting elementwise ternaryop
            /// references.
            ///
            /// \tparam U Element type of the selector tensor.
            /// \tparam T Element type of the input tensors.
            /// \tparam Functor Type of the functor for the elementwise operation. Must support
            ///                 operator()(U,T,T), and operator()(U,T,T) must return a value of type
            ///                 T.
            ///
            /// \param arg0 Pointer to the buffer for selector tensor.
            /// \param arg1 Pointer to the buffer for left operand input tensor.
            /// \param arg2 Pointer to the buffer for right operand input tensor.
            /// \param out Pointer to the buffer for output tensor. This must be pre-allocated by
            ///            the caller, and must be large enough to hold a tensor of the correct
            ///            shape.
            /// \param broadcast_spec Specification of the auto-broadcasting scheme.
            /// \param elementwise_functor Functor implementing the elementwise operation to be
            ///                            applied across the input tensors. Must accept an argument
            ///                            of
            ///                            type U and two of type T, and return a value of type T.
            template <typename T, typename U, typename Functor>
            void autobroadcast_select(const U* arg0,
                                      const T* arg1,
                                      const T* arg2,
                                      T* out,
                                      const Shape& arg0_shape,
                                      const Shape& arg1_shape,
                                      const Shape& arg2_shape,
                                      const op::AutoBroadcastSpec& broadcast_spec,
                                      Functor elementwise_functor)
            {
                switch (broadcast_spec.m_type)
                {
                case op::AutoBroadcastType::NONE:
                    for (size_t i = 0; i < shape_size(arg0_shape); i++)
                    {
                        out[i] = elementwise_functor(arg0[i], arg1[i], arg2[i]);
                    }
                    break;
                case op::AutoBroadcastType::NUMPY:
                    // Uses same approach as autobroadcast_binop.
                    {
                        Shape arg0_padded_shape = arg0_shape;
                        Shape arg1_padded_shape = arg1_shape;
                        Shape arg2_padded_shape = arg2_shape;

                        while (arg1_padded_shape.size() < arg2_padded_shape.size())
                        {
                            arg1_padded_shape.insert(arg1_padded_shape.begin(), 1);
                        }

                        while (arg2_padded_shape.size() < arg1_padded_shape.size())
                        {
                            arg2_padded_shape.insert(arg2_padded_shape.begin(), 1);
                        }

                        while (arg0_padded_shape.size() < arg1_padded_shape.size())
                        {
                            arg0_padded_shape.insert(arg0_padded_shape.begin(), 1);
                        }

                        Shape arg0_squeezed_shape;
                        Shape arg1_squeezed_shape;
                        Shape arg2_squeezed_shape;
                        AxisSet arg0_squeezed_axes;
                        AxisSet arg1_squeezed_axes;
                        AxisSet arg2_squeezed_axes;
                        Shape output_shape;

                        for (size_t i = 0; i < arg1_padded_shape.size(); i++)
                        {
                            if (arg1_padded_shape[i] == 1)
                            {
                                arg1_squeezed_axes.insert(i);
                            }
                            else
                            {
                                arg1_squeezed_shape.push_back(arg1_padded_shape[i]);
                            }

                            if (arg2_padded_shape[i] == 1)
                            {
                                arg2_squeezed_axes.insert(i);
                            }
                            else
                            {
                                arg2_squeezed_shape.push_back(arg2_padded_shape[i]);
                            }

                            if (arg0_padded_shape[i] == 1)
                            {
                                arg0_squeezed_axes.insert(i);
                            }
                            else
                            {
                                arg0_squeezed_shape.push_back(arg0_padded_shape[i]);
                            }

                            output_shape.push_back(arg1_padded_shape[i] == 1
                                                       ? arg2_padded_shape[i]
                                                       : arg1_padded_shape[i]);
                        }

                        CoordinateTransform arg0_transform(arg0_squeezed_shape);
                        CoordinateTransform arg1_transform(arg1_squeezed_shape);
                        CoordinateTransform arg2_transform(arg2_squeezed_shape);
                        CoordinateTransform output_transform(output_shape);

                        for (const Coordinate& output_coord : output_transform)
                        {
                            Coordinate arg0_coord = reduce(output_coord, arg0_squeezed_axes);
                            Coordinate arg1_coord = reduce(output_coord, arg1_squeezed_axes);
                            Coordinate arg2_coord = reduce(output_coord, arg2_squeezed_axes);
                            out[output_transform.index(output_coord)] =
                                elementwise_functor(arg0[arg0_transform.index(arg0_coord)],
                                                    arg1[arg1_transform.index(arg1_coord)],
                                                    arg2[arg2_transform.index(arg2_coord)]);
                        }
                    }
                    break;
                case op::AutoBroadcastType::PDPD:
                {
                    // arg0 and arg2 are broadcast to arg1 shape
                    int64_t axis = broadcast_spec.m_axis;
                    if (axis == -1)
                    {
                        axis = arg1_shape.size() - arg2_shape.size();
                    }

                    Shape arg0_padded_shape = arg0_shape;
                    Shape arg2_padded_shape = arg2_shape;
                    // Trim trailing ones
                    while (arg0_padded_shape.size() > 0 && arg0_padded_shape.back() == 1)
                    {
                        arg0_padded_shape.pop_back();
                    }

                    for (int64_t i = 0; i < axis; ++i)
                    {
                        arg0_padded_shape.insert(arg0_padded_shape.begin(), 1);
                    }

                    while (arg0_padded_shape.size() < arg1_shape.size())
                    {
                        arg0_padded_shape.insert(arg0_padded_shape.end(), 1);
                    }

                    while (arg2_padded_shape.size() > 0 && arg2_padded_shape.back() == 1)
                    {
                        arg2_padded_shape.pop_back();
                    }

                    for (int64_t i = 0; i < axis; ++i)
                    {
                        arg2_padded_shape.insert(arg2_padded_shape.begin(), 1);
                    }

                    while (arg2_padded_shape.size() < arg1_shape.size())
                    {
                        arg2_padded_shape.insert(arg2_padded_shape.end(), 1);
                    }

                    Shape arg0_squeezed_shape;
                    AxisSet arg0_squeezed_axes;
                    Shape arg2_squeezed_shape;
                    AxisSet arg2_squeezed_axes;

                    for (size_t i = 0; i < arg1_shape.size(); i++)
                    {
                        if (arg0_padded_shape[i] == 1)
                        {
                            arg0_squeezed_axes.insert(i);
                        }
                        else
                        {
                            arg0_squeezed_shape.push_back(arg0_padded_shape[i]);
                        }
                        if (arg2_padded_shape[i] == 1)
                        {
                            arg2_squeezed_axes.insert(i);
                        }
                        else
                        {
                            arg2_squeezed_shape.push_back(arg2_padded_shape[i]);
                        }
                    }

                    CoordinateTransform arg0_transform(arg0_squeezed_shape);
                    CoordinateTransform arg1_transform(arg1_shape);
                    CoordinateTransform arg2_transform(arg2_squeezed_shape);
                    CoordinateTransform output_transform(arg1_shape);

                    for (const Coordinate& output_coord : output_transform)
                    {
                        Coordinate arg0_coord = reduce(output_coord, arg0_squeezed_axes);
                        Coordinate arg2_coord = reduce(output_coord, arg2_squeezed_axes);
                        out[output_transform.index(output_coord)] =
                            elementwise_functor(arg0[arg0_transform.index(arg0_coord)],
                                                arg1[arg1_transform.index(output_coord)],
                                                arg2[arg2_transform.index(arg2_coord)]);
                    }
                }
                }
            }
        }
    }
}
