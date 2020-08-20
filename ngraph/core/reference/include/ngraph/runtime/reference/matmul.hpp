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
#include <numeric>
#include <utility>
#include <vector>

#include "ngraph/axis_vector.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/runtime/opt_kernel/reshape.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/runtime/reference/dot.hpp"
#include "ngraph/shape_util.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            /// \brief Reference kernel for matmul computation.
            ///
            /// \tparam T Type of input and output tensors.
            ///
            /// \param arg0 Pointer to the buffer for left operand input tensor.
            /// \param arg1 Pointer to the buffer for right operand input tensor.
            /// \param out Pointer to the buffer for output tensor. This must be pre-allocated by
            ///            the caller, and must be large enough to hold a tensor of the correct
            ///            shape.
            /// \param arg0_shape Shape of arg0.
            /// \param arg1_shape Shape of arg1.
            /// \param out_shape Shape of out.
            /// \param transpose_arg0 Flag to indicate if transpose on arg0.
            /// \param transpose_arg1 Flag to indicate if transpose on arg1.
            template <typename T>
            void matmul(const T* arg0,
                        const T* arg1,
                        T* out,
                        const Shape& arg0_shape,
                        const Shape& arg1_shape,
                        const Shape& out_shape,
                        bool transpose_arg0,
                        bool transpose_arg1)
            {
                // Steps to compute matmul:
                // 1) Check inputs and perform transpose on arg if applicable
                // 2) If ranks of both args are 2D and below (no batch dim),
                //    perform dot and return result; otherwise, continue next
                // 3) Check if auto broadcast is needed on args or transposed args,
                //    and perform broadcast if applicable
                // 4) Perform dot on the args or updated args and return result

                size_t arg0_rank = arg0_shape.size();
                size_t arg1_rank = arg1_shape.size();
                size_t out_rank = out_shape.size();

                // vector vars to hold pontential intermediate transpose,
                // broadcast result
                vector<T> arg0_transpose_vec;
                vector<T> arg1_transpose_vec;
                vector<T> arg0_broadcast_vec;
                vector<T> arg1_broadcast_vec;

                // pointers to updated inputs
                const T* arg0_update = arg0;
                const T* arg1_update = arg1;

                // vars for updated inputs shapes
                Shape wip_arg0_shape = arg0_shape;
                Shape wip_arg1_shape = arg1_shape;

                auto get_transpose_order = [](const Shape& input_shape) {
                    size_t rank = input_shape.size();
                    NGRAPH_CHECK(rank > 1, "Invalid input for transpose");
                    vector<size_t> axes_order(rank);
                    iota(axes_order.begin(), axes_order.end(), 0);
                    swap(axes_order[rank - 1], axes_order[rank - 2]);
                    return AxisVector{begin(axes_order), end(axes_order)};
                };

                auto get_broadcast_axes = [](const Shape& marker_shape, const Shape& target_shape) {
                    NGRAPH_CHECK(marker_shape.size() == target_shape.size(),
                                 "Incompatible input shapes");
                    AxisSet broadcast_axes;
                    for (size_t i = 0; i < marker_shape.size(); i++)
                    {
                        if (marker_shape[i] == 1 && target_shape[i] != 1)
                        {
                            broadcast_axes.insert(i);
                        }
                    }
                    return broadcast_axes;
                };

                // Perform transpose if requested
                if (transpose_arg0 && arg0_rank > 1)
                {
                    arg0_transpose_vec.reserve(shape_size(arg0_shape));
                    auto axis_vector = get_transpose_order(arg0_shape);
                    swap(wip_arg0_shape[arg0_rank - 1], wip_arg0_shape[arg0_rank - 2]);
                    opt_kernel::reshape(reinterpret_cast<const char*>(arg0),
                                        reinterpret_cast<char*>(arg0_transpose_vec.data()),
                                        arg0_shape,
                                        axis_vector,
                                        wip_arg0_shape,
                                        sizeof(T));

                    arg0_update = arg0_transpose_vec.data();
                }

                if (transpose_arg1 && arg1_rank > 1)
                {
                    arg1_transpose_vec.reserve(shape_size(arg1_shape));
                    auto axis_vector = get_transpose_order(arg1_shape);
                    swap(wip_arg1_shape[arg1_rank - 1], wip_arg1_shape[arg1_rank - 2]);
                    opt_kernel::reshape(reinterpret_cast<const char*>(arg1),
                                        reinterpret_cast<char*>(arg1_transpose_vec.data()),
                                        arg1_shape,
                                        axis_vector,
                                        wip_arg1_shape,
                                        sizeof(T));

                    arg1_update = arg1_transpose_vec.data();
                }

                // Inputs are 2D and below, perform dot directly
                if (arg0_rank <= 2 && arg1_rank <= 2)
                {
                    return dot(arg0_update,
                               arg1_update,
                               out,
                               wip_arg0_shape,
                               wip_arg1_shape,
                               out_shape,
                               1);
                }

                // Check and perform auto-broadcast if needed
                // If one of the arg is 2D or below, no need to
                // do broadcast on it, just use its value for
                // every batch of dot compuatation later

                if (arg0_rank > 2 && arg1_rank > 2)
                {
                    const auto& broadcast_shapes = builder::get_numpy_broadcast_shapes(
                        {Shape{begin(wip_arg0_shape), next(end(wip_arg0_shape), -2)},
                         Shape{begin(wip_arg1_shape), next(end(wip_arg1_shape), -2)}});

                    Shape arg0_br_target_shape = broadcast_shapes.first;
                    Shape arg1_br_target_shape = broadcast_shapes.first;
                    Shape arg0_br_marker_shape = broadcast_shapes.second.at(0);
                    Shape arg1_br_marker_shape = broadcast_shapes.second.at(1);

                    arg0_br_target_shape.insert(
                        end(arg0_br_target_shape),
                        next(begin(wip_arg0_shape), wip_arg0_shape.size() - 2),
                        end(wip_arg0_shape));
                    arg1_br_target_shape.insert(
                        end(arg1_br_target_shape),
                        next(begin(wip_arg1_shape), wip_arg1_shape.size() - 2),
                        end(wip_arg1_shape));

                    arg0_br_marker_shape.insert(
                        end(arg0_br_marker_shape),
                        next(begin(wip_arg0_shape), wip_arg0_shape.size() - 2),
                        end(wip_arg0_shape));
                    arg1_br_marker_shape.insert(
                        end(arg1_br_marker_shape),
                        next(begin(wip_arg1_shape), wip_arg1_shape.size() - 2),
                        end(wip_arg1_shape));

                    if (arg0_br_target_shape != wip_arg0_shape)
                    {
                        auto broadcast_axes =
                            get_broadcast_axes(arg0_br_marker_shape, arg0_br_target_shape);
                        if (!broadcast_axes.empty())
                        {
                            arg0_broadcast_vec.reserve(shape_size(arg0_br_target_shape));
                            broadcast(arg0_update,
                                      arg0_broadcast_vec.data(),
                                      wip_arg0_shape,
                                      arg0_br_target_shape,
                                      broadcast_axes);

                            arg0_update = arg0_broadcast_vec.data();
                            wip_arg0_shape = arg0_br_target_shape;
                            arg0_rank = wip_arg0_shape.size();
                        }
                    }

                    if (arg1_br_target_shape != wip_arg1_shape)
                    {
                        auto broadcast_axes =
                            get_broadcast_axes(arg1_br_marker_shape, arg1_br_target_shape);
                        if (!broadcast_axes.empty())
                        {
                            arg1_broadcast_vec.reserve(shape_size(arg1_br_target_shape));
                            broadcast(arg1_update,
                                      arg1_broadcast_vec.data(),
                                      wip_arg1_shape,
                                      arg1_br_target_shape,
                                      broadcast_axes);

                            arg1_update = arg1_broadcast_vec.data();
                            wip_arg1_shape = arg1_br_target_shape;
                            arg1_rank = wip_arg1_shape.size();
                        }
                    }
                }

                // Perform batched dot

                size_t output_batch_size = 1;

                // Calculate number of batches
                if (out_rank < 3)
                {
                    // Output is {batch_size, dot_result}, i.e.,
                    // arg 0 shape {2}, arg1 shape {3, 2, 1}, output shape {3, 1}
                    output_batch_size = out_shape[0];
                }
                else
                {
                    for (size_t i = 0; i < (out_rank - 2); i++)
                    {
                        output_batch_size *= out_shape[i];
                    }
                }

                Shape dot_arg0_shape = (arg0_rank > 2) ? Shape{wip_arg0_shape[arg0_rank - 2],
                                                               wip_arg0_shape[arg0_rank - 1]}
                                                       : wip_arg0_shape;
                Shape dot_arg1_shape = (arg1_rank > 2) ? Shape{wip_arg1_shape[arg1_rank - 2],
                                                               wip_arg1_shape[arg1_rank - 1]}
                                                       : wip_arg1_shape;
                Shape dot_output_shape =
                    (out_rank > 2) ? Shape{out_shape[out_rank - 2], out_shape[out_rank - 1]}
                                   : Shape{out_shape[out_rank - 1]};

                const size_t arg0_offset = (arg0_rank > 2) ? shape_size(dot_arg0_shape) : 0;
                const size_t arg1_offset = (arg1_rank > 2) ? shape_size(dot_arg1_shape) : 0;
                const size_t output_offset = shape_size(dot_output_shape);
                for (size_t i = 0; i < output_batch_size; i++)
                {
                    dot(arg0_update + i * arg0_offset,
                        arg1_update + i * arg1_offset,
                        out + i * output_offset,
                        dot_arg0_shape,
                        dot_arg1_shape,
                        dot_output_shape,
                        1);
                }
            }
        }
    }
}

NGRAPH_SUPPRESS_DEPRECATED_END
