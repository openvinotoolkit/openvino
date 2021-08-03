// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <iterator>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/reference/transpose.hpp"
#include "ngraph/type/bfloat16.hpp"
#include "ngraph/type/float16.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            namespace details
            {
                template <typename InputIterator, typename OutputIterator>
                void flat_cumsum(InputIterator it_input_begin,
                                 InputIterator it_input_end,
                                 OutputIterator it_output_begin,
                                 const bool exclusive)
                {
                    if (exclusive)
                    {
                        --it_input_end;
                        ++it_output_begin;
                    }
                    std::partial_sum(it_input_begin, it_input_end, it_output_begin);
                }
                template <typename T>
                void loop_cumsum(std::vector<T>& output_data,
                                 const bool exclusive,
                                 const bool reverse,
                                 const Shape& shape,
                                 const T* data_ptr,
                                 const size_t axis_dim_size)
                {
                   const auto slices_count = shape_size(Shape(
                        shape.begin(), shape.begin() + shape.size() - 1));
                    auto axis_dim_counter = 0;
                    for (auto i = 0; i < slices_count; ++i)
                    {
                        std::vector<T> input_data(data_ptr, data_ptr + axis_dim_size);
                        std::vector<T> output_tmp_data(axis_dim_size, 0);
                        if (reverse)
                        {
                            details::flat_cumsum(input_data.rbegin(),
                                                 input_data.rend(),
                                                 output_tmp_data.rbegin(),
                                                 exclusive);
                        }
                        else
                        {
                            details::flat_cumsum(input_data.begin(),
                                                 input_data.end(),
                                                 output_tmp_data.begin(),
                                                 exclusive);
                        }
                        std::copy(begin(output_tmp_data),
                                  end(output_tmp_data),
                                  output_data.begin() + axis_dim_counter);

                        data_ptr += axis_dim_size;
                        axis_dim_counter += axis_dim_size;
                    }
                }

                template <typename T, typename P>
                void transpose_to_last_axis(const T* arg,
                                            T* out,
                                            const P axis,
                                            const Shape& tensor_shape,
                                            Shape& transposed_shape)
                {
                    std::vector<int64_t> transposed_axes(tensor_shape.size());
                    std::iota(transposed_axes.begin(), transposed_axes.end(), 0);
                    std::rotate(transposed_axes.begin() + axis,
                                transposed_axes.begin() + axis + 1,
                                transposed_axes.end());
                    std::rotate(transposed_shape.begin() + axis,
                                transposed_shape.begin() + axis + 1,
                                transposed_shape.end());
                    reference::transpose(reinterpret_cast<const char*>(arg),
                                         reinterpret_cast<char*>(out),
                                         tensor_shape,
                                         sizeof(T),
                                         transposed_axes.data(),
                                         transposed_shape);
                }

                template <typename T, typename P>
                void transpose_to_original(T* arg,
                                            T* out,
                                            const P axis,
                                            const Shape& tensor_shape,
                                            Shape& transposed_shape)
                {
                    std::vector<int64_t> transposed_axes(tensor_shape.size());
                    std::iota(transposed_axes.begin(), transposed_axes.end(), 0);
                    std::rotate(transposed_axes.begin() + axis,
                                transposed_axes.end() - 1,
                                transposed_axes.end());

                    std::vector<T> transposed_output_data(shape_size(tensor_shape));
                    reference::transpose(reinterpret_cast<char*>(arg),
                                         reinterpret_cast<char*>(out),
                                         transposed_shape,
                                         sizeof(T),
                                         transposed_axes.data(),
                                         tensor_shape);
                }
            } // namespace details

            template <typename T, typename P>
            void cumsum(const T* arg,
                        const P* axis_tensor,
                        T* out,
                        const Shape& tensor_shape,
                        const bool exclusive,
                        const bool reverse)
            {
                const auto axis = axis_tensor[0];
                const bool is_last_axis = axis == tensor_shape.size() - 1;
                if (is_last_axis)
                {
                    std::vector<T> output_data(shape_size(tensor_shape), 0);
                    details::loop_cumsum(
                        output_data, exclusive, reverse, tensor_shape, arg, tensor_shape[axis]);
                    std::copy(begin(output_data), end(output_data), out);
                }
                else
                {
                    Shape transposed_shape(tensor_shape);
                    details::transpose_to_last_axis(arg, out, axis, tensor_shape, transposed_shape);

                    std::vector<T> output_data(shape_size(tensor_shape), 0);
                    details::loop_cumsum(
                        output_data, exclusive, reverse, transposed_shape, out, tensor_shape[axis]);

                    std::vector<T> transposed_output_data(shape_size(tensor_shape));
                    details::transpose_to_original(output_data.data(), transposed_output_data.data(), axis, tensor_shape, transposed_shape);
                    std::copy(begin(transposed_output_data), end(transposed_output_data), out);
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
