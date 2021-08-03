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
                                 const size_t slices_count,
                                 const T* data_ptr,
                                 const size_t axis_dim_size)
                {
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
            } // namespace details

            template <typename T, typename P>
            void cumsum(const T* arg,
                        const P* axis_tensor,
                        T* out,
                        const Shape& tensor_shape,
                        const bool exclusive,
                        const bool reverse)
            {
                if (tensor_shape.size() == 1)
                {
                    std::vector<T> input_data(arg, arg + shape_size(tensor_shape));
                    std::vector<T> output_data(shape_size(tensor_shape), 0);
                    if (reverse)
                    {
                        details::flat_cumsum(input_data.rbegin(),
                                             input_data.rend(),
                                             output_data.rbegin(),
                                             exclusive);
                    }
                    else
                    {
                        details::flat_cumsum(
                            input_data.begin(), input_data.end(), output_data.begin(), exclusive);
                    }
                    std::copy(begin(output_data), end(output_data), out);
                }
                else
                {
                    const auto axis = axis_tensor[0];
                    const bool is_last_axis = axis == tensor_shape.size() - 1;
                    if (is_last_axis)
                    {
                        std::vector<T> output_data(shape_size(tensor_shape), 0);
                        const auto slices_count = shape_size(Shape(
                            tensor_shape.begin(), tensor_shape.begin() + tensor_shape.size() - 1));
                        details::loop_cumsum(
                            output_data, exclusive, reverse, slices_count, arg, tensor_shape[axis]);
                        std::copy(begin(output_data), end(output_data), out);
                    }
                    else
                    {
                        std::vector<int64_t> transposed_axes(tensor_shape.size());
                        std::vector<size_t> transposed_shape(tensor_shape);

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
                        std::vector<T> transposed_output_data(shape_size(tensor_shape));

                        std::vector<T> output_data(shape_size(tensor_shape), 0);
                        const auto slices_count = shape_size(
                            Shape(transposed_shape.begin(),
                                  transposed_shape.begin() + transposed_shape.size() - 1));
                        details::loop_cumsum(
                            output_data, exclusive, reverse, slices_count, out, tensor_shape[axis]);

                        std::iota(transposed_axes.begin(), transposed_axes.end(), 0);
                        std::rotate(transposed_axes.begin() + axis,
                                    transposed_axes.end() - 1,
                                    transposed_axes.end());
                        reference::transpose(reinterpret_cast<char*>(output_data.data()),
                                             reinterpret_cast<char*>(transposed_output_data.data()),
                                             transposed_shape,
                                             sizeof(T),
                                             transposed_axes.data(),
                                             tensor_shape);

                        std::copy(begin(transposed_output_data), end(transposed_output_data), out);
                    }
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
