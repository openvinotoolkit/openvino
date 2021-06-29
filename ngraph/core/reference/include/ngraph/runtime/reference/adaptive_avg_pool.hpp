// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cfenv>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            inline size_t window_start(size_t idx, size_t arg_shape, size_t out_shape)
            {
                return floor(static_cast<double>(idx * arg_shape) / out_shape);
            }
            inline size_t window_end(size_t idx, size_t arg_shape, size_t out_shape)
            {
                return ceil(static_cast<double>((idx + 1) * arg_shape) / out_shape);
            }
            template <typename T>
            T avg(const T sum, size_t n)
            {
                if (n == 0)
                {
                    throw std::runtime_error("AdaptiveAvgPool elements == 0, must be non-zero");
                }

                if (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value)
                {
                    return static_cast<T>(std::nearbyint(static_cast<float>(sum) / n));
                }
                else
                {
                    return sum / n;
                }
            }

            template <typename T>
            void adaptive_avg_pool_1d(const T* arg, T* out, size_t h_in, size_t h_out)
            {
                for (size_t i = 0; i < h_out; i++)
                {
                    size_t h_start = window_start(i, h_in, h_out);
                    size_t h_end = window_end(i, h_in, h_out);
                    out[i] =
                        avg(std::accumulate(arg + h_start, arg + h_end, T{0}), h_end - h_start);
                }
            }
            template <typename T>
            void adaptive_avg_pool_2d(
                const T* arg, T* out, size_t h_in, size_t h_out, size_t w_in, size_t w_out)
            {
                for (size_t i = 0; i < h_out; i++)
                {
                    size_t h_start = window_start(i, h_in, h_out);
                    size_t h_end = window_end(i, h_in, h_out);
                    for (size_t j = 0; j < w_out; j++)
                    {
                        size_t w_start = window_start(j, w_in, w_out);
                        size_t w_end = window_end(j, w_in, w_out);
                        T result = 0;
                        for (size_t n = h_start; n < h_end; n++)
                            result = std::accumulate(
                                arg + n * w_in + w_start, arg + n * w_in + w_end, result);
                        out[i * w_out + j] = avg(result, (w_end - w_start) * (h_end - h_start));
                    }
                }
            }
            template <typename T>
            void adaptive_avg_pool_3d(const T* arg,
                                      T* out,
                                      size_t d_in,
                                      size_t d_out,
                                      size_t h_in,
                                      size_t h_out,
                                      size_t w_in,
                                      size_t w_out)
            {
                for (size_t i = 0; i < d_out; i++)
                {
                    size_t d_start = window_start(i, d_in, d_out);
                    size_t d_end = window_end(i, d_in, d_out);
                    for (size_t j = 0; j < h_out; j++)
                    {
                        size_t h_start = window_start(j, h_in, h_out);
                        size_t h_end = window_end(j, h_in, h_out);
                        for (size_t k = 0; k < w_out; k++)
                        {
                            size_t w_start = window_start(k, w_in, w_out);
                            size_t w_end = window_end(k, w_in, w_out);
                            T result = 0;
                            for (size_t n = d_start; n < d_end; n++)
                                for (size_t m = h_start; m < h_end; m++)
                                {
                                    auto pos = arg + n * h_in * w_in + m * w_in;
                                    result = std::accumulate(pos + w_start, pos + w_end, result);
                                }
                            out[i * h_out * w_out + j * w_out + k] = avg(
                                result, (d_end - d_start) * (w_end - w_start) * (h_end - h_start));
                        }
                    }
                }
            }
            template <typename T>
            void adaptive_avg_pool(const T* arg,
                                   T* out,
                                   const Shape& arg_shape,
                                   const Shape& out_shape)
            {
                size_t channel_size = 1;
                for (size_t i = 2; i < arg_shape.size(); i++)
                    channel_size *= arg_shape[i];
                size_t batch_size = arg_shape[1] * channel_size;
                size_t out_channel_size = 1;
                for (size_t i = 2; i < out_shape.size(); i++)
                    out_channel_size *= out_shape[i];
                size_t out_batch_size = arg_shape[1] * out_channel_size;
                for (size_t b = 0; b < arg_shape[0]; b++)
                {
                    for (size_t c = 0; c < arg_shape[1]; c++)
                    {
                        if (arg_shape.size() == 3 && out_shape.size() == 3)
                        {
                            adaptive_avg_pool_1d<T>(arg + b * batch_size + c * channel_size,
                                                    out + b * out_batch_size + c * out_channel_size,
                                                    arg_shape[2],
                                                    out_shape[2]);
                        }
                        else if (arg_shape.size() == 4 && out_shape.size() == 4)
                        {
                            adaptive_avg_pool_2d<T>(arg + b * batch_size + c * channel_size,
                                                    out + b * out_batch_size + c * out_channel_size,
                                                    arg_shape[2],
                                                    out_shape[2],
                                                    arg_shape[3],
                                                    out_shape[3]);
                        }
                        else if (arg_shape.size() == 5 && out_shape.size() == 5)
                        {
                            adaptive_avg_pool_3d<T>(arg + b * batch_size + c * channel_size,
                                                    out + b * out_batch_size + c * out_channel_size,
                                                    arg_shape[2],
                                                    out_shape[2],
                                                    arg_shape[3],
                                                    out_shape[3],
                                                    arg_shape[4],
                                                    out_shape[4]);
                        }
                    }
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
