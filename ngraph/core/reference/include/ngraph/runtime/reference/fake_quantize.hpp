// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            namespace
            {
                std::vector<size_t>
                    calc_broadcast_index_offset(const std::vector<size_t>& memory_offsets,
                                                const std::vector<size_t>& broadcast_shape)
                {
                    std::vector<size_t> broadcast_offsets(broadcast_shape.size(), 0);
                    for (int i = broadcast_shape.size() - 2; i >= 0; --i)
                    {
                        if (broadcast_shape[i] == 1)
                        {
                            broadcast_offsets[i] = memory_offsets[i];
                        }
                    }
                    if (!std::all_of(broadcast_shape.begin(),
                                     broadcast_shape.end(),
                                     [](size_t i) { return i == 1; }) &&
                        broadcast_shape.back() == 1)
                    {
                        broadcast_offsets[broadcast_offsets.size() - 1] = 1;
                    }
                    if (broadcast_shape.back() == 1)
                    {
                        for (int i = broadcast_shape.size() - 1; i >= 0; --i)
                        {
                            if (broadcast_shape[i] != 1)
                            {
                                broadcast_offsets[i] = memory_offsets[i] - 1;
                                break;
                            }
                        }
                    }
                    return broadcast_offsets;
                }

                size_t calc_full_broadcast_offset(const std::vector<size_t>& current_dims,
                                                  const std::vector<size_t>& offsets)
                {
                    size_t full_index_offset = 0;
                    for (size_t i = 0; i < current_dims.size(); ++i)
                    {
                        full_index_offset += offsets[i] * current_dims[i];
                    }
                    return full_index_offset;
                }

                void align_shape_sizes(Shape& shape, size_t target_size)
                {
                    for (size_t i = 0; i < shape.size() - target_size; ++i)
                    {
                        shape.insert(shape.begin(), 1);
                    }
                }

                void increment_current_dim(std::vector<size_t>& current_dims,
                                           const std::vector<size_t>& shape,
                                           size_t incremented_dim_number)
                {
                    current_dims[incremented_dim_number] += 1;
                    if (current_dims[incremented_dim_number] == shape[incremented_dim_number] &&
                        incremented_dim_number != 0)
                    {
                        for (size_t i = incremented_dim_number; i < shape.size(); ++i)
                        {
                            current_dims[i] = 0;
                        }
                        increment_current_dim(current_dims, shape, incremented_dim_number - 1);
                    }
                }
            } // namespace

            template <typename T>
            void fake_quantize(const T* arg,
                               const T* in_low,
                               const T* in_high,
                               const T* out_low,
                               const T* out_high,
                               T* out,
                               const Shape& arg_shape,
                               const Shape& _in_low_shape,
                               const Shape& _in_high_shape,
                               const Shape& _out_low_shape,
                               const Shape& _out_high_shape,
                               size_t levels)
            {
                auto initial_round_mode = std::fegetround();
                std::fesetround(FE_TONEAREST);
                Shape in_low_shape(_in_low_shape);
                Shape in_high_shape(_in_high_shape);
                Shape out_low_shape(_out_low_shape);
                Shape out_high_shape(_out_high_shape);

                if (in_low_shape.size() > arg_shape.size() ||
                    in_high_shape.size() > arg_shape.size() ||
                    out_low_shape.size() > arg_shape.size() ||
                    out_high_shape.size() > arg_shape.size())
                {
                    throw std::runtime_error(
                        std::string("Tensors with inout\\output ranges should have rank less or "
                                    "equal to data tensor rank equal to ") +
                        std::to_string(arg_shape.size()));
                }

                std::vector<size_t> arg_memory_offsets(arg_shape.size(), 0);
                for (int i = arg_shape.size() - 2; i >= 0; i--)
                {
                    arg_memory_offsets[i] = std::accumulate(
                        arg_shape.begin() + i + 1, arg_shape.end(), 1, std::multiplies<size_t>());
                }
                align_shape_sizes(in_low_shape, arg_shape.size());
                align_shape_sizes(in_high_shape, arg_shape.size());
                align_shape_sizes(out_low_shape, arg_shape.size());
                align_shape_sizes(out_high_shape, arg_shape.size());

                std::vector<size_t> in_low_offsets, in_high_offsets, out_low_offsets,
                    out_high_offsets;
                bool in_low_trivial_broadcast = false;
                bool in_high_trivial_broadcast = false;
                bool out_low_trivial_broadcast = false;
                bool out_high_trivial_broadcast = false;
                bool in_low_aligned = false;
                bool in_high_aligned = false;
                bool out_low_aligned = false;
                bool out_high_aligned = false;

                auto check_trivial_broadcast =
                    [&arg_shape, &arg_memory_offsets](Shape& shape_to_check,
                                                      std::vector<size_t>& target_offsets,
                                                      bool& trivial_broadcast,
                                                      bool& aligned) {
                        if (shape_size(shape_to_check) == 1 || shape_size(shape_to_check) == 0)
                        {
                            trivial_broadcast = true;
                        }
                        else if (shape_to_check == arg_shape)
                        {
                            aligned = true;
                        }
                        else
                        {
                            target_offsets =
                                calc_broadcast_index_offset(arg_memory_offsets, shape_to_check);
                        }
                    };
                check_trivial_broadcast(
                    in_low_shape, in_low_offsets, in_low_trivial_broadcast, in_low_aligned);
                check_trivial_broadcast(
                    in_high_shape, in_high_offsets, in_high_trivial_broadcast, in_high_aligned);
                check_trivial_broadcast(
                    out_low_shape, out_low_offsets, out_low_trivial_broadcast, out_low_aligned);
                check_trivial_broadcast(
                    out_high_shape, out_high_offsets, out_high_trivial_broadcast, out_high_aligned);

                std::vector<size_t> current_dim(arg_shape.size(), 0);

                auto get_value = [&current_dim](bool is_trivial_broadcast,
                                                bool is_aligned,
                                                const T* data,
                                                size_t idx,
                                                const std::vector<size_t>& offsets) {
                    T val;
                    if (is_aligned)
                    {
                        val = data[idx];
                    }
                    else if (is_trivial_broadcast)
                    {
                        val = data[0];
                    }
                    else
                    {
                        size_t index_offset = calc_full_broadcast_offset(current_dim, offsets);
                        if (index_offset != 0)
                        {
                            NGRAPH_CHECK(idx >= index_offset, "Incorrect index offset value!");
                        }
                        val = data[idx - index_offset];
                    }
                    return val;
                };
                for (size_t i = 0; i < shape_size(arg_shape); ++i)
                {
                    T in_low_val = get_value(
                        in_low_trivial_broadcast, in_low_aligned, in_low, i, in_low_offsets);
                    T in_high_val = get_value(
                        in_high_trivial_broadcast, in_high_aligned, in_high, i, in_high_offsets);
                    T out_low_val = get_value(
                        out_low_trivial_broadcast, out_low_aligned, out_low, i, out_low_offsets);
                    T out_high_val = get_value(out_high_trivial_broadcast,
                                               out_high_aligned,
                                               out_high,
                                               i,
                                               out_high_offsets);
                    if (arg[i] <= std::min(in_low_val, in_high_val))
                    {
                        out[i] = out_low_val;
                    }
                    else if (arg[i] > std::max(in_low_val, in_high_val))
                    {
                        out[i] = out_high_val;
                    }
                    else
                    {
                        out[i] = nearbyint((arg[i] - in_low_val) / (in_high_val - in_low_val) *
                                           (levels - 1)) /
                                     (levels - 1) * (out_high_val - out_low_val) +
                                 out_low_val;
                    }
                    increment_current_dim(current_dim, arg_shape, arg_shape.size() - 1);
                }
                std::fesetround(initial_round_mode);
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
