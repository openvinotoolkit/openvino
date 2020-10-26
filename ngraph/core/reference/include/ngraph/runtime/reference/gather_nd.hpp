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

#include <cassert>
#include <chrono>
#include <iostream>
#include <numeric>

#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            namespace old_impl
            {
                // foreach leaf_vector_index in indices.shape[:-1]
                //     vector = indices[leaf_vector_index]
                //     out[leaf_vector_index:] = params[vector]
                template <typename T, typename U>
                void gather_nd_batch(const T* params,
                                     const U* indices,
                                     T* out,
                                     const Shape& params_shape,
                                     const Shape& indices_shape,
                                     const Shape& out_shape)
                {
                    using namespace std;
                    // Create a CoordinateTransform for "indices" that visits only the first element
                    // along inner most axis
                    size_t indices_ndim = static_cast<size_t>(indices_shape.size());
                    Coordinate indices_outer_start_corner(indices_ndim, 0);
                    Coordinate indices_outer_end_corner(indices_shape);
                    size_t slice_rank = indices_shape[indices_ndim - 1];
                    indices_outer_end_corner[indices_ndim - 1] = 1;
                    Strides indices_strides(indices_ndim, 1);
                    AxisVector indices_axis_order(indices_ndim);
                    std::iota(indices_axis_order.begin(), indices_axis_order.end(), 0);
                    CoordinateTransform indices_outer_transform(indices_shape,
                                                                indices_outer_start_corner,
                                                                indices_outer_end_corner,
                                                                indices_strides,
                                                                indices_axis_order);

                    // Create a matching CoordinateTransform for "out" that visits the same outer
                    // coordinates
                    size_t out_ndim = static_cast<size_t>(out_shape.size());
                    Coordinate out_start_corner(out_ndim, 0);
                    Coordinate out_end_corner(out_shape);
                    for (size_t i = indices_ndim - 1; i < out_ndim; i++)
                    {
                        out_end_corner[i] = 1;
                    }
                    Strides out_strides(out_ndim, 1);
                    AxisVector out_axis_order(out_ndim);
                    std::iota(out_axis_order.begin(), out_axis_order.end(), 0);
                    CoordinateTransform out_transform(
                        out_shape, out_start_corner, out_end_corner, out_strides, out_axis_order);
                    size_t params_ndim = static_cast<size_t>(params_shape.size());
                    Strides params_strides(params_ndim, 1);
                    AxisVector params_axis_order(params_ndim);
                    std::iota(params_axis_order.begin(), params_axis_order.end(), 0);

                    // Gather slices from "params" and copy to "out"
                    auto out_coord_iter = out_transform.begin();
                    for (const Coordinate& indices_coord : indices_outer_transform)
                    {
                        Coordinate params_start_corner(params_ndim, 0);
                        Coordinate params_end_corner(params_shape);
                        auto indices_index = indices_outer_transform.index(indices_coord);
                        for (size_t i = 0; i < slice_rank; i++)
                        {
                            U index = indices[indices_index];
                            // take care of negative indices
                            index = index >= 0 ? index : index + params_shape[i];
                            params_start_corner[i] = index;
                            params_end_corner[i] = index + 1;
                            indices_index++;
                        }
                        CoordinateTransform params_transform(params_shape,
                                                             params_start_corner,
                                                             params_end_corner,
                                                             params_strides,
                                                             params_axis_order);
                        if (out_coord_iter == out_transform.end())
                            break;
                        auto out_index = out_transform.index(*out_coord_iter);
                        for (const Coordinate& params_coord : params_transform)
                        {
                            out[out_index] = params[params_transform.index(params_coord)];
                            out_index++;
                        }
                        out_coord_iter++;
                    }
                }

                template <typename T, typename U>
                void gather_nd(const T* params,
                               const U* indices,
                               T* out,
                               const Shape& params_shape,
                               const Shape& indices_shape,
                               const Shape& out_shape,
                               int batch_dims = 0)
                {
                    using namespace std;
                    if (batch_dims == 0)
                    {
                        gather_nd_batch(
                            params, indices, out, params_shape, indices_shape, out_shape);
                        return;
                    }

                    size_t indices_ndim = static_cast<size_t>(indices_shape.size());
                    Coordinate indices_outer_start_corner(indices_ndim, 0);
                    Coordinate indices_outer_end_corner(indices_shape);
                    for (size_t i = batch_dims; i < indices_ndim; i++)
                    {
                        indices_outer_end_corner[i] = 1;
                    }
                    Strides indices_strides(indices_ndim, 1);
                    AxisVector indices_axis_order(indices_ndim);
                    std::iota(indices_axis_order.begin(), indices_axis_order.end(), 0);
                    CoordinateTransform indices_outer_transform(indices_shape,
                                                                indices_outer_start_corner,
                                                                indices_outer_end_corner,
                                                                indices_strides,
                                                                indices_axis_order);

                    size_t params_ndim = static_cast<size_t>(params_shape.size());
                    Coordinate params_outer_start_corner(params_ndim, 0);
                    Coordinate params_outer_end_corner(params_shape);
                    for (size_t i = batch_dims; i < params_ndim; i++)
                    {
                        params_outer_end_corner[i] = 1;
                    }
                    Strides params_strides(params_ndim, 1);
                    AxisVector params_axis_order(params_ndim);
                    std::iota(params_axis_order.begin(), params_axis_order.end(), 0);
                    CoordinateTransform params_outer_transform(params_shape,
                                                               params_outer_start_corner,
                                                               params_outer_end_corner,
                                                               params_strides,
                                                               params_axis_order);

                    size_t out_ndim = static_cast<size_t>(out_shape.size());
                    Coordinate out_start_corner(out_ndim, 0);
                    Coordinate out_end_corner(out_shape);
                    for (size_t i = 1; i < out_ndim; i++)
                    {
                        out_end_corner[i] = 1;
                    }
                    Strides out_strides(out_ndim, 1);
                    AxisVector out_axis_order(out_ndim);
                    std::iota(out_axis_order.begin(), out_axis_order.end(), 0);
                    CoordinateTransform out_transform(
                        out_shape, out_start_corner, out_end_corner, out_strides, out_axis_order);

                    Shape indices_shape_batch(indices_shape.begin() + batch_dims,
                                              indices_shape.end());
                    Shape params_shape_batch(params_shape.begin() + batch_dims, params_shape.end());
                    Shape output_shape_batch(out_shape.begin() + 1, out_shape.end());
                    auto out_coord_iter = out_transform.begin();
                    auto params_coord_iter = params_outer_transform.begin();
                    for (const Coordinate& indices_coord : indices_outer_transform)
                    {
                    if (params_coord_iter == params_outer_transform.end() ||
                        out_coord_iter == out_transform.end())
                        break;
                        auto indices_index = indices_outer_transform.index(indices_coord);
                        auto params_index = params_outer_transform.index(*params_coord_iter);
                        auto output_index = out_transform.index(*out_coord_iter);
                        gather_nd_batch(params + params_index,
                                        indices + indices_index,
                                        out + output_index,
                                        params_shape_batch,
                                        indices_shape_batch,
                                        output_shape_batch);

                        out_coord_iter++;
                        params_coord_iter++;
                    }
                }
            } // namespace old_impl

            namespace new_impl
            {
                ///
                /// Implementation find maximum length of *slice* of input *params* which might be
                /// copied to *out* index by index.
                /// +-------+-------------+-------+
                /// | batch | indices[-1] | slice |
                /// | shape |   shape     | shape |
                /// +-------+-------------+-------+
                ///
                template <typename T, typename U>
                void gather_nd(const T* const params,
                               const U* const indices,
                               T* const out,
                               const Shape& params_shape,
                               const Shape& indices_shape,
                               const Shape& out_shape,
                               const int batch_dims = 0)
                {
                    using std::begin;
                    using std::end;
                    using std::next;
                    using std::prev;
                    const auto rbegin = [](const Shape& s) { // generic since C++14
                        return s.rbegin();
                    };

                    const Shape batch_shape(begin(params_shape),
                                            next(begin(params_shape), batch_dims));
                    const auto batch_size = shape_size(batch_shape);

                    // out_shape should have on first dim multiplication of batch number of first
                    // dimensions of shape
                    assert(!batch_dims || batch_size == out_shape.front());

                    // dimensions in params and indices have to be equal on batch dimensions
                    assert(std::equal(begin(params_shape),
                                      next(begin(params_shape), batch_dims),
                                      begin(indices_shape)));

                    const auto first_slice_index_in_params = batch_dims + indices_shape.back();

                    // params_shape should have enough rank to be index by indices
                    assert(first_slice_index_in_params <= params_shape.size());

                    const auto slice_shape = Shape(
                        next(begin(params_shape), first_slice_index_in_params), end(params_shape));
                    const auto slice_size = shape_size(slice_shape);

                    const auto indices_offsets = [&] {
                        std::vector<size_t> offsets{slice_size};
                        const auto beg_s = next(rbegin(params_shape), slice_shape.size());
                        const auto end_s = next(beg_s, indices_shape.back() - 1);
                        for (auto s = beg_s; s != end_s; ++s)
                        {
                            const auto o = offsets.back() * *s;
                            offsets.push_back(o);
                        }

                        std::reverse(begin(offsets), end(offsets));
                        return offsets;
                    }();

                    assert(indices_shape.back() == indices_offsets.size());

                    const auto batch_offset = indices_offsets.front() * params_shape[batch_dims];

                    const Shape k_1_indecies(next(begin(indices_shape), batch_dims),
                                             prev(end(indices_shape)));

                    const Shape k_1_params(next(begin(params_shape), batch_dims),
                                           prev(end(params_shape)));

                    const auto number_of_slices_to_copy_in_one_batch = shape_size(k_1_indecies);

                    const auto coordinates_size = indices_shape.back();

                    for (size_t b = 0; b != batch_size; ++b)
                    {
                        const auto input_batch_offset = b * batch_offset;
                        const auto output_batch_offset =
                            b * number_of_slices_to_copy_in_one_batch * slice_size;
                        const auto coordinates_batch_offset =
                            b * number_of_slices_to_copy_in_one_batch * coordinates_size;
                        for (size_t i = 0; i != number_of_slices_to_copy_in_one_batch; ++i)
                        {
                            const auto coordinates =
                                indices + coordinates_batch_offset + i * coordinates_size;

                            size_t mem = input_batch_offset;
                            for (size_t c = 0; c != coordinates_size; ++c)
                            {
                                const auto i_c = coordinates[c];
                                const auto index = i_c < 0 ? k_1_params[c] + i_c : i_c;
                                mem += index * indices_offsets[c];
                            }
                            std::copy(next(params, mem),
                                      next(params, mem + slice_size),
                                      next(out, output_batch_offset + i * slice_size));
                        }
                    }
                }
            } // namespace new_impl

            class Timer
            {
            public:
                using Clock = std::chrono::high_resolution_clock;
                using TimePoint = std::chrono::time_point<Clock>;

                std::chrono::nanoseconds getPeriod() const { return now() - m_start_time; }

                std::string getPeriodStr() const
                {
                    return std::to_string(getPeriod().count()) + "ns";
                }

                static TimePoint now() { return Clock::now(); }

            private:
                TimePoint m_start_time = now();
            };

            template <typename T, typename U>
            void gather_nd(const T* const params,
                           const U* const indices,
                           T* const out,
                           const Shape& params_shape,
                           const Shape& indices_shape,
                           const Shape& out_shape,
                           const int batch_dims = 0)
            {
                {
                    Timer t{};
                    new_impl::gather_nd(
                        params, indices, out, params_shape, indices_shape, out_shape, batch_dims);
                    std::cout << "gather_nd new impl: " << t.getPeriodStr() << std::endl;
                }
                {
                    Timer t{};
                    old_impl::gather_nd(
                        params, indices, out, params_shape, indices_shape, out_shape, batch_dims);
                    std::cout << "gather_nd old impl: " << t.getPeriodStr() << std::endl;
                }
            }

        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
