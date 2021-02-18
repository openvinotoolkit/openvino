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

#include "ngraph/runtime/reference/convolution.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void extend_to_3D(ConvolutionParams& p,
                              Shape& in_shape,
                              Shape& offset_shape,
                              Shape& filter_shape)
            {
                int spatial_rank = in_shape.size() - 2;
                if (spatial_rank < 3)
                {
                    int missing_dims = 3 - spatial_rank;
                    p.dilation.insert(std::prev(p.dilation.end(), spatial_rank), missing_dims, 1);
                    p.strides.insert(std::prev(p.strides.end(), spatial_rank), missing_dims, 1);
                    p.pads_begin.insert(
                        std::prev(p.pads_begin.end(), spatial_rank), missing_dims, 0);
                    p.pads_end.insert(std::prev(p.pads_end.end(), spatial_rank), missing_dims, 0);
                    in_shape.insert(std::next(in_shape.end(), -spatial_rank), missing_dims, 1);
                    offset_shape.insert(
                        std::prev(offset_shape.end(), spatial_rank), missing_dims, 1);
                    filter_shape.insert(
                        std::prev(filter_shape.end(), spatial_rank), missing_dims, 1);
                }
            }

            template <typename T>
            void deformable_convolution(const T* in,
                                        const T* f,
                                        const T* o,
                                        T* out,
                                        const Shape& in_shape,
                                        const Shape& o_shape,
                                        const Shape& f_shape,
                                        const Shape& out_shape,
                                        const Strides& strides,
                                        const Strides& dilation,
                                        const CoordinateDiff& pads_begin,
                                        const CoordinateDiff& pads_end)

            {
                // this implementation supports 2D deformable convolutions
                NGRAPH_CHECK(in_shape.size() == 4, "Unsupported input rank: ", in_shape);

                NGRAPH_CHECK(o_shape.size() == 4, "Unsupported offset rank: ", o_shape);

                NGRAPH_CHECK(f_shape.size() == 4, "Unsupported kernel rank: ", f_shape);

                // here we are converting all param types to int's to avoid arithmetic issues
                // (e.g signed + unsigned) in indexes calculation later
                ConvolutionParams params{strides, dilation, pads_begin, pads_end};

                // here we are extending spatial dimensions to 3D, because we are going to use 3D
                // convolution implementation to convolve also in 1D & 2D case
                Shape input_shape{in_shape};
                Shape offset_shape(o_shape);
                Shape filters_shape{f_shape};
                if (in_shape.size() < 5)
                {
                    extend_to_3D(params, input_shape, offset_shape, filters_shape);
                }

                const size_t batches_count = input_shape[in_batch_axis];
                const Shape batch_shape(++input_shape.begin(), input_shape.end());
                const size_t batch_size = shape_size(batch_shape);

                const size_t filters_count = filters_shape[filter_out_ch_axis];
                const Shape filter_shape(++filters_shape.begin(), filters_shape.end());
                const size_t filter_size = shape_size(filter_shape);

                auto batch = in;
                for (size_t batch_idx = 0; batch_idx < batches_count; ++batch_idx)
                {
                    auto filter = f;
                    for (size_t f_idx = 0; f_idx < filters_count; ++f_idx)
                    {
                        convolve_3D_channels(params, batch, batch_shape, filter, filter_shape, out);
                        filter += filter_size;
                    }
                    batch += batch_size;
                }
            }

            template <typename T>
            void deformable_convolution(const T* in,
                                        const T* o,
                                        const T* f,
                                        T* out,
                                        const Shape& in_shape,
                                        const Shape& o_shape,
                                        const Shape& f_shape,
                                        const Shape& out_shape,
                                        const Strides& strides,
                                        const Strides& dilation,
                                        const CoordinateDiff& pads_begin,
                                        const CoordinateDiff& pads_end,
                                        const int64_t groups,
                                        const int64_t deformable_groups)

            {
                // this implementation supports 2D convolutions
                NGRAPH_CHECK(in_shape.size() == 4, "Unsupported input rank: ", in_shape);

                NGRAPH_CHECK(o_shape.size() == 4, "Unsupported offset rank: ", o_shape);

                NGRAPH_CHECK(f_shape.size() == 4, "Unsupported kernel rank: ", f_shape);

                const T* group_batch = in;
                const Shape group_batch_shape = [&]() {
                    Shape new_shape{in_shape};
                    new_shape[in_batch_axis] = 1;
                    new_shape[in_channel_axis] /= groups;
                    return new_shape;
                }();
                const size_t group_batch_size = shape_size(group_batch_shape);

                const T* group_offset = o;
                const Shape group_offset_shape = [&]() {
                    Shape new_shape{o_shape};
                    new_shape[in_batch_axis] = 1;
                    new_shape[in_channel_axis] /= deformable_groups;
                    return new_shape;
                }();
                const size_t group_offset_size = shape_size(group_batch_shape);

                const T* group_filter = f;
                const size_t group_filter_size = shape_size(f_shape);

                T* group_out = out;
                const Shape group_out_shape = [&]() {
                    Shape new_shape{out_shape};
                    new_shape[out_batch_axis] = 1;
                    new_shape[out_channel_axis] /= groups;
                    return new_shape;
                }();
                const size_t group_out_size = shape_size(group_out_shape);
                for (size_t batch_idx = 0; batch_idx < in_shape[in_batch_axis]; ++batch_idx)
                {
                    group_filter = f;
                    for (size_t group_idx = 0; group_idx < groups; ++group_idx)
                    {
                        deformable_convolution(group_batch,
                                               group_filter,
                                               group_offset,
                                               group_out,
                                               group_batch_shape,
                                               group_offset_shape,
                                               f_shape,
                                               group_out_shape,
                                               strides,
                                               dilation,
                                               pads_begin,
                                               pads_end);
                        group_batch += group_batch_size;
                        group_offset += group_offset_size;
                        group_filter += group_filter_size;
                        group_out += group_out_size;
                    }
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph