//*****************************************************************************
// Copyright 2021 Intel Corporation
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

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            namespace def_conv_impl
            {
                inline void validate_params(const Shape& in_shape,
                                            const Shape& o_shape,
                                            const Shape& f_shape,
                                            const Strides& strides,
                                            const Strides& dilations,
                                            const CoordinateDiff& pads_begin,
                                            const CoordinateDiff& pads_end)
                {
                    // this implementation supports 2D deformable convolutions
                    NGRAPH_CHECK(in_shape.size() == 4, "Unsupported input rank: ", in_shape);

                    NGRAPH_CHECK(o_shape.size() == 4, "Unsupported offset rank: ", o_shape);

                    NGRAPH_CHECK(f_shape.size() == 4, "Unsupported kernel rank: ", f_shape);

                    const auto spatial_dims = in_shape.size() - 2;
                    NGRAPH_CHECK(strides.size() == spatial_dims,
                                 "Strides not definied for all and only spatial dimensions");

                    NGRAPH_CHECK(dilations.size() == spatial_dims,
                                 "Dilations not defined for all and only spatial dimensions");

                    NGRAPH_CHECK((pads_begin.size() == pads_end.size()) &&
                                     (pads_begin.size() == spatial_dims),
                                 "Pads not defined for all and only spatial dimensions");
                }

                inline Shape shape_reduce(const Shape& s) { return Shape(++s.begin(), s.end()); }

                inline Shape shape_scale(Shape s, size_t groups)
                {
                    s[0] /= groups;
                    return s;
                }

                template <typename T>
                void convolve_2D_channels(const ConvolutionParams& p,
                                          const T* batch,
                                          const Shape& batch_shape,
                                          const T* offsets,
                                          const Shape& offset_shape,
                                          const T* filter,
                                          const Shape& filter_shape,
                                          T* out)
                {
                    const int input_size_y = batch_shape[1];
                    const int input_size_x = batch_shape[2];
                    const int filter_size_y = filter_shape[1];
                    const int filter_size_x = filter_shape[2];
                    const int dilated_filter_size_y =
                        filter_size_y + (filter_size_y - 1) * (p.dilation[0] - 1);
                    const int dilated_filter_size_x =
                        filter_size_x + (filter_size_x - 1) * (p.dilation[1] - 1);

                    const int input_channel_size = shape_size(shape_reduce(batch_shape));
                    const int filter_channel_size = shape_size(shape_reduce(filter_shape));
                    const int offsets_spatial_size = shape_size(shape_reduce(offset_shape));

                    int out_idx = 0;
                    for (int i_y = -p.pads_begin[0];
                         i_y <= (p.pads_end[0] + input_size_y - dilated_filter_size_y);
                         i_y += p.strides[0])
                    {
                        for (int i_x = -p.pads_begin[1];
                             i_x <= (p.pads_end[1] + input_size_x - dilated_filter_size_x);
                             i_x += p.strides[1])
                        {
                            auto input_channel = batch;
                            auto filter_channel = filter;
                            T sum = 0;
                            int filter_channels_count = filter_shape[0];
                            while (filter_channels_count--)
                            {
                                auto offsets_channel = offsets;
                                for (int f_y = 0; f_y < filter_size_y; ++f_y)
                                {
                                    for (int f_x = 0; f_x < filter_size_x; ++f_x)
                                    {
                                        int y_offset = offsets_channel[out_idx];
                                        int x_offset =
                                            offsets_channel[offsets_spatial_size + out_idx];
                                        int rel_i_y = i_y + (f_y * p.dilation[0]) + y_offset;
                                        int rel_i_x = i_x + (f_x * p.dilation[1]) + x_offset;

                                        int f_buf_idx = (f_y * filter_size_x) + f_x;

                                        offsets_channel += (2 * offsets_spatial_size);
                                        bool padding = !(in_range(rel_i_x, {0, input_size_x}) &&
                                                         in_range(rel_i_y, {0, input_size_y}));
                                        if (padding)
                                            continue;

                                        int i_buf_idx = (rel_i_y * input_size_x) + rel_i_x;
                                        sum += input_channel[i_buf_idx] * filter_channel[f_buf_idx];
                                    }
                                }

                                input_channel += input_channel_size;
                                filter_channel += filter_channel_size;
                            }
                            out[out_idx++] = sum;
                        }
                    }
                }

            }
            template <typename T>
            void deformable_convolution(const T* in,
                                        const T* offsets,
                                        const T* filters,
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
                using namespace def_conv_impl;

                validate_params(
                    in_shape, o_shape, f_shape, strides, dilation, pads_begin, pads_end);

                // here we are converting all param types to int's to avoid arithmetic issues
                // (e.g signed + unsigned) in indexes calculation later
                ConvolutionParams params{strides, dilation, pads_begin, pads_end};

                const size_t batches_count = in_shape[in_batch_axis];
                const Shape group_in_shape = shape_scale(shape_reduce(in_shape), groups);
                const size_t group_in_size = shape_size(group_in_shape);

                const Shape group_offset_shape =
                    shape_scale(shape_reduce(o_shape), deformable_groups);
                const size_t batch_defgroup_offset_size = shape_size(group_offset_shape);

                const size_t group_filters_count = f_shape[filter_out_ch_axis] / groups;
                const Shape group_filter_shape = shape_reduce(shape_scale(f_shape, groups));
                const size_t group_filter_size = shape_size(group_filter_shape);

                const size_t out_ch_size = shape_size(shape_reduce(shape_reduce(out_shape)));

                for (size_t batch_idx = 0; batch_idx < batches_count; ++batch_idx)
                {
                    const T* group_filters = filters;
                    for (size_t group_idx = 0; group_idx < groups; ++group_idx)
                    {
                        for (size_t f_idx = 0; f_idx < group_filters_count; ++f_idx)
                        {
                            convolve_2D_channels(params,
                                                 in,
                                                 group_in_shape,
                                                 offsets,
                                                 group_offset_shape,
                                                 group_filters,
                                                 group_filter_shape,
                                                 out);
                            group_filters += group_filter_size;
                            out += out_ch_size;
                        }
                        in += group_in_size;
                    }
                    offsets += batch_defgroup_offset_size;
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph