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
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            namespace
            {
                template <typename T>
                void deformable_convolve_3D_channels(const ConvolutionParams& p,
                                                     const T* batch,
                                                     const Shape& batch_shape,
                                                     const T* offset,
                                                     const Shape& offset_shape,
                                                     const T* filter,
                                                     const Shape& filter_shape,
                                                     T*& out)
                {
                    const int input_size_z = batch_shape[1];
                    const int input_size_y = batch_shape[2];
                    const int input_size_x = batch_shape[3];
                    const int filter_size_z = filter_shape[1];
                    const int filter_size_y = filter_shape[2];
                    const int filter_size_x = filter_shape[3];
                    const int dilated_filter_size_z =
                        filter_size_z + (filter_size_z - 1) * (p.dilation[0] - 1);
                    const int dilated_filter_size_y =
                        filter_size_y + (filter_size_y - 1) * (p.dilation[1] - 1);
                    const int dilated_filter_size_x =
                        filter_size_x + (filter_size_x - 1) * (p.dilation[2] - 1);

                    const Shape input_channel_shape(++batch_shape.begin(), batch_shape.end());
                    const size_t input_channel_size = shape_size(input_channel_shape);
                    const Shape filter_channel_shape(++filter_shape.begin(), filter_shape.end());
                    const size_t filter_channel_size = shape_size(filter_channel_shape);
                    int f_c = 0;
                    int off = 0;
                    const Shape offset_spatial_dims_shape(++offset_shape.begin(),
                                                          offset_shape.end());
                    const int spatial_size = shape_size(offset_spatial_dims_shape);
                    for (int i_z = -p.pads_begin[0];
                         i_z <= (p.pads_end[0] + input_size_z - dilated_filter_size_z);
                         i_z += p.strides[0])
                    {
                        for (int i_y = -p.pads_begin[1];
                             i_y <= (p.pads_end[1] + input_size_y - dilated_filter_size_y);
                             i_y += p.strides[1])
                        {
                            for (int i_x = -p.pads_begin[2];
                                 i_x <= (p.pads_end[2] + input_size_x - dilated_filter_size_x);
                                 i_x += p.strides[2])
                            {
                                auto input_channel = batch;
                                auto filter_channel = filter;
                                T sum = 0;
                                size_t filter_channels_count = filter_shape[0];
                                while (filter_channels_count--)
                                {
                                    off = 0;
                                    for (int f_z = 0; f_z < filter_size_z; ++f_z)
                                    {
                                        for (int f_y = 0; f_y < filter_size_y; ++f_y)
                                        {
                                            for (int f_x = 0; f_x < filter_size_x; ++f_x)
                                            {
                                                int rel_i_z = i_z + (f_z * p.dilation[0]);
                                                int rel_i_y = i_y + (f_y * p.dilation[1]);
                                                int rel_i_x = i_x + (f_x * p.dilation[2]);

                                                int f_buf_idx =
                                                    (f_z * filter_size_y * filter_size_x) +
                                                    (f_y * filter_size_x) + f_x;

                                                int y_offset =
                                                    offset[(f_buf_idx + off++) * spatial_size +
                                                           f_c];
                                                int x_offset =
                                                    offset[(f_buf_idx + off) * spatial_size + f_c];

                                                if (x_offset + rel_i_x >= input_size_x ||
                                                    y_offset + rel_i_y >= input_size_y)
                                                    continue;
                                                if (x_offset + rel_i_x < 0 ||
                                                    y_offset + rel_i_y < 0)
                                                    continue;

                                                int i_buf_idx =
                                                    (rel_i_z * input_size_y * input_size_x) +
                                                    ((rel_i_y + y_offset) * input_size_x) +
                                                    rel_i_x + x_offset;
                                                sum += static_cast<T>(input_channel[i_buf_idx]) *
                                                       static_cast<T>(filter_channel[f_buf_idx]);
                                            }
                                        }
                                    }
                                    input_channel += input_channel_size;
                                    filter_channel += filter_channel_size;
                                }
                                f_c++;
                                *out = sum;
                                ++out;
                            }
                        }
                    }
                }

                void validate_deformable_convolution_parameters(const Shape& in_shape,
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

                void extend_to_3D(ConvolutionParams& p,
                                  Shape& in_shape,
                                  Shape& offset_shape,
                                  Shape& filter_shape)
                {
                    int spatial_rank = in_shape.size() - 2;
                    if (spatial_rank < 3)
                    {
                        int missing_dims = 3 - spatial_rank;
                        p.dilation.insert(
                            std::prev(p.dilation.end(), spatial_rank), missing_dims, 1);
                        p.strides.insert(std::prev(p.strides.end(), spatial_rank), missing_dims, 1);
                        p.pads_begin.insert(
                            std::prev(p.pads_begin.end(), spatial_rank), missing_dims, 0);
                        p.pads_end.insert(
                            std::prev(p.pads_end.end(), spatial_rank), missing_dims, 0);
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
                    // here we are converting all param types to int's to avoid arithmetic issues
                    // (e.g signed + unsigned) in indexes calculation later
                    ConvolutionParams params{strides, dilation, pads_begin, pads_end};
                    // here we are extending spatial dimensions to 3D, because we are going to use
                    // 3D
                    // convolution implementation to convolve also in 1D & 2D case
                    Shape input_shape{in_shape};
                    Shape offsets_shape(o_shape);
                    Shape filters_shape{f_shape};
                    if (in_shape.size() < 5)
                    {
                        extend_to_3D(params, input_shape, offsets_shape, filters_shape);
                    }

                    const size_t batches_count = input_shape[in_batch_axis];
                    const Shape batch_shape(++input_shape.begin(), input_shape.end());
                    const size_t batch_size = shape_size(batch_shape);

                    const Shape offset_batch_shape(++offsets_shape.begin(), offsets_shape.end());
                    const size_t offset_batch_size = shape_size(offset_batch_shape);

                    const size_t filters_count = filters_shape[filter_out_ch_axis];
                    const Shape filter_shape(++filters_shape.begin(), filters_shape.end());
                    const size_t filter_size = shape_size(filter_shape);

                    auto batch = in;
                    auto offsets = o;
                    for (size_t batch_idx = 0; batch_idx < batches_count; ++batch_idx)
                    {
                        auto filter = f;
                        for (size_t f_idx = 0; f_idx < filters_count; ++f_idx)
                        {
                            deformable_convolve_3D_channels(params,
                                                            batch,
                                                            batch_shape,
                                                            offsets,
                                                            offset_batch_shape,
                                                            filter,
                                                            filter_shape,
                                                            out);
                            filter += filter_size;
                        }
                        batch += batch_size;
                        offsets += offset_batch_size;
                    }
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
                validate_deformable_convolution_parameters(
                    in_shape, o_shape, f_shape, strides, dilation, pads_begin, pads_end);

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
                // const size_t group_offset_size = shape_size(group_batch_shape); TODO

                const T* group_filter = f;
                const Shape group_filter_shape = [&]() {
                    Shape new_shape{f_shape};
                    new_shape[filter_out_ch_axis] /= groups;
                    new_shape[filter_in_ch_axis] /= groups;
                    return new_shape;
                }();
                const size_t group_filter_size = shape_size(group_filter_shape) * groups;

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
                                               group_filter_shape,
                                               group_out_shape,
                                               strides,
                                               dilation,
                                               pads_begin,
                                               pads_end);
                        group_batch += group_batch_size;
                        // group_offset += group_offset_size; TODO
                        group_filter += group_filter_size;
                        group_out += group_out_size;
                    }
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph