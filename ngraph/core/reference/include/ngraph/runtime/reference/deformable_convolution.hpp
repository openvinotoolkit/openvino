// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
                inline void validate_deformable_convolution_params(const Shape& in_shape,
                                                                   const Shape& o_shape,
                                                                   const Shape& f_shape,
                                                                   const Shape& out_shape,
                                                                   const Strides& strides,
                                                                   const Strides& dilations,
                                                                   const CoordinateDiff& pads_begin,
                                                                   const CoordinateDiff& pads_end,
                                                                   const int64_t groups,
                                                                   const int64_t deformable_groups)
                {
                    // this implementation supports 2D deformable convolutions
                    NGRAPH_CHECK(in_shape.size() == 4, "Unsupported input rank: ", in_shape);
                    NGRAPH_CHECK(o_shape.size() == 4, "Unsupported offset rank: ", o_shape);
                    NGRAPH_CHECK(f_shape.size() == 4, "Unsupported kernel rank: ", f_shape);

                    NGRAPH_CHECK(in_shape[1] % groups == 0,
                                 "Input channels of data batch input must be evenly divisible by "
                                 "'groups' attribute");
                    NGRAPH_CHECK(f_shape[0] % groups == 0,
                                 "Output channels of filters must be evenly divisible by 'groups' "
                                 "attribute");

                    const Shape scaled_f_shape = [f_shape](int64_t g) {
                        Shape shape{f_shape};
                        shape[1] *= g;
                        return shape;
                    }(groups);

                    validate_convolution_parameters(in_shape,
                                                    scaled_f_shape,
                                                    out_shape,
                                                    strides,
                                                    dilations,
                                                    pads_begin,
                                                    pads_end);

                    const Shape f_spatial_shape{std::next(f_shape.begin(), 2), std::end(f_shape)};
                    const Shape o_spatial_shape{std::next(o_shape.begin(), 2), std::end(o_shape)};
                    const Shape out_spatial_shape{std::next(out_shape.begin(), 2),
                                                  std::end(out_shape)};

                    NGRAPH_CHECK(o_shape[1] == deformable_groups * shape_size(f_spatial_shape) * 2,
                                 "The channels dimension of offsets input is not "
                                 "compatible with filters and 'deformable group' attribute");
                    NGRAPH_CHECK(out_spatial_shape == o_spatial_shape,
                                 "Spatial dimensions of output and offsets values must be equal");
                }

                inline Shape shape_reduce(const Shape& s) { return Shape(++s.begin(), s.end()); }

                inline Shape shape_scale(Shape s, size_t groups)
                {
                    s[0] /= groups;
                    return s;
                }

                template <typename inputType>
                inline float bilinear_interpolation(const inputType* data,
                                                    const float x_idx,
                                                    const float y_idx,
                                                    const int x_size,
                                                    const int y_size)
                {
                    const int x1 = std::max(static_cast<int>(std::floor(x_idx)), 0);
                    const int x2 = std::min(static_cast<int>(std::ceil(x_idx)), x_size - 1);
                    const int y1 = std::max(static_cast<int>(std::floor(y_idx)), 0);
                    const int y2 = std::min(static_cast<int>(std::ceil(y_idx)), y_size - 1);

                    const float distX = x_idx - x1;
                    const float distY = y_idx - y1;

                    const float value11 = data[y1 * x_size + x1];
                    const float value12 = data[y2 * x_size + x1];
                    const float value21 = data[y1 * x_size + x2];
                    const float value22 = data[y2 * x_size + x2];

                    const float value = (1 - distX) * (1 - distY) * value11 +
                                        (1 - distX) * distY * value12 +
                                        distX * (1 - distY) * value21 + distX * distY * value22;
                    return value;
                }

                template <typename T>
                void convolve_2D_channels(const ConvolutionParams& p,
                                          const int64_t deformable_groups,
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
                    const int offsets_size = shape_size(offset_shape);
                    const int offsets_spatial_size = shape_size(shape_reduce(offset_shape));
                    const int offsets_channel_size = 2 * offsets_spatial_size;
                    const int filter_channels_count = filter_shape[0];

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
                            auto group_offsets_channel = offsets;
                            for (int dg = 0; dg < deformable_groups; dg++)
                            {
                                for (int fc = 0; fc < filter_channels_count / deformable_groups;
                                     fc++)
                                {
                                    auto offsets_channel = group_offsets_channel;
                                    for (int f_y = 0; f_y < filter_size_y; ++f_y)
                                    {
                                        for (int f_x = 0; f_x < filter_size_x; ++f_x)
                                        {
                                            T y_offset = offsets_channel[out_idx];
                                            T x_offset =
                                                offsets_channel[offsets_spatial_size + out_idx];
                                            T rel_i_y = i_y + (f_y * p.dilation[0]) + y_offset;
                                            T rel_i_x = i_x + (f_x * p.dilation[1]) + x_offset;

                                            offsets_channel += offsets_channel_size;
                                            bool padding = !(in_range(rel_i_x, {0, input_size_x}) &&
                                                             in_range(rel_i_y, {0, input_size_y}));
                                            if (padding)
                                                continue;

                                            int f_buf_idx = (f_y * filter_size_x) + f_x;
                                            sum += bilinear_interpolation(input_channel,
                                                                          rel_i_x,
                                                                          rel_i_y,
                                                                          input_size_x,
                                                                          input_size_y) *
                                                   filter_channel[f_buf_idx];
                                        }
                                    }
                                    input_channel += input_channel_size;
                                    filter_channel += filter_channel_size;
                                }
                                group_offsets_channel += offsets_size / deformable_groups;
                            }
                            out[out_idx++] = sum;
                        }
                    }
                }

            } // namespace def_conv_impl
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

                validate_deformable_convolution_params(in_shape,
                                                       o_shape,
                                                       f_shape,
                                                       out_shape,
                                                       strides,
                                                       dilation,
                                                       pads_begin,
                                                       pads_end,
                                                       groups,
                                                       deformable_groups);

                // here we are converting all param types to int's to avoid arithmetic issues
                // (e.g signed + unsigned) in indexes calculation later
                ConvolutionParams params{strides, dilation, pads_begin, pads_end};
                const size_t groups_count = static_cast<size_t>(groups);

                const size_t batches_count = in_shape[in_batch_axis];
                const Shape group_in_shape = shape_scale(shape_reduce(in_shape), groups);
                const size_t group_in_size = shape_size(group_in_shape);

                const Shape group_offset_shape = shape_scale(shape_reduce(o_shape), groups);
                const size_t group_offset_size = shape_size(group_offset_shape);
                const size_t group_offset_batch_size = shape_size(shape_reduce(o_shape));
                const size_t deformable_groups_per_group =
                    std::ceil(static_cast<float>(deformable_groups) / static_cast<float>(groups));

                const size_t group_filters_count = f_shape[filter_out_ch_axis] / groups;
                const Shape group_filter_shape = shape_reduce(f_shape);
                const size_t group_filter_size = shape_size(group_filter_shape);

                const size_t out_ch_size = shape_size(shape_reduce(shape_reduce(out_shape)));

                for (size_t batch_idx = 0; batch_idx < batches_count; ++batch_idx)
                {
                    const T* group_filters = filters;
                    const T* group_offsets = offsets;
                    for (size_t group_idx = 0; group_idx < groups_count; ++group_idx)
                    {
                        for (size_t f_idx = 0; f_idx < group_filters_count; ++f_idx)
                        {
                            convolve_2D_channels(params,
                                                 deformable_groups_per_group,
                                                 in,
                                                 group_in_shape,
                                                 group_offsets,
                                                 group_offset_shape,
                                                 group_filters,
                                                 group_filter_shape,
                                                 out);
                            group_filters += group_filter_size;
                            out += out_ch_size;
                        }
                        in += group_in_size;
                        if (deformable_groups > 1)
                        {
                            group_offsets += (deformable_groups_per_group * group_offset_size);
                        }
                    }
                    offsets += group_offset_batch_size;
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
