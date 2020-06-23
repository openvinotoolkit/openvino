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

#include <cfenv>
#include <cmath>
#include <functional>
#include <numeric>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/reference/reverse.hpp"
#include "ngraph/runtime/reference/concat.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            struct widen
            {
                using type = T;
            };

            template <>
            struct widen<float>
            {
                using type = double;
            };

            template <>
            struct widen<double>
            {
                using type = long double;
            };

            // in: NC_I...
            // filter: C_OC_I...
            // out: NC_O...
            template <typename INPUT,
                      typename FILTER,
                      typename OUTPUT,
                      typename ACCUMULATION = typename widen<OUTPUT>::type>
            void general_convolution(const INPUT* in,
                                     const FILTER* filter,
                                     OUTPUT* out,
                                     const Shape& in_shape,
                                     const Shape& filter_shape,
                                     const Shape& out_shape,
                                     const Strides& stride,
                                     const Strides& filter_dilation,
                                     const CoordinateDiff& in_pad_below,
                                     const CoordinateDiff& in_pad_above,
                                     const Strides& in_dilation,
                                     size_t num_groups,
                                     size_t in_batch_axis,
                                     size_t in_channel_axis,
                                     size_t filter_out_channel_axis,
                                     size_t filter_in_channel_axis,
                                     size_t out_batch_axis,
                                     size_t out_channel_axis,
                                     const float* input_scale = nullptr,
                                     const INPUT* input_zero_point = nullptr,
                                     const float* filter_scale = nullptr,
                                     const FILTER* filter_zero_point = nullptr,
                                     const float* output_scale = nullptr,
                                     const OUTPUT* output_zero_point = nullptr)
            {
                bool is_quantized = false;
                if (input_scale && input_zero_point && filter_scale && filter_zero_point &&
                    output_scale && output_zero_point)
                {
                    is_quantized = true;
                }

                auto old_mode = std::fegetround();

                Shape group_out_shape(out_shape);
                Shape group_in_shape(in_shape);
                Shape filter_group_shape(filter_shape);
                size_t filter_groups_stride = 0;
                size_t channels_in_group = in_shape[in_channel_axis];
                std::vector<std::vector<OUTPUT>> result_groups(num_groups);
                if (num_groups > 1) {
                    NGRAPH_CHECK(in_shape[in_channel_axis] % num_groups == 0,
                                 "Number of input channels and number of groups must be multiplies of each other");
                    channels_in_group = in_shape[in_channel_axis] / num_groups;
                    group_out_shape[out_channel_axis] = filter_shape.at(filter_out_channel_axis);
                    group_in_shape[in_channel_axis] = channels_in_group;
                    filter_group_shape = Shape(std::vector<size_t>(filter_shape.begin() + 1, filter_shape.end()));
                    filter_groups_stride = std::accumulate(filter_shape.begin() + 1, filter_shape.end(), 1,
                                                           std::multiplies<size_t>());
                    // Further we will operate with filter_group_shape which doesn't have groups dimension
                    filter_out_channel_axis -= 1;
                    filter_in_channel_axis -= 1;

                }




                std::fesetround(FE_TONEAREST);
                // Comments throughout assume without loss of generality that:
                //
                // * batch axes for both in and out are 0
                // * in channel axes for both in and filter are 1
                // * out channel axes for filter is 0
                // * out channel axis for out is 1

                // At the outermost level we will walk over every out coordinate O.
                CoordinateTransform out_transform(group_out_shape);
                for (size_t g = 0; g < num_groups; g++) {
                    const FILTER *filter_group_data = filter + filter_groups_stride * g;
                    result_groups[g].resize(shape_size(group_out_shape));
                    const size_t ch_start = channels_in_group * g;
                    const size_t ch_end = channels_in_group * (g + 1);

                    for (const Coordinate &out_coord : out_transform) {
                        // Our out coordinate O will have the form:
                        //
                        //   (N,chan_out,i_1,...,i_n)

                        size_t batch_index = out_coord[out_batch_axis];
                        size_t out_channel = out_coord[out_channel_axis];

                        // For the in we need to iterate the coordinate:
                        //
                        //   I:
                        //
                        // over the range (noninclusive on the right):
                        //
                        //   (N,0,s_1*i_1,s_2*i_2,...,s_n*i_n) ->
                        //
                        //     (N+1,
                        //      chans_in_count,
                        //      s_1*i_1+ l_1*filter_dims_1,
                        ///       ...,
                        ///     s_n*i_n +l_n*filter_dims_n)
                        //
                        // with strides:
                        //
                        //   (1,l_1,...,l_n).
                        //
                        // Note that we are iterating within the *padded* and *dilated* in batch, so
                        // further down we must check the current coordinate is in the pad or dilation
                        // gap.

                        size_t n_spatial_dimensions = group_in_shape.size() - 2;
                        size_t n_in_channels = group_in_shape[in_channel_axis];

                        Coordinate in_transform_start(2 + n_spatial_dimensions);
                        Coordinate in_transform_end(2 + n_spatial_dimensions);
                        Strides in_transform_movement_strides(2 + n_spatial_dimensions, 1);
                        CoordinateDiff in_transform_pad_below(2 + n_spatial_dimensions, 0);
                        CoordinateDiff in_transform_pad_above(2 + n_spatial_dimensions, 0);
                        Strides in_transform_dilation_strides(2 + n_spatial_dimensions, 1);

                        in_transform_start[in_batch_axis] = batch_index;
                        in_transform_end[in_batch_axis] = batch_index + 1;
                        in_transform_start[in_channel_axis] = 0;
                        in_transform_end[in_channel_axis] = 1;

                        for (size_t i = 2; i < n_spatial_dimensions + 2; i++) {
                            size_t filter_dilation_stride = filter_dilation[i - 2];
                            size_t filter_movement_stride = stride[i - 2];
                            std::ptrdiff_t below_pad = in_pad_below[i - 2];
                            std::ptrdiff_t above_pad = in_pad_above[i - 2];
                            size_t in_dilation_stride = in_dilation[i - 2];

                            in_transform_start[i] = filter_movement_stride * out_coord[i];
                            in_transform_end[i] = in_transform_start[i] +
                                                  (filter_group_shape[i] - 1) * filter_dilation_stride + 1;
                            in_transform_movement_strides[i] = filter_dilation_stride;
                            in_transform_pad_below[i] = below_pad;
                            in_transform_pad_above[i] = above_pad;
                            in_transform_dilation_strides[i] = in_dilation_stride;
                        }

                        AxisVector in_transform_axis_order(2 + n_spatial_dimensions);
                        for (size_t i = 0; i < in_transform_axis_order.size(); i++) {
                            in_transform_axis_order[i] = i;
                        }
                        CoordinateTransform in_transform(group_in_shape,
                                                         in_transform_start,
                                                         in_transform_end,
                                                         in_transform_movement_strides,
                                                         in_transform_axis_order,
                                                         in_transform_pad_below,
                                                         in_transform_pad_above,
                                                         in_transform_dilation_strides);

                        // Simultaneously with iterating I, for the filter we need to iterate the
                        // coordinate:
                        //
                        //   F
                        //
                        // over the range (noninclusive on the right):
                        //
                        //   (chan_out,0,0,...,0) ->
                        //     (chan_out+1,
                        //      chans_in_count,
                        //      filter_dims_1,
                        //        ...,
                        //      filter_dims_n)
                        //
                        // with unit stride.

                        Shape filter_transform_start(2 + n_spatial_dimensions);
                        Shape filter_transform_end(2 + n_spatial_dimensions);

                        filter_transform_start[filter_out_channel_axis] = out_channel;
                        filter_transform_end[filter_out_channel_axis] = out_channel + 1;
                        filter_transform_start[filter_in_channel_axis] = 0;
                        filter_transform_end[filter_in_channel_axis] = 1;

                        for (size_t i = 2; i < n_spatial_dimensions + 2; i++) {
                            filter_transform_start[i] = 0;
                            filter_transform_end[i] = filter_group_shape[i];
                        }

                        CoordinateTransform filter_transform(
                                filter_group_shape, filter_transform_start, filter_transform_end);

                        // As we go, we sum up:
                        //
                        //   out[O] += in[I] * filter[F].

                        ACCUMULATION result = 0;

                        CoordinateTransform::Iterator in_it = in_transform.begin();
                        CoordinateTransform::Iterator filter_it = filter_transform.begin();
                        CoordinateTransform::Iterator in_it_end = in_transform.end();
                        CoordinateTransform::Iterator filter_it_end = filter_transform.end();

                        size_t in_channel_stride = row_major_strides(group_in_shape).at(in_channel_axis);
                        size_t filter_in_channel_stride =
                                row_major_strides(filter_group_shape).at(filter_in_channel_axis);
                        size_t group_channel_offset = in_channel_stride * channels_in_group * g;
                        while (in_it != in_it_end && filter_it != filter_it_end) {
                            const Coordinate &in_coord = *in_it;
                            if (in_transform.has_source_coordinate(in_coord)) {
                                size_t in_idx = in_transform.index(in_coord) + group_channel_offset;
                                const Coordinate &filter_coord = *filter_it;
                                size_t filter_idx = filter_transform.index(filter_coord);
                                for (size_t in_channel = ch_start; in_channel < ch_end; ++in_channel) {
                                    ACCUMULATION in_v = static_cast<ACCUMULATION>(in[in_idx]);
                                    ACCUMULATION f_v = static_cast<ACCUMULATION>(filter_group_data[filter_idx]);
                                    if (is_quantized) {
                                        in_v = in_v - static_cast<ACCUMULATION>(*input_zero_point);
                                        f_v = f_v - static_cast<ACCUMULATION>(*filter_zero_point);
                                    }
                                    result += in_v * f_v;
                                    in_idx += in_channel_stride;
                                    filter_idx += filter_in_channel_stride;
                                }
                            }
                            ++in_it;
                            ++filter_it;
                        }
                        if (is_quantized) {
                            float scale = *input_scale * *filter_scale / *output_scale;
                            result_groups[g][out_transform.index(out_coord)] =
                                    static_cast<OUTPUT>(std::round(static_cast<float>(result) * scale)) +
                                    *output_zero_point;
                        } else {
                            result_groups[g][out_transform.index(out_coord)] = result;
                        }
                    }
                }
                if (num_groups > 1){
                    std::vector<const OUTPUT*> const_results_cpy;
                    std::vector<Shape> in_shapes;
                    for (size_t g = 0; g < num_groups; g++){
                        const_results_cpy.push_back(result_groups[g].data());
                        in_shapes.push_back(group_out_shape);
                    }
                    concat<OUTPUT>(const_results_cpy, out, in_shapes, Shape(out_shape), in_channel_axis);
                } else {
                    std::copy(result_groups[0].data(), result_groups[0].data() + shape_size(out_shape), out);
                }

                std::fesetround(old_mode);
            }

            template <typename INPUT,
                      typename FILTER,
                      typename OUTPUT,
                      typename ACCUMULATION = typename widen<OUTPUT>::type>
            void convolution(const INPUT* in,
                             const FILTER* filter,
                             OUTPUT* out,
                             const Shape& in_shape,
                             const Shape& filter_shape,
                             const Shape& out_shape,
                             const Strides& stride,
                             const Strides& filter_dilation,
                             const CoordinateDiff& in_pad_below,
                             const CoordinateDiff& in_pad_above,
                             const Strides& in_dilation,
                             size_t num_groups = 1,
                             const float* input_scale = nullptr,
                             const INPUT* input_zero_point = nullptr,
                             const float* filter_scale = nullptr,
                             const FILTER* filter_zero_point = nullptr,
                             const float* output_scale = nullptr,
                             const OUTPUT* output_zero_point = nullptr)

            {
                size_t filter_out_channel_axis = num_groups == 1 ? 0 : 1;
                size_t filter_in_channel_axis = num_groups == 1 ? 1 : 2;
                general_convolution<INPUT, FILTER, OUTPUT, ACCUMULATION>(in,
                                                                         filter,
                                                                         out,
                                                                         in_shape,
                                                                         filter_shape,
                                                                         out_shape,
                                                                         stride,
                                                                         filter_dilation,
                                                                         in_pad_below,
                                                                         in_pad_above,
                                                                         in_dilation,
                                                                         num_groups,
                                                                         0,
                                                                         1,
                                                                         filter_out_channel_axis,
                                                                         filter_in_channel_axis,
                                                                         0,
                                                                         1,
                                                                         input_scale,
                                                                         input_zero_point,
                                                                         filter_scale,
                                                                         filter_zero_point,
                                                                         output_scale,
                                                                         output_zero_point);
            }

            template <typename OUTPUT,
                      typename FILTER,
                      typename INPUT,
                      typename ACCUMULATION = typename widen<INPUT>::type>
            void convolution_backprop_in(const OUTPUT* delta_out,
                                         const FILTER* filter,
                                         INPUT* delta_in,
                                         const Shape& out_shape,
                                         const Shape& filter_shape,
                                         const Shape& in_shape,
                                         const Strides& in_dilation,
                                         const Strides& filter_dilation,
                                         const CoordinateDiff& backward_delta_out_pad_below,
                                         const CoordinateDiff& backward_delta_out_pad_above,
                                         const Strides& stride,
                                         size_t num_groups = 1)
            {
                // Note that we only reverse the spatial dimensions here (loop
                // starts at 2)
                std::vector<INPUT> reversed(shape_size(filter_shape));
                AxisSet reverse_axes;
                for (size_t i = 2; i < filter_shape.size(); ++i)
                {
                    reverse_axes.insert(i);
                }
                reverse<FILTER>(filter, &reversed[0], filter_shape, filter_shape, reverse_axes);
                size_t filter_out_channel_axis = num_groups == 1 ? 1 : 2;
                size_t filter_in_channel_axis = num_groups == 1 ? 0 : 1;
                general_convolution<OUTPUT, FILTER, INPUT, ACCUMULATION>(
                    delta_out,
                    &reversed[0],
                    delta_in,
                    out_shape,
                    filter_shape,
                    in_shape,
                    in_dilation,
                    filter_dilation,
                    backward_delta_out_pad_below,
                    backward_delta_out_pad_above,
                    stride,
                    num_groups,
                    0,
                    1,
                    filter_out_channel_axis,
                    filter_in_channel_axis,
                    0,
                    1);
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
