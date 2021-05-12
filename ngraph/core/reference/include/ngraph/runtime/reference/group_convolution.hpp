// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/runtime/reference/convolution.hpp"
#include "ngraph/util.hpp"

namespace
{
    constexpr size_t filter_group_axis = 0;
    constexpr size_t filter_in_ch_axis = 2;
    constexpr size_t in_batch_axis = 0;
    constexpr size_t in_channel_axis = 1;
    constexpr size_t out_batch_axis = 0;
    constexpr size_t out_channel_axis = 1;
} // namespace

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void validate_group_convolution_parameters(const Shape& in_shape,
                                                       const Shape& f_shape,
                                                       const Shape& out_shape,
                                                       const Strides& strides,
                                                       const Strides& dilations,
                                                       const CoordinateDiff& pads_begin,
                                                       const CoordinateDiff& pads_end)
            {
                // this implementation supports 1D, 2D and 3D convolutions
                NGRAPH_CHECK(in_shape.size() >= 3 && in_shape.size() <= 5,
                             "Unsupported input rank: ",
                             in_shape);

                NGRAPH_CHECK(in_shape.size() + 1 == f_shape.size(),
                             "Unsupported filter rank: ",
                             f_shape.size());

                NGRAPH_CHECK(in_shape.size() == out_shape.size(),
                             "Incompatible input and output ranks: ",
                             in_shape.size(),
                             " and ",
                             out_shape.size());

                const size_t groups = f_shape[filter_group_axis];
                const size_t in_channels = in_shape[in_channel_axis];
                NGRAPH_CHECK(in_channels % groups == 0,
                             "Input channels of data batch input must be multiple of groups");
                const Shape in_group_shape = [&]() {
                    Shape new_shape{in_shape};
                    new_shape[in_channel_axis] /= groups;
                    return new_shape;
                }();

                const size_t out_channels = out_shape[out_channel_axis];
                NGRAPH_CHECK(out_channels % groups == 0,
                             "Output channels of output must be multiple of groups");
                const Shape out_group_shape = [&]() {
                    Shape new_shape{out_shape};
                    new_shape[out_channel_axis] /= groups;
                    return new_shape;
                }();

                const Shape f_group_shape{std::next(f_shape.begin(), 1), std::end(f_shape)};
                validate_convolution_parameters(in_group_shape,
                                                f_group_shape,
                                                out_group_shape,
                                                strides,
                                                dilations,
                                                pads_begin,
                                                pads_end);
            }

            template <typename INPUT,
                      typename FILTER,
                      typename OUTPUT,
                      typename ACCU = typename widen<OUTPUT>::type>
            void group_convolution(const INPUT* in,
                                   const FILTER* f,
                                   OUTPUT* out,
                                   const Shape& in_shape,
                                   const Shape& filter_shape,
                                   const Shape& out_shape,
                                   const Strides& strides,
                                   const Strides& dilation,
                                   const CoordinateDiff& pads_begin,
                                   const CoordinateDiff& pads_end)

            {
                validate_group_convolution_parameters(
                    in_shape, filter_shape, out_shape, strides, dilation, pads_begin, pads_end);

                const size_t group_count = filter_shape[filter_group_axis];

                const INPUT* group_batch = in;
                const Shape group_batch_shape = [&]() {
                    Shape new_shape{in_shape};
                    new_shape[in_batch_axis] = 1;
                    new_shape[in_channel_axis] /= group_count;
                    return new_shape;
                }();
                const size_t group_batch_size = shape_size(group_batch_shape);

                const FILTER* group_filter = f;
                const Shape group_filter_shape = [&]() {
                    Shape new_shape{++filter_shape.begin(), filter_shape.end()};
                    return new_shape;
                }();
                const size_t group_filter_size = shape_size(group_filter_shape);

                OUTPUT* group_out = out;
                const Shape group_out_shape = [&]() {
                    Shape new_shape{out_shape};
                    new_shape[out_batch_axis] = 1;
                    new_shape[out_channel_axis] /= group_count;
                    return new_shape;
                }();
                const size_t group_out_size = shape_size(group_out_shape);

                for (size_t batch_idx = 0; batch_idx < in_shape[in_batch_axis]; ++batch_idx)
                {
                    group_filter = f;
                    for (size_t group_idx = 0; group_idx < group_count; ++group_idx)
                    {
                        runtime::reference::convolution(group_batch,
                                                        group_filter,
                                                        group_out,
                                                        group_batch_shape,
                                                        group_filter_shape,
                                                        group_out_shape,
                                                        strides,
                                                        dilation,
                                                        pads_begin,
                                                        pads_end);
                        group_batch += group_batch_size;
                        group_filter += group_filter_size;
                        group_out += group_out_size;
                    }
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
