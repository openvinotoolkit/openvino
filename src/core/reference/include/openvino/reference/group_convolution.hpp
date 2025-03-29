// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/reference/convolution.hpp"
#include "openvino/reference/helpers.hpp"

namespace {
constexpr size_t filter_group_axis = 0;
constexpr size_t filter_in_ch_axis = 2;
constexpr size_t in_batch_axis = 0;
constexpr size_t in_channel_axis = 1;
constexpr size_t out_batch_axis = 0;
constexpr size_t out_channel_axis = 1;
}  // namespace

namespace ov {
namespace reference {
void validate_group_convolution_parameters(const Shape& in_shape,
                                           const Shape& f_shape,
                                           const Shape& out_shape,
                                           const Strides& strides,
                                           const Strides& dilations,
                                           const CoordinateDiff& pads_begin,
                                           const CoordinateDiff& pads_end);

template <typename INPUT, typename FILTER, typename OUTPUT, typename ACCU = typename widen<OUTPUT>::type>
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
    validate_group_convolution_parameters(in_shape, filter_shape, out_shape, strides, dilation, pads_begin, pads_end);

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

    for (size_t batch_idx = 0; batch_idx < in_shape[in_batch_axis]; ++batch_idx) {
        group_filter = f;
        for (size_t group_idx = 0; group_idx < group_count; ++group_idx) {
            reference::convolution(group_batch,
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
}  // namespace reference
}  // namespace ov
