// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/reference/group_convolution.hpp"

namespace ov {
namespace reference {

void infer_backward_conv_output_shape(const Shape& in_spatial_shape,
                                      const Shape& f_spatial_shape,
                                      Shape& out_spatial_shape,
                                      const Strides& strides,
                                      const Strides& dilations,
                                      const CoordinateDiff& pads_begin,
                                      const CoordinateDiff& pads_end);

void validate_convolution_backprop_data_parameters(const Shape& in_shape,
                                                   const Shape& f_shape,
                                                   const Shape& out_shape,
                                                   const Strides& strides,
                                                   const Strides& dilations,
                                                   const CoordinateDiff& pads_begin,
                                                   const CoordinateDiff& pads_end);

void validate_group_convolution_backprop_data_parameters(const Shape& in_shape,
                                                         const Shape& f_shape,
                                                         const Shape& out_shape,
                                                         const Strides& strides,
                                                         const Strides& dilations,
                                                         const CoordinateDiff& pads_begin,
                                                         const CoordinateDiff& pads_end);

template <typename T>
void group_convolution_backprop_data(const T* in,
                                     const T* f,
                                     T* out,
                                     const Shape& in_shape,
                                     const Shape& filter_shape,
                                     const Shape& out_shape,
                                     const Strides& strides,
                                     const Strides& dilation,
                                     const CoordinateDiff& pads_begin,
                                     const CoordinateDiff& pads_end,
                                     const CoordinateDiff& output_padding)

{
    const size_t group_count = filter_shape[filter_group_axis];

    const T* group_batch = in;
    const Shape group_batch_shape = [&]() {
        Shape new_shape{in_shape};
        new_shape[in_batch_axis] = 1;
        new_shape[in_channel_axis] /= group_count;
        return new_shape;
    }();
    const size_t group_batch_size = shape_size(group_batch_shape);

    const T* group_filter = f;
    const Shape group_filter_shape = [&]() {
        Shape new_shape{++filter_shape.begin(), filter_shape.end()};
        return new_shape;
    }();
    const size_t group_filter_size = shape_size(group_filter_shape);

    T* group_out = out;
    const Shape group_out_shape = [&]() {
        Shape new_shape{out_shape};
        new_shape[out_batch_axis] = 1;
        new_shape[out_channel_axis] /= group_count;
        return new_shape;
    }();
    const size_t group_out_size = shape_size(group_out_shape);

    Strides in_dilation(in_shape.size(), 1);
    for (size_t batch_idx = 0; batch_idx < in_shape[in_batch_axis]; ++batch_idx) {
        group_filter = f;
        for (size_t group_idx = 0; group_idx < group_count; ++group_idx) {
            reference::convolution_backprop_in(group_batch,
                                               group_filter,
                                               group_out,
                                               group_batch_shape,
                                               group_filter_shape,
                                               group_out_shape,
                                               in_dilation,
                                               dilation,
                                               pads_begin,
                                               pads_end,
                                               strides,
                                               output_padding);
            group_batch += group_batch_size;
            group_filter += group_filter_size;
            group_out += group_out_size;
        }
    }
}
}  // namespace reference
}  // namespace ov
