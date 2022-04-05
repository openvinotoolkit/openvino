// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/runtime/reference/convolution_backprop_data.hpp"
#include "ngraph/util.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
void infer_backward_conv_output_shape(const Shape& in_spatial_shape,
                                      const Shape& f_spatial_shape,
                                      Shape& out_spatial_shape,
                                      const Strides& strides,
                                      const Strides& dilations,
                                      const CoordinateDiff& pads_begin,
                                      const CoordinateDiff& pads_end) {
    for (size_t idx = 0; idx < in_spatial_shape.size(); idx++) {
        size_t in_padded_dim = (in_spatial_shape[idx] - 1) * strides[idx] - pads_begin[idx] - pads_end[idx];
        size_t filter_dilated_dim = dilations[idx] * (f_spatial_shape[idx] - 1) + 1;
        size_t out_spatial_dim = in_padded_dim + filter_dilated_dim;
        out_spatial_shape.push_back(out_spatial_dim);
    }
}

void validate_convolution_backprop_data_parameters(const Shape& in_shape,
                                                   const Shape& f_shape,
                                                   const Shape& out_shape,
                                                   const Strides& strides,
                                                   const Strides& dilations,
                                                   const CoordinateDiff& pads_begin,
                                                   const CoordinateDiff& pads_end) {
    // this implementation supports 1D, 2D and 3D convolutions
    NGRAPH_CHECK(in_shape.size() >= 3 && in_shape.size() <= 5, "Unsupported input rank: ", in_shape);
    NGRAPH_CHECK(in_shape.size() == f_shape.size(),
                 "Incompatible input ranks: ",
                 in_shape.size(),
                 " and ",
                 f_shape.size());
    NGRAPH_CHECK(in_shape[in_channel_axis] == f_shape[filter_in_ch_axis],
                 "Incompatible input channels in data batch and filters shapes: ",
                 in_shape[in_channel_axis],
                 " and ",
                 f_shape[filter_in_ch_axis]);
    NGRAPH_CHECK(in_shape.size() == out_shape.size(),
                 "Incompatible input and output ranks: ",
                 in_shape.size(),
                 " and ",
                 out_shape.size());
    const auto spatial_dims = in_shape.size() - 2;
    NGRAPH_CHECK(strides.size() == spatial_dims, "Strides not definied for all and only spatial dimensions");
    NGRAPH_CHECK(dilations.size() == spatial_dims, "Dilations not defined for all and only spatial dimensions");
    NGRAPH_CHECK((pads_begin.size() == pads_end.size()) && (pads_begin.size() == spatial_dims),
                 "Pads not defined for all and only spatial dimensions");

    Shape out_spatial_shape{std::next(out_shape.begin(), 2), std::end(out_shape)};
    Shape infered_out_spatial_shape{};
    infer_backward_conv_output_shape(Shape{std::next(in_shape.begin(), 2), std::end(in_shape)},
                                     Shape{std::next(f_shape.begin(), 2), std::end(f_shape)},
                                     infered_out_spatial_shape,
                                     strides,
                                     dilations,
                                     pads_begin,
                                     pads_end);
    NGRAPH_CHECK(out_spatial_shape == infered_out_spatial_shape, "Incorrect output shape provided");
}

void validate_group_convolution_backprop_data_parameters(const Shape& in_shape,
                                                         const Shape& f_shape,
                                                         const Shape& out_shape,
                                                         const Strides& strides,
                                                         const Strides& dilations,
                                                         const CoordinateDiff& pads_begin,
                                                         const CoordinateDiff& pads_end) {
    // this implementation supports 1D, 2D and 3D convolutions
    NGRAPH_CHECK(in_shape.size() >= 3 && in_shape.size() <= 5, "Unsupported input rank: ", in_shape);

    NGRAPH_CHECK(in_shape.size() + 1 == f_shape.size(), "Unsupported filter rank: ", f_shape.size());

    NGRAPH_CHECK(in_shape.size() == out_shape.size(),
                 "Incompatible input and output ranks: ",
                 in_shape.size(),
                 " and ",
                 out_shape.size());

    const size_t groups = f_shape[filter_group_axis];
    const size_t in_channels = in_shape[in_channel_axis];
    NGRAPH_CHECK(in_channels % groups == 0, "Input channels of data batch input must be multiple of groups");
    const Shape in_group_shape = [&]() {
        Shape new_shape{in_shape};
        new_shape[in_channel_axis] /= groups;
        return new_shape;
    }();

    const size_t out_channels = out_shape[out_channel_axis];
    NGRAPH_CHECK(out_channels % groups == 0, "Output channels of output must be multiple of groups");
    const Shape out_group_shape = [&]() {
        Shape new_shape{out_shape};
        new_shape[out_channel_axis] /= groups;
        return new_shape;
    }();

    const Shape f_group_shape{std::next(f_shape.begin(), 1), std::end(f_shape)};
    validate_convolution_backprop_data_parameters(in_group_shape,
                                                  f_group_shape,
                                                  out_group_shape,
                                                  strides,
                                                  dilations,
                                                  pads_begin,
                                                  pads_end);
}

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
            runtime::reference::convolution_backprop_in(group_batch,
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

// DEPRECATED, can't be removed currently due to arm-plugin dependency
template <typename OUTPUT, typename FILTER, typename INPUT, typename ACCUMULATION = typename widen<INPUT>::type>
NGRAPH_DEPRECATED("group_convolution_backprop_data function without output_paddings is deprecated, "
                  "use the one with output_padding.")
void group_convolution_backprop_data(const INPUT* in,
                                     const FILTER* f,
                                     OUTPUT* out,
                                     const Shape& in_shape,
                                     const Shape& filter_shape,
                                     const Shape& out_shape,
                                     const Strides& strides,
                                     const Strides& dilation,
                                     const CoordinateDiff& pads_begin,
                                     const CoordinateDiff& pads_end) {
    const ngraph::CoordinateDiff output_padding(in_shape.size() - 2, 0);

    group_convolution_backprop_data(in,
                                    f,
                                    out,
                                    in_shape,
                                    filter_shape,
                                    out_shape,
                                    strides,
                                    dilation,
                                    pads_begin,
                                    pads_end,
                                    output_padding);
}
}  // namespace reference

}  // namespace runtime

}  // namespace ngraph