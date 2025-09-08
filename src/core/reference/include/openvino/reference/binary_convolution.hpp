// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"
#include "openvino/reference/convolution.hpp"

namespace ov {
namespace reference {
namespace details {
inline uint8_t extract_bit(uint8_t val, uint8_t bit) {
    return (uint8_t)((val >> bit) & 0x01);
}

template <typename T>
inline bool xnor(T a, T b) {
    return a == b;
}

template <typename T_IN, typename T_F>
void binary_convolve_3D_channels(const ConvolutionParams& p,
                                 const T_IN* batch,
                                 const Shape& batch_shape,
                                 const T_F* filter,
                                 const Shape& filter_shape,
                                 T_IN*& out,
                                 const float pad_value) {
    const int n_bits = 8;
    const int64_t input_size_z = batch_shape[1];
    const int64_t input_size_y = batch_shape[2];
    const int64_t input_size_x = batch_shape[3];
    const int64_t filter_size_z = filter_shape[1];
    const int64_t filter_size_y = filter_shape[2];
    const int64_t filter_size_x = filter_shape[3];
    const int64_t dilated_filter_size_z = filter_size_z + (filter_size_z - 1) * (p.dilation[0] - 1);
    const int64_t dilated_filter_size_y = filter_size_y + (filter_size_y - 1) * (p.dilation[1] - 1);
    const int64_t dilated_filter_size_x = filter_size_x + (filter_size_x - 1) * (p.dilation[2] - 1);

    const Shape input_channel_shape(++batch_shape.begin(), batch_shape.end());
    const size_t input_channel_size = shape_size(input_channel_shape);
    const Shape filter_channel_shape(++filter_shape.begin(), filter_shape.end());
    const size_t filter_channel_size = shape_size(filter_channel_shape);
    const T_IN bit_count = static_cast<T_IN>(filter_channel_size);

    for (int64_t i_z = -p.pads_begin[0]; i_z <= (p.pads_end[0] + input_size_z - dilated_filter_size_z);
         i_z += p.strides[0]) {
        for (int64_t i_y = -p.pads_begin[1]; i_y <= (p.pads_end[1] + input_size_y - dilated_filter_size_y);
             i_y += p.strides[1]) {
            for (int64_t i_x = -p.pads_begin[2]; i_x <= (p.pads_end[2] + input_size_x - dilated_filter_size_x);
                 i_x += p.strides[2]) {
                auto input_channel = batch;
                size_t filter_channels_count = filter_shape[0];
                int filter_count = 0;
                T_IN sum = 0;
                while (filter_channels_count--) {
                    T_IN popcount = 0;
                    for (int64_t f_z = 0; f_z < filter_size_z; ++f_z) {
                        for (int64_t f_y = 0; f_y < filter_size_y; ++f_y) {
                            for (int64_t f_x = 0; f_x < filter_size_x; ++f_x) {
                                int64_t rel_i_z = i_z + (f_z * p.dilation[0]);
                                int64_t rel_i_y = i_y + (f_y * p.dilation[1]);
                                int64_t rel_i_x = i_x + (f_x * p.dilation[2]);

                                bool padding =
                                    !(in_range(rel_i_x, {0, input_size_x}) && in_range(rel_i_y, {0, input_size_y}) &&
                                      in_range(rel_i_z, {0, input_size_z}));
                                int64_t i_buf_idx =
                                    (rel_i_z * input_size_y * input_size_x) + (rel_i_y * input_size_x) + rel_i_x;

                                T_IN in_val = padding ? static_cast<T_IN>(pad_value)
                                                      : static_cast<T_IN>(input_channel[i_buf_idx]);

                                int f_buf_idx = static_cast<int>((f_z * filter_size_y * filter_size_x) +
                                                                 (f_y * filter_size_x) + f_x);

                                int f_byte_idx = (f_buf_idx + filter_count) / n_bits;
                                int bit_idx = (n_bits - 1) - ((f_buf_idx + filter_count) % n_bits);
                                uint8_t f_val = extract_bit(filter[f_byte_idx], bit_idx);

                                if (xnor(in_val, static_cast<T_IN>(f_val))) {
                                    popcount += static_cast<T_IN>(1);
                                }
                            }
                        }
                    }
                    input_channel += input_channel_size;
                    filter_count += static_cast<int>(filter_channel_size);
                    sum += (2 * popcount - bit_count);
                }
                *out = sum;
                ++out;
            }
        }
    }
}
}  // namespace details

template <typename T_IN, typename T_F>
void binary_convolution(const T_IN* in,
                        const T_F* f,
                        T_IN* out,
                        const Shape& in_shape,
                        const Shape& f_shape,
                        const Shape& out_shape,
                        const Strides& strides,
                        const Strides& dilations,
                        const CoordinateDiff& pads_begin,
                        const CoordinateDiff& pads_end,
                        const float pad_value) {
    validate_convolution_parameters(in_shape, f_shape, out_shape, strides, dilations, pads_begin, pads_end);

    // here we are converting all param types to int's to avoid arithmetic issues
    // (e.g signed + unsigned) in indexes calculation later
    ConvolutionParams params{strides, dilations, pads_begin, pads_end};

    // here we are extending spatial dimensions to 3D, because we are going to use 3D
    // convolution implementation to convolve also in 1D & 2D case
    Shape input_shape{in_shape};
    Shape filters_shape{f_shape};
    if (in_shape.size() < 5) {
        extend_to_3D(params, input_shape, filters_shape);
    }

    const size_t batches_count = input_shape[in_batch_axis];
    const Shape batch_shape(++input_shape.begin(), input_shape.end());
    const size_t batch_size = shape_size(batch_shape);

    const size_t filters_count = filters_shape[filter_out_ch_axis];
    const Shape filter_shape(++filters_shape.begin(), filters_shape.end());
    const size_t filter_size = shape_size(filter_shape);

    auto batch = in;
    for (size_t batch_idx = 0; batch_idx < batches_count; ++batch_idx) {
        auto filter = f;
        for (size_t f_idx = 0; f_idx < filters_count; ++f_idx) {
            details::binary_convolve_3D_channels(params, batch, batch_shape, filter, filter_shape, out, pad_value);
            filter += filter_size;
        }
        batch += batch_size;
    }
}
}  // namespace reference
}  // namespace ov
