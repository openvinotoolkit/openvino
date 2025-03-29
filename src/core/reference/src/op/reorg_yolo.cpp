// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/reorg_yolo.hpp"

#include <stdio.h>

#include <cmath>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
void reorg_yolo(const char* arg, char* out, const Shape& in_shape, int64_t stride, const size_t elem_size) {
    // [N, C, H, W]
    size_t in_N = in_shape[0];
    size_t in_C = in_shape[1];
    size_t in_H = in_shape[2];
    size_t in_W = in_shape[3];

    // Inference output shape logic:
    // in_shape [N,C,H,W] -> out_shape [N, C*(stride*stride), H/stride, W/stride]
    // ReorgYolo implementation calculates new indices like for backprop:
    // in_shape [N,C,H,W] -> out_shape [N, C/(stride*stride), H*stride, W*stride]

    size_t impl_out_C = in_C / (stride * stride);
    if (impl_out_C == 0) {
        OPENVINO_THROW("ReorgYolo. For [N, C, H, W] input shape, C >= (stride*stride) is "
                       "required.");
    }
    size_t impl_out_H = in_H * stride;
    size_t impl_out_W = in_W * stride;

    for (size_t n = 0; n < in_N; ++n) {
        for (size_t c = 0; c < in_C; ++c) {
            for (size_t h = 0; h < in_H; ++h) {
                for (size_t w = 0; w < in_W; ++w) {
                    size_t offset = c / impl_out_C;
                    size_t impl_c = c % impl_out_C;
                    size_t impl_h = h * stride + offset / stride;
                    size_t impl_w = w * stride + offset % stride;

                    size_t arg_index = ((n * impl_out_C + impl_c) * impl_out_H + impl_h) * impl_out_W + impl_w;
                    size_t dest_index = ((n * in_C + c) * in_H + h) * in_W + w;

                    arg_index *= elem_size;
                    dest_index *= elem_size;

                    std::copy(arg + arg_index, arg + (arg_index + elem_size), out + dest_index);
                }
            }
        }
    }
}
}  // namespace reference
}  // namespace ov
