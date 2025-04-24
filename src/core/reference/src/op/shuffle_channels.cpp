// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/shuffle_channels.hpp"

#include "openvino/reference/reshape.hpp"

namespace ov {
namespace reference {
void shuffle_channels(const char* arg,
                      char* out,
                      const Shape& data_shape,
                      size_t elem_size,
                      const int64_t axis,
                      const int64_t group) {
    // Input ND tensor of data_shape (ds) is always considered as 4D tensor with the
    // following shape:
    // dim 0: ds[0] * ds[1] * ... * ds[axis-1] (or 1 if axis == 0)
    // dim 1: group
    // dim 2: ds[axis] / group
    // dim 3: ds[axis+1] * ds[axis+2] * ... * ds[ds.size()-1]
    // (or 1 if axis points to last dimension)

    // The representation of ND tensor as 4D tensor doesn't affect flat data order
    Shape reshaped_input_shape(4, 1);
    const size_t axis_zb = axis >= 0 ? axis : axis + data_shape.size();  // Allow negative indices
    for (size_t i = 0; i < axis_zb; ++i) {
        // All dimensions before input channels dim axis
        reshaped_input_shape[0] *= data_shape[i];
    }
    reshaped_input_shape[1] = group;
    reshaped_input_shape[2] = data_shape[axis_zb] / group;
    for (size_t i = axis_zb + 1; i < data_shape.size(); ++i) {
        // All dimensions after input channels dim axis
        reshaped_input_shape[3] *= data_shape[i];
    }

    // The two dimensions in the middle are swapped
    const Shape transposed_shape{reshaped_input_shape[0],
                                 reshaped_input_shape[2],
                                 reshaped_input_shape[1],
                                 reshaped_input_shape[3]};
    AxisVector axis_vector{0, 2, 1, 3};
    reshape(arg, out, reshaped_input_shape, axis_vector, transposed_shape, elem_size);

    // Reshaped 4D tensor is interpreted as ND output tensor with original shape of data
    // input
}
}  // namespace reference
}  // namespace ov
