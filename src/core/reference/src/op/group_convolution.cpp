// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/group_convolution.hpp"

namespace ov {
namespace reference {

void validate_group_convolution_parameters(const Shape& in_shape,
                                           const Shape& f_shape,
                                           const Shape& out_shape,
                                           const Strides& strides,
                                           const Strides& dilations,
                                           const CoordinateDiff& pads_begin,
                                           const CoordinateDiff& pads_end) {
    // this implementation supports 1D, 2D and 3D convolutions
    OPENVINO_ASSERT(in_shape.size() >= 3 && in_shape.size() <= 5, "Unsupported input rank: ", in_shape);

    OPENVINO_ASSERT(in_shape.size() + 1 == f_shape.size(), "Unsupported filter rank: ", f_shape.size());

    OPENVINO_ASSERT(in_shape.size() == out_shape.size(),
                    "Incompatible input and output ranks: ",
                    in_shape.size(),
                    " and ",
                    out_shape.size());

    const size_t groups = f_shape[filter_group_axis];
    const size_t in_channels = in_shape[in_channel_axis];
    OPENVINO_ASSERT(in_channels % groups == 0, "Input channels of data batch input must be multiple of groups");
    const Shape in_group_shape = [&]() {
        Shape new_shape{in_shape};
        new_shape[in_channel_axis] /= groups;
        return new_shape;
    }();

    const size_t out_channels = out_shape[out_channel_axis];
    OPENVINO_ASSERT(out_channels % groups == 0, "Output channels of output must be multiple of groups");
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

}  // namespace reference
}  // namespace ov
