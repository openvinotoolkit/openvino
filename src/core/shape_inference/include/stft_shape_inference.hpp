// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "openvino/op/stft.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v15 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const STFT* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    using TDim = typename TRShape::value_type;
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4);

    const auto& signal_shape = input_shapes[0];
    const auto& window_shape = input_shapes[1];
    const auto& frame_size_shape = input_shapes[2];
    const auto& frame_step_shape = input_shapes[3];

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           signal_shape.rank().compatible(2),
                           "The shape of signal must be 2D [batch, signal_size].");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           window_shape.rank().compatible(1),
                           "The shape of window must be 1D [window_size].");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           frame_size_shape.rank().compatible(0),
                           "The shape of frame_size must be a scalar.");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           frame_step_shape.rank().compatible(0),
                           "The shape of frame_step must be a scalar.");

    const auto frame_size = get_input_const_data_as<TRShape, int64_t>(op, 2, ta);
    const auto frame_step = get_input_const_data_as<TRShape, int64_t>(op, 3, ta);

    if (!frame_size || !frame_step) {
        TRShape output_shape{signal_shape[0], -1, -1, 2};
        return {output_shape};
    }

    const auto& frame_size_val = (*frame_size)[0];
    const auto& frame_step_val = (*frame_step)[0];

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           0 < frame_size_val && (signal_shape.rank().is_dynamic() ||
                                                  frame_size_val < signal_shape[1].get_interval().get_max_val()),
                           "Provided frame size is ",
                           frame_size_val,
                           " but must be in range [1, ",
                           signal_shape[1],
                           "]");

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           window_shape.is_dynamic() ||
                               (0 < window_shape[0].get_length() && window_shape[0].get_length() <= frame_size_val),
                           "Window input dimension must be in range [1, ",
                           frame_size_val,
                           "]");

    const auto& batch_dim = signal_shape[0];
    const TDim frame_size_dim = TDim{frame_size_val};
    const TDim signal_frame_size_diff = signal_shape[1] - frame_size_dim;
    const TDim fft_samples_dim = (frame_size_val / 2) + 1;

    // Divsion opeartor for static Dimension of PartialShape can return non static dimension,
    // so get_length() is used to ensure static result for such case
    const TDim frames_dim = (signal_frame_size_diff.is_static() ? (signal_frame_size_diff.get_length() / frame_step_val)
                                                                : (signal_frame_size_diff / frame_step_val)) +
                            1;

    TRShape output_shape;
    if (op->get_transpose_frames()) {
        output_shape = TRShape{batch_dim, fft_samples_dim, frames_dim, 2};
    } else {
        output_shape = TRShape{batch_dim, frames_dim, fft_samples_dim, 2};
    }
    return {output_shape};
}
}  // namespace v15
}  // namespace op
}  // namespace ov
