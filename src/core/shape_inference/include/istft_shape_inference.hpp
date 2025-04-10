// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "openvino/op/istft.hpp"
#include "utils.hpp"

namespace ov::op::v16 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const ISTFT* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    using TDim = typename TRShape::value_type;
    using TDimVal = typename TDim::value_type;

    const auto inputs_count = input_shapes.size();
    const auto is_in_count_correct = inputs_count == 4 || inputs_count == 5;
    NODE_VALIDATION_CHECK(op, is_in_count_correct);

    const auto& data_shape = input_shapes[0];
    const auto& window_shape = input_shapes[1];
    const auto& frame_size_shape = input_shapes[2];
    const auto& frame_step_shape = input_shapes[3];

    const auto data_shape_rank = data_shape.rank();
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           data_shape_rank.compatible(3) || data_shape_rank.compatible(4),
                           "The shape of data must be 3D or 4D.");
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

    if (frame_size) {
        const auto& frame_size_val = (*frame_size)[0];
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               0 < frame_size_val,
                               "Provided frame size must be greater than zero, but got: ",
                               frame_size_val);
        const bool is_win_shape_correct =
            window_shape.is_dynamic() || (TDimVal{0} < window_shape[0].get_length() &&
                                          window_shape[0].get_length() <= static_cast<TDimVal>(frame_size_val));

        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               is_win_shape_correct,
                               "Window input dimension must be in range [1, ",
                               frame_size_val,
                               "].");

        if (data_shape.is_static()) {
            const auto& in_fft_dim = data_shape[data_shape.size() - 3];
            const auto expected_fft_dim = TDim(frame_size_val / 2 + 1);
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   in_fft_dim.compatible(expected_fft_dim),
                                   "The dimension at data_shape[-3] must be equal to: (frame_size // 2 + 1) ");
        }
    }

    if (frame_step) {
        const auto& frame_step_val = (*frame_step)[0];
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               0 < frame_step_val,
                               "Provided frame step must be greater than zero, but got: ",
                               frame_step_val);
    }

    // For the input with dynamic rank, output shape is also fully dynamic
    if (data_shape_rank.is_dynamic()) {
        return {data_shape};
    }
    const auto is_data_3D = data_shape.size() == 3;

    // Init output shape with dynamic dimension and update if more info can be inferred
    std::vector<TRShape> output_shapes{TRShape{TDim(ov::util::dim::inf_bound)}};
    if (inputs_count == 5) {
        const auto& length_shape = input_shapes[4];
        const bool has_len_valid_shape =
            length_shape.rank().is_dynamic() ||
            (length_shape.size() == 0 || (length_shape.size() == 1 && length_shape[0].compatible(1)));
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               has_len_valid_shape,
                               "The shape of 'signal_length' input must be a scalar or single element 1D tensor.");

        const auto sig_len_in = get_input_const_data_as_shape<TRShape>(op, 4, ta);
        if (sig_len_in) {  // Set desired length of the signal dimension, if provided
            output_shapes[0][0] = TDim{(*sig_len_in)[0]};
        }
    } else if (frame_size && frame_step) {  // Otherwise infer the length of the signal
        const auto& frame_size_val = (*frame_size)[0];
        const auto& frame_step_val = (*frame_step)[0];

        const int64_t frames_axis = 1 + (is_data_3D ? 0 : 1);
        const TDim& num_frames_dim = data_shape[frames_axis];
        TDim signal_length = (num_frames_dim - 1) * frame_step_val + frame_size_val;
        if (op->get_center()) {
            signal_length = signal_length - (frame_size_val & ~1);
        }
        output_shapes[0][0] = std::move(signal_length);
    }

    if (!is_data_3D) {  // Copy batch dimension
        const auto& batch_dim = data_shape[0];
        output_shapes[0].insert(output_shapes[0].begin(), batch_dim);
    }

    return output_shapes;
}
}  // namespace ov::op::v16
