// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "openvino/op/istft.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v16 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const ISTFT* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    using TDim = typename TRShape::value_type;
    using TDimVal = typename TDim::value_type;

    NODE_VALIDATION_CHECK(op, input_shapes.size() == 5);

    const auto& data_shape = input_shapes[0];
    const auto& window_shape = input_shapes[1];
    const auto& frame_size_shape = input_shapes[2];
    const auto& frame_step_shape = input_shapes[3];
    const auto& length_shape = input_shapes[4];

    const auto data_shape_rank = data_shape.rank();
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           data_shape_rank.compatible(3) || data_shape_rank.compatible(4),
                           "The shape of data must be 3D [signal_size] or 4D [batch, signal_size].");
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
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           length_shape.rank().compatible(0),
                           "The shape of length input must be a scalar.");

    const auto frame_size = get_input_const_data_as<TRShape, int64_t>(op, 2, ta);
    const auto frame_step = get_input_const_data_as<TRShape, int64_t>(op, 3, ta);
    const auto sig_len_in = get_input_const_data_as_shape<TRShape>(op, 4, ta);

    if (frame_size) {
        const auto& frame_size_val = (*frame_size)[0];
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               0 < frame_size_val,
                               "Provided frame size is ",
                               frame_size_val,
                               " but must be greater than zero.");
        const bool is_win_shape_correct =
            window_shape.is_dynamic() || (TDimVal{0} < window_shape[0].get_length() &&
                                          window_shape[0].get_length() <= static_cast<TDimVal>(frame_size_val));

        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               is_win_shape_correct,
                               "Window input dimension must be in range [1, ",
                               frame_size_val,
                               "].");
    }

    if (frame_step) {
        const auto& frame_step_val = (*frame_step)[0];
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               0 < frame_step_val,
                               "Provided frame step is ",
                               frame_step_val,
                               " but must be greater than zero.");
    }

    // For the input with dynamic rank, output shape is also fully dynamic
    if (data_shape_rank.is_dynamic()) {
        return {data_shape};
    }
    const auto is_data_3D = data_shape.size() == 3;

    std::vector<TRShape> output_shapes;
    if (sig_len_in && (*sig_len_in)[0].is_static()) {  // Set desired length of the signal dimension, if provided
        output_shapes.emplace_back(TRShape{(*sig_len_in)[0]});
    } else if (frame_size && frame_step) {  // Otherwise infer the length of the signal
        const auto& frame_size_val = (*frame_size)[0];
        const auto& frame_step_val = (*frame_step)[0];

        const int64_t frames_axis = 1 + (is_data_3D ? 0 : 1);
        const TDim& num_frames_dim = data_shape[frames_axis];
        TDim signal_length = (num_frames_dim - 1) * frame_step_val;
        if (!op->get_center()) {
            signal_length += frame_size_val;
        }
        output_shapes.emplace_back(TRShape{std::move(signal_length)});
    } else {  // Not enough info to infer the signal lenght, set dynamic dimension
        output_shapes.emplace_back(TRShape{TDim(ov::util::dim::inf_bound)});
    }

    if (!is_data_3D) {  // Copy batch dimension
        const auto& batch_dim = data_shape[0];
        output_shapes[0].insert(output_shapes[0].begin(), batch_dim);
    }

    return output_shapes;
}
}  // namespace v16
}  // namespace op
}  // namespace ov
