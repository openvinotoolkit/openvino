// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/stft.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/op_types.hpp"
#include "utils/common.hpp"
#include "utils/dft.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_17 {

namespace {
// Check if input is complex (last dimension == 2)
bool is_complex_input(const ov::Output<ov::Node>& data) {
    return data.get_partial_shape().rank().is_static() && (data.get_partial_shape().cend() - 1)->is_static() &&
           (data.get_partial_shape().cend() - 1)->get_length() == 2;
}

// Implementation using v15::STFT for onesided=1 with real input
ov::OutputVector stft_v15(const ov::frontend::onnx::Node& node, const ov::OutputVector& ov_inputs) {
    // ONNX STFT inputs: signal, frame_step, window (optional), frame_length (optional)
    // ONNX STFT output shape: [batch_size, num_frames, fft_length, 2]
    // OpenVINO v15::STFT inputs: data, window, frame_size, frame_step
    // OpenVINO v15::STFT output shape: [batch, num_frames, fft_length, 2] (transpose_frames=false)
    auto signal = ov_inputs.at(0);
    const auto frame_step_input = ov_inputs.at(1);

    // ONNX signal shape is [batch, signal_length, 1] for real input
    // OpenVINO STFT expects [batch, signal_length]
    // Squeeze the last dimension from signal
    const auto squeeze_axis = v0::Constant::create(ov::element::i64, {1}, {-1});
    auto signal_squeezed = std::make_shared<v0::Squeeze>(signal, squeeze_axis);

    // Get frame_length (frame_size for OpenVINO STFT)
    ov::Output<ov::Node> frame_size;
    const auto frame_length_provided = ov_inputs.size() > 3 && !ov::op::util::is_null(ov_inputs[3]);
    if (frame_length_provided) {
        frame_size = ov_inputs[3];
    } else {
        // Default frame_length: If not provided, use the window length
        // Per ONNX spec: frame_length defaults to the size of the window
        const auto window_provided = ov_inputs.size() > 2 && !ov::op::util::is_null(ov_inputs[2]);
        CHECK_VALID_NODE(node,
                         window_provided,
                         "frame_length must be provided when window is not specified.");
        // Get window length using ShapeOf
        const auto window_shape = std::make_shared<v3::ShapeOf>(ov_inputs[2], ov::element::i64);
        const auto zero_idx = v0::Constant::create(ov::element::i64, {1}, {0});
        frame_size = std::make_shared<v8::Gather>(window_shape, zero_idx, zero_idx);
    }

    // Get or create window
    ov::Output<ov::Node> window;
    const auto window_provided = ov_inputs.size() > 2 && !ov::op::util::is_null(ov_inputs[2]);
    if (window_provided) {
        window = ov_inputs[2];
        // Validate window rank
        if (window.get_partial_shape().rank().is_static()) {
            CHECK_VALID_NODE(node,
                             window.get_partial_shape().rank().get_length() == 1,
                             "The rank of window input must be 1D.");
        }
    } else {
        // Create a window of ones with frame_size length
        const auto one = v0::Constant::create(ov::element::f32, {}, {1.0f});
        const auto one_like = std::make_shared<v1::ConvertLike>(one, signal);
        const auto frame_size_1d = std::make_shared<v1::Reshape>(
            frame_size,
            v0::Constant::create(ov::element::i64, {1}, {1}),
            false);
        window = std::make_shared<v3::Broadcast>(one_like, frame_size_1d);
    }

    // Create STFT operation
    // transpose_frames=false gives output shape [batch, num_frames, fft_length, 2]
    // which matches ONNX expected output shape
    constexpr bool transpose_frames = false;
    auto stft_result =
        std::make_shared<v15::STFT>(signal_squeezed, window, frame_size, frame_step_input, transpose_frames);

    return {stft_result};
}

// Legacy DFT-based implementation for onesided=0 or complex input
ov::OutputVector stft_dft_fallback(const ov::frontend::onnx::Node& node,
                                   const ov::OutputVector& ov_inputs,
                                   int64_t onesided) {
    auto signal = ov_inputs.at(0);
    const auto dft_length_provided = ov_inputs.size() > 3 && !ov::op::util::is_null(ov_inputs[3]);
    const int64_t axis = 1;

    const auto& frame_step_node = ov_inputs.at(1);
    CHECK_VALID_NODE(node,
                     ov::op::util::is_constant(frame_step_node.get_node_shared_ptr()) &&
                         ov::shape_size(frame_step_node.get_shape()) <= 1,
                     "frame_step input must be a scalar or Shape{1} constant.");
    const auto frame_step =
        ov::as_type_ptr<v0::Constant>(frame_step_node.get_node_shared_ptr())->cast_vector<int64_t>()[0];
    CHECK_VALID_NODE(node,
                     frame_step > 0,
                     "Provided frame_step input value must be greater than zero. Got: ",
                     frame_step);
    const auto signal_param_shape = signal.get_partial_shape();
    CHECK_VALID_NODE(node,
                     signal_param_shape.is_static() && signal_param_shape.size() == 3,
                     "Shape of signal input must be static with the rank equal to 3.");

    int64_t frame_length = signal_param_shape[axis].get_length() / frame_step;  // default value
    if (dft_length_provided) {
        const auto& frame_length_node = ov_inputs[3];
        CHECK_VALID_NODE(node,
                         ov::op::util::is_constant(frame_length_node.get_node_shared_ptr()) &&
                             ov::shape_size(frame_length_node.get_shape()) <= 1,
                         "frame_length input must be a scalar or Shape{1} constant.");
        frame_length =
            ov::as_type_ptr<v0::Constant>(frame_length_node.get_node_shared_ptr())->cast_vector<int64_t>()[0];
    }

    const auto window_node_provided = ov_inputs.size() > 2 && !ov::op::util::is_null(ov_inputs[2]);
    if (window_node_provided) {  // window input provided
        if (ov_inputs[2].get_partial_shape().rank().is_static()) {
            CHECK_VALID_NODE(node,
                             ov_inputs[2].get_partial_shape().rank().get_length() == 1,
                             "The rank of window input must be 1D.");
            if (ov_inputs[2].get_partial_shape()[0].is_static()) {
                CHECK_VALID_NODE(node,
                                 ov_inputs[2].get_partial_shape()[0].get_length() == frame_length,
                                 "The length of window input must be equal to frame_length.");
            }
        }
    }

    if (onesided == 1) {
        CHECK_VALID_NODE(node,
                         !is_complex_input(signal),
                         "If attribute onesided==1, signal input can NOT be complex.");
    }
    const int64_t batch_size = signal_param_shape[0].get_length();
    const auto nstfts = static_cast<int64_t>((signal_param_shape[axis].get_length() - frame_length) / frame_step) + 1;
    const auto zero_const = v0::Constant::create(ov::element::i64, {}, {0});
    const auto step = v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1});
    ov::OutputVector all_signals;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        ov::OutputVector signals_in_batch;
        for (int64_t sig_idx = 0; sig_idx < nstfts; ++sig_idx) {
            const auto start =
                v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{batch, sig_idx * frame_step});
            const auto stop =
                v0::Constant::create(ov::element::i64,
                                     ov::Shape{2},
                                     std::vector<int64_t>{batch + 1, sig_idx * frame_step + frame_length});
            const auto slice_axes = v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, axis});
            const auto slice = std::make_shared<v8::Slice>(signal, start, stop, step, slice_axes);
            const ov::Output<ov::Node> flatten_slice = std::make_shared<v1::Reshape>(
                slice,
                is_complex_input(slice) ? v0::Constant::create(ov::element::i64, {2}, {-1, 2})
                                        : (onesided ? v0::Constant::create(ov::element::i64, {1}, {-1})
                                                    : v0::Constant::create(ov::element::i64, {2}, {-1, 1})),
                false);
            const auto dft = dft::make_dft(
                window_node_provided
                    ? std::make_shared<v1::Multiply>(
                          flatten_slice,
                          is_complex_input(flatten_slice)
                              ? std::make_shared<v3::Broadcast>(  // align window shape with signal shape
                                    std::make_shared<v0::Unsqueeze>(ov_inputs[2],
                                                                    v0::Constant::create(ov::element::i64, {1}, {1})),
                                    std::make_shared<v3::ShapeOf>(flatten_slice))
                              : ov_inputs[2])
                    : flatten_slice,
                dft_length_provided ? ov_inputs[3] : std::make_shared<NullNode>(),
                0,
                false,
                onesided == 1);
            signals_in_batch.push_back(std::make_shared<v0::Unsqueeze>(dft, zero_const));
        }
        all_signals.push_back(
            std::make_shared<v0::Unsqueeze>(std::make_shared<v0::Concat>(signals_in_batch, 0), zero_const));
    }
    return {std::make_shared<v0::Concat>(all_signals, 0)};
}
}  // namespace

ov::OutputVector stft(const ov::frontend::onnx::Node& node) {
    const ov::OutputVector ov_inputs{node.get_ov_inputs()};
    auto signal = ov_inputs.at(0);
    const auto onesided = node.get_attribute_value<int64_t>("onesided", 1);

    // Use v15::STFT for onesided=1 with real input (optimized path)
    // Fall back to DFT-based implementation for onesided=0 or complex input
    if (onesided == 1 && !is_complex_input(signal)) {
        return stft_v15(node, ov_inputs);
    } else {
        return stft_dft_fallback(node, ov_inputs, onesided);
    }
}

ONNX_OP("STFT", OPSET_SINCE(1), ai_onnx::opset_17::stft);
}  // namespace opset_17
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
