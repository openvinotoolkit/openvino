// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/stft.hpp"

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/op_types.hpp"
#include "utils/common.hpp"
#include "utils/dft.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov::frontend::onnx::ai_onnx::opset_17 {

namespace {
ov::Output<ov::Node> to_i64_scalar(const ov::Output<ov::Node>& value) {
    const auto scalar = std::make_shared<v1::Reshape>(value, v0::Constant::create(ov::element::i64, {0}, {}), false);
    return std::make_shared<v0::Convert>(scalar, ov::element::i64);
}

// ONNX STFT inputs: signal, frame_step, window (optional), frame_length (optional)
// OpenVINO v15::STFT inputs: signal, window, frame_size, frame_step
// Both produce [batch_size, num_frames, dft_bins, 2] (transpose_frames=false)
ov::Output<ov::Node> make_stft_real(const ov::Output<ov::Node>& signal,
                                    const ov::OutputVector& inputs,
                                    bool window_provided,
                                    bool frame_length_provided,
                                    bool onesided) {
    ov::Output<ov::Node> data = signal;
    if (signal.get_partial_shape().size() == 3) {
        data = std::make_shared<v0::Squeeze>(signal, v0::Constant::create(ov::element::i64, {1}, {2}));
    }
    const auto frame_step = to_i64_scalar(inputs[1]);

    // Default frame_length is the window length, or the signal length if window is not provided
    ov::Output<ov::Node> frame_length;
    if (frame_length_provided) {
        frame_length = to_i64_scalar(inputs[3]);
    } else if (window_provided) {
        frame_length = to_i64_scalar(std::make_shared<v3::ShapeOf>(inputs[2], ov::element::i64));
    } else {
        frame_length = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(data, ov::element::i64),
                                                    v0::Constant::create(ov::element::i64, {}, {1}),
                                                    v0::Constant::create(ov::element::i64, {}, {0}));
    }

    ov::Output<ov::Node> window;
    if (window_provided) {
        window = inputs[2];
    } else {
        window = std::make_shared<v3::Broadcast>(
            v0::Constant::create(signal.get_element_type(), {}, {1}),
            std::make_shared<v0::Unsqueeze>(frame_length, v0::Constant::create(ov::element::i64, {}, {0})));
    }

    const ov::Output<ov::Node> stft = std::make_shared<v15::STFT>(data, window, frame_length, frame_step, false);
    if (onesided) {
        return stft;
    }
    // Restore two-sided spectrum using Hermitian symmetry: X[k] = conj(X[N - k]), k = N/2+1 .. N-1
    const auto bins_axis = v0::Constant::create(ov::element::i64, {}, {2});
    const auto onesided_bins = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(stft, ov::element::i64),
                                                            bins_axis,
                                                            v0::Constant::create(ov::element::i64, {}, {0}));
    const auto mirror_indices = std::make_shared<v4::Range>(std::make_shared<v1::Subtract>(frame_length, onesided_bins),
                                                            v0::Constant::create(ov::element::i64, {}, {0}),
                                                            v0::Constant::create(ov::element::i64, {}, {-1}),
                                                            ov::element::i64);
    const auto mirrored = std::make_shared<v8::Gather>(stft, mirror_indices, bins_axis);
    const auto conj = std::make_shared<v1::Multiply>(
        mirrored,
        v0::Constant::create(signal.get_element_type(), {2}, std::vector<float>{1.f, -1.f}));
    return std::make_shared<v0::Concat>(ov::OutputVector{stft, conj}, 2);
}
}  // namespace

ov::OutputVector stft(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 2, 4);

    const ov::OutputVector ov_inputs{node.get_ov_inputs()};
    auto signal = ov_inputs.at(0);
    const auto signal_param_shape = signal.get_partial_shape();
    const auto window_node_provided = ov_inputs.size() > 2 && !ov::op::util::is_null(ov_inputs[2]);
    const auto dft_length_provided = ov_inputs.size() > 3 && !ov::op::util::is_null(ov_inputs[3]);
    const auto onesided = node.get_attribute_value<int64_t>("onesided", 1);
    const auto is_int_or_dynamic = [](const ov::Output<ov::Node>& value) {
        return value.get_element_type().is_dynamic() || value.get_element_type().is_integral_number();
    };
    CHECK_VALID_NODE(node, is_int_or_dynamic(ov_inputs[1]), "frame_step input must be of integer type.");
    CHECK_VALID_NODE(node,
                     !dft_length_provided || is_int_or_dynamic(ov_inputs[3]),
                     "frame_length input must be of integer type.");

    // Real signal [batch, signal_length] or [batch, signal_length, 1] is mapped to ov::op::v15::STFT.
    // 2D input is not described in onnx spec, but it is allowed by onnx model checker and seen in real models.
    const auto signal_rank = signal_param_shape.rank();
    const bool is_real_signal = signal_rank.is_static() &&
                                (signal_rank.get_length() == 2 ||
                                 (signal_rank.get_length() == 3 &&
                                  (signal_param_shape[2] == 1 || (onesided && signal_param_shape[2].compatible(1)))));
    if (is_real_signal) {
        const auto is_scalar_like = [](const ov::Output<ov::Node>& value) {
            const auto& shape = value.get_partial_shape();
            return shape.rank().is_dynamic() || shape.size() == 0 || (shape.size() == 1 && shape[0].compatible(1));
        };
        CHECK_VALID_NODE(node, is_scalar_like(ov_inputs[1]), "frame_step input must be a scalar or Shape{1}.");
        CHECK_VALID_NODE(node,
                         !dft_length_provided || is_scalar_like(ov_inputs[3]),
                         "frame_length input must be a scalar or Shape{1}.");
        if (window_node_provided) {
            CHECK_VALID_NODE(node,
                             ov_inputs[2].get_partial_shape().rank().compatible(1),
                             "The rank of window input must be 1D.");
        }
        return {make_stft_real(signal, ov_inputs, window_node_provided, dft_length_provided, onesided == 1)};
    }
    // Use DFT-based decomposition for complex signal
    CHECK_VALID_NODE(node,
                     signal_param_shape.is_static() && signal_param_shape.size() == 3,
                     "Shape of signal input must be static with the rank equal to 3.");
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

    // Default frame_length is the window length, or the signal length if window is not provided
    int64_t frame_length = signal_param_shape[axis].get_length();
    if (dft_length_provided) {
        const auto& frame_length_node = ov_inputs[3];
        CHECK_VALID_NODE(node,
                         ov::op::util::is_constant(frame_length_node.get_node_shared_ptr()) &&
                             ov::shape_size(frame_length_node.get_shape()) <= 1,
                         "frame_length input must be a scalar or Shape{1} constant.");
        frame_length =
            ov::as_type_ptr<v0::Constant>(frame_length_node.get_node_shared_ptr())->cast_vector<int64_t>()[0];
    } else if (window_node_provided) {
        CHECK_VALID_NODE(node,
                         ov_inputs[2].get_partial_shape().is_static() && ov_inputs[2].get_partial_shape().size() == 1,
                         "Window input must be 1D with static shape if frame_length is not provided.");
        frame_length = static_cast<int64_t>(ov_inputs[2].get_shape()[0]);
    }

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
    const auto is_complex = [](const ov::Output<ov::Node>& data) {
        return data.get_partial_shape().rank().is_static() && (data.get_partial_shape().cend() - 1)->is_static() &&
               (data.get_partial_shape().cend() - 1)->get_length() == 2;
    };
    if (onesided == 1) {
        CHECK_VALID_NODE(node, !is_complex(signal), "If attribute onesided==1, signal input can NOT be complex.");
    }
    const int64_t batch_size = signal_param_shape[0].get_length();
    const auto nstfts = static_cast<int64_t>((signal_param_shape[axis].get_length() - frame_length) / frame_step) + 1;
    const auto axis_const = v0::Constant::create(ov::element::i64, {}, {axis});
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
                is_complex(slice) ? v0::Constant::create(ov::element::i64, {2}, {-1, 2})
                                  : (onesided ? v0::Constant::create(ov::element::i64, {1}, {-1})
                                              : v0::Constant::create(ov::element::i64, {2}, {-1, 1})),
                false);
            const auto dft = dft::make_dft(
                window_node_provided
                    ? std::make_shared<v1::Multiply>(
                          flatten_slice,
                          is_complex(flatten_slice)
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

ONNX_OP("STFT", OPSET_SINCE(1), ai_onnx::opset_17::stft);
}  // namespace ov::frontend::onnx::ai_onnx::opset_17
