// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/stft.hpp"

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/op_types.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::ai_onnx::opset_17 {

namespace {
std::shared_ptr<v0::Constant> i64_const(const ov::Shape& shape, const std::vector<int64_t>& values) {
    return v0::Constant::create(ov::element::i64, shape, values);
}

ov::Output<ov::Node> to_i64_scalar(const ov::Output<ov::Node>& value) {
    const auto scalar = std::make_shared<v1::Reshape>(value, i64_const({0}, {}), false);
    return std::make_shared<v0::Convert>(scalar, ov::element::i64);
}

ov::Output<ov::Node> get_dim(const ov::Output<ov::Node>& value, int64_t axis) {
    return std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(value, ov::element::i64),
                                        i64_const({}, {axis}),
                                        i64_const({}, {0}));
}

// STFT of real signal [batch, signal_length] -> [batch, num_frames, dft_bins, 2].
// ONNX STFT inputs: signal, frame_step, window (optional), frame_length (optional)
// OpenVINO v15::STFT inputs: signal, window, frame_size, frame_step
ov::Output<ov::Node> make_real_stft(const ov::Output<ov::Node>& data,
                                    const ov::OutputVector& inputs,
                                    bool window_provided,
                                    bool frame_length_provided,
                                    bool onesided) {
    const auto frame_step = to_i64_scalar(inputs[1]);

    // Default frame_length is the window length, or the signal length if window is not provided
    ov::Output<ov::Node> frame_length;
    if (frame_length_provided) {
        frame_length = to_i64_scalar(inputs[3]);
    } else if (window_provided) {
        frame_length = to_i64_scalar(std::make_shared<v3::ShapeOf>(inputs[2], ov::element::i64));
    } else {
        frame_length = get_dim(data, 1);
    }

    ov::Output<ov::Node> window;
    if (window_provided) {
        window = inputs[2];
    } else {
        window = std::make_shared<v3::Broadcast>(v0::Constant::create(data.get_element_type(), {}, {1}),
                                                 std::make_shared<v0::Unsqueeze>(frame_length, i64_const({}, {0})));
    }

    const ov::Output<ov::Node> stft = std::make_shared<v15::STFT>(data, window, frame_length, frame_step, false);
    if (onesided) {
        return stft;
    }
    // Restore two-sided spectrum using Hermitian symmetry: X[k] = conj(X[N - k]), k = N/2+1 .. N-1
    const auto bins_axis = i64_const({}, {2});
    const auto mirror_indices =
        std::make_shared<v4::Range>(std::make_shared<v1::Subtract>(frame_length, get_dim(stft, 2)),
                                    i64_const({}, {0}),
                                    i64_const({}, {-1}),
                                    ov::element::i64);
    const auto mirrored = std::make_shared<v8::Gather>(stft, mirror_indices, bins_axis);
    const auto conj = std::make_shared<v1::Multiply>(
        mirrored,
        v0::Constant::create(data.get_element_type(), {2}, std::vector<float>{1.f, -1.f}));
    return std::make_shared<v0::Concat>(ov::OutputVector{stft, conj}, 2);
}

// STFT of complex signal [batch, signal_length, 2] using linearity: STFT(a + ib) = STFT(a) + i * STFT(b)
ov::Output<ov::Node> make_complex_stft(const ov::Output<ov::Node>& signal,
                                       const ov::OutputVector& inputs,
                                       bool window_provided,
                                       bool frame_length_provided) {
    // [batch, signal_length, 2] -> [2 * batch, signal_length], real parts first
    const auto parts = std::make_shared<v1::Transpose>(signal, i64_const({3}, {2, 0, 1}));
    const auto target_shape = std::make_shared<v0::Concat>(
        ov::OutputVector{i64_const({1}, {-1}), std::make_shared<v0::Unsqueeze>(get_dim(signal, 1), i64_const({}, {0}))},
        0);
    const auto data = std::make_shared<v1::Reshape>(parts, target_shape, false);

    const auto stft = make_real_stft(data, inputs, window_provided, frame_length_provided, false);
    const auto split = std::make_shared<v1::Split>(stft, i64_const({}, {0}), 2);
    const auto& real_stft = split->output(0);
    const auto& imag_stft = split->output(1);

    // i * (x + iy) = -y + ix
    const auto imag_rotated = std::make_shared<v1::Multiply>(
        std::make_shared<v8::Gather>(imag_stft, i64_const({2}, {1, 0}), i64_const({}, {-1})),
        v0::Constant::create(signal.get_element_type(), {2}, std::vector<float>{-1.f, 1.f}));
    return std::make_shared<v1::Add>(real_stft, imag_rotated);
}
}  // namespace

ov::OutputVector stft(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 2, 4);

    const ov::OutputVector ov_inputs{node.get_ov_inputs()};
    const auto& signal = ov_inputs.at(0);
    const auto& signal_shape = signal.get_partial_shape();
    const auto window_provided = ov_inputs.size() > 2 && !ov::op::util::is_null(ov_inputs[2]);
    const auto frame_length_provided = ov_inputs.size() > 3 && !ov::op::util::is_null(ov_inputs[3]);
    const auto onesided = node.get_attribute_value<int64_t>("onesided", 1) == 1;

    const auto is_int_or_dynamic = [](const ov::Output<ov::Node>& value) {
        return value.get_element_type().is_dynamic() || value.get_element_type().is_integral_number();
    };
    const auto is_scalar_like = [](const ov::Output<ov::Node>& value) {
        const auto& shape = value.get_partial_shape();
        return shape.rank().is_dynamic() || shape.size() == 0 || (shape.size() == 1 && shape[0].compatible(1));
    };
    CHECK_VALID_NODE(node, is_int_or_dynamic(ov_inputs[1]), "frame_step input must be of integer type.");
    CHECK_VALID_NODE(node, is_scalar_like(ov_inputs[1]), "frame_step input must be a scalar or Shape{1}.");
    if (frame_length_provided) {
        CHECK_VALID_NODE(node, is_int_or_dynamic(ov_inputs[3]), "frame_length input must be of integer type.");
        CHECK_VALID_NODE(node, is_scalar_like(ov_inputs[3]), "frame_length input must be a scalar or Shape{1}.");
    }
    if (window_provided) {
        const auto& window_shape = ov_inputs[2].get_partial_shape();
        CHECK_VALID_NODE(node, window_shape.rank().compatible(1), "The rank of window input must be 1D.");
        if (frame_length_provided && window_shape.is_static() &&
            ov::op::util::is_constant(ov_inputs[3].get_node_shared_ptr())) {
            const auto frame_length =
                ov::as_type_ptr<v0::Constant>(ov_inputs[3].get_node_shared_ptr())->cast_vector<int64_t>()[0];
            CHECK_VALID_NODE(node,
                             static_cast<int64_t>(window_shape[0].get_length()) == frame_length,
                             "The length of window input must be equal to frame_length.");
        }
    }

    // 2D signal [batch, signal_length] is not described in onnx spec, but it is allowed by onnx model checker
    // and seen in real models. It is treated as real signal.
    CHECK_VALID_NODE(node,
                     signal_shape.rank().is_static() && (signal_shape.size() == 2 || signal_shape.size() == 3),
                     "Signal input must be of rank 3 [batch_size, signal_length, 1 or 2] or 2 [batch_size, "
                     "signal_length]. Got: ",
                     signal_shape);
    if (signal_shape.size() == 2) {
        return {make_real_stft(signal, ov_inputs, window_provided, frame_length_provided, onesided)};
    }
    const auto& last_dim = signal_shape[2];
    CHECK_VALID_NODE(node,
                     last_dim.compatible(1) || last_dim.compatible(2),
                     "The last dimension of signal input must be 1 (real) or 2 (complex). Got: ",
                     last_dim);
    if (onesided) {
        CHECK_VALID_NODE(node, last_dim.compatible(1), "If attribute onesided==1, signal input can NOT be complex.");
    }
    if (last_dim.compatible(1) && (onesided || last_dim.is_static())) {
        const auto data = std::make_shared<v0::Squeeze>(signal, i64_const({1}, {2}));
        return {make_real_stft(data, ov_inputs, window_provided, frame_length_provided, onesided)};
    }

    ov::Output<ov::Node> complex_signal = signal;
    if (last_dim.is_dynamic()) {
        // Real or complex signal is not known, pad the last dimension to 2 with zero imaginary part
        const auto pads_end = std::make_shared<v0::Concat>(
            ov::OutputVector{i64_const({2}, {0, 0}),
                             std::make_shared<v1::Subtract>(
                                 i64_const({1}, {2}),
                                 std::make_shared<v0::Unsqueeze>(get_dim(signal, 2), i64_const({}, {0})))},
            0);
        complex_signal = std::make_shared<v12::Pad>(signal,
                                                    i64_const({3}, {0, 0, 0}),
                                                    pads_end,
                                                    v0::Constant::create(signal.get_element_type(), {}, {0}),
                                                    ov::op::PadMode::CONSTANT);
    }
    return {make_complex_stft(complex_signal, ov_inputs, window_provided, frame_length_provided)};
}

ONNX_OP("STFT", OPSET_SINCE(1), ai_onnx::opset_17::stft);
}  // namespace ov::frontend::onnx::ai_onnx::opset_17
