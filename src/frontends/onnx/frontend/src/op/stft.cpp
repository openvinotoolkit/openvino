// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/stft.hpp"

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "onnx_import/core/null_node.hpp"
#include "openvino/op/util/op_types.hpp"
#include "utils/common.hpp"
#include "utils/dft.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_17 {

OutputVector stft(const Node& node) {
    const OutputVector ng_inputs{node.get_ng_inputs()};
    auto signal = ng_inputs.at(0);
    const auto dft_length_provided = ng_inputs.size() > 3 && !ngraph::op::is_null(ng_inputs[3]);
    const auto onesided = node.get_attribute_value<int64_t>("onesided", 1);
    const int64_t axis = 1;

    const auto& frame_step_node = ng_inputs.at(1);
    CHECK_VALID_NODE(node,
                     ov::op::util::is_constant(frame_step_node.get_node_shared_ptr()) &&
                         ov::shape_size(frame_step_node.get_shape()) <= 1,
                     "frame_step input must be a scalar or Shape{1} constant.");
    const auto frame_step =
        ov::as_type_ptr<default_opset::Constant>(frame_step_node.get_node_shared_ptr())->cast_vector<int64_t>()[0];
    const auto signal_param_shape = signal.get_partial_shape();
    CHECK_VALID_NODE(node,
                     signal_param_shape.is_static() && signal_param_shape.size() == 3,
                     "Shape of signal input must be static with the rank equal to 3.");

    int64_t frame_length = signal_param_shape[axis].get_length() / frame_step;  // default value
    if (dft_length_provided) {
        const auto& frame_length_node = ng_inputs[3];
        CHECK_VALID_NODE(node,
                         ov::op::util::is_constant(frame_length_node.get_node_shared_ptr()) &&
                             ov::shape_size(frame_length_node.get_shape()) <= 1,
                         "frame_length input must be a scalar or Shape{1} constant.");
        frame_length = ov::as_type_ptr<default_opset::Constant>(frame_length_node.get_node_shared_ptr())
                           ->cast_vector<int64_t>()[0];
    }

    const auto window_node_provided = ng_inputs.size() > 2 && !ngraph::op::is_null(ng_inputs[2]);
    if (window_node_provided) {  // window input provided
        if (ng_inputs[2].get_partial_shape().rank().is_static()) {
            CHECK_VALID_NODE(node,
                             ng_inputs[2].get_partial_shape().rank().get_length() == 1,
                             "The rank of window input must be 1D.");
            if (ng_inputs[2].get_partial_shape()[0].is_static()) {
                CHECK_VALID_NODE(node,
                                 ng_inputs[2].get_partial_shape()[0].get_length() == frame_length,
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
    const auto axis_const = default_opset::Constant::create(element::i64, {}, {axis});
    const auto zero_const = default_opset::Constant::create(element::i64, {}, {0});
    const auto step = default_opset::Constant::create(element::i64, Shape{2}, {1, 1});
    ov::OutputVector all_signals;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        ov::OutputVector signals_in_batch;
        for (int64_t sig_idx = 0; sig_idx < nstfts; ++sig_idx) {
            const auto start = default_opset::Constant::create(element::i64,
                                                               Shape{2},
                                                               std::vector<int64_t>{batch, sig_idx * frame_step});
            const auto stop =
                default_opset::Constant::create(element::i64,
                                                Shape{2},
                                                std::vector<int64_t>{batch + 1, sig_idx * frame_step + frame_length});
            const auto slice_axes =
                default_opset::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{0, axis});
            const auto slice = std::make_shared<default_opset::Slice>(signal, start, stop, step, slice_axes);
            const ov::Output<ov::Node> flatten_slice = std::make_shared<default_opset::Reshape>(
                slice,
                is_complex(slice) ? default_opset::Constant::create(element::i64, {2}, {-1, 2})
                                  : (onesided ? default_opset::Constant::create(element::i64, {1}, {-1})
                                              : default_opset::Constant::create(element::i64, {2}, {-1, 1})),
                false);
            const auto dft = dft::make_dft(
                window_node_provided
                    ? std::make_shared<default_opset::Multiply>(
                          flatten_slice,
                          is_complex(flatten_slice)
                              ? std::make_shared<default_opset::Broadcast>(  // align window shape with signal shape
                                    std::make_shared<default_opset::Unsqueeze>(
                                        ng_inputs[2],
                                        default_opset::Constant::create(element::i64, {1}, {1})),
                                    std::make_shared<default_opset::ShapeOf>(flatten_slice))
                              : ng_inputs[2])
                    : flatten_slice,
                dft_length_provided ? ng_inputs[3] : std::make_shared<NullNode>(),
                0,
                false,
                onesided == 1);
            signals_in_batch.push_back(std::make_shared<default_opset::Unsqueeze>(dft, zero_const));
        }
        all_signals.push_back(
            std::make_shared<default_opset::Unsqueeze>(std::make_shared<default_opset::Concat>(signals_in_batch, 0),
                                                       zero_const));
    }
    return {std::make_shared<default_opset::Concat>(all_signals, 0)};
}

}  // namespace set_17

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
