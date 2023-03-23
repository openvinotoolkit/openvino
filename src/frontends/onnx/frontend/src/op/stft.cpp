// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/stft.hpp"

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "onnx_import/core/null_node.hpp"
#include "utils/common.hpp"
#include "utils/dft.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_17 {

OutputVector stft(const Node& node) {
    const OutputVector ng_inputs{node.get_ng_inputs()};
    auto signal = ng_inputs.at(0);
    //dft::try_convert_real_to_complex(signal);
    const auto dft_length_provided = ng_inputs.size() > 3 && !ngraph::op::is_null(ng_inputs[3]);
    const auto onesided = node.get_attribute_value<int64_t>("onesided", 1);
    const int64_t axis = 1;

    const auto& frame_step_node = ng_inputs.at(1);
    CHECK_VALID_NODE(node,
                     ngraph::op::is_constant(frame_step_node.get_node_shared_ptr()),
                     "Non-constant frame_step input is not supported.");  // TODO: CHECK IF SCALAR
    const auto frame_step =
        ov::as_type_ptr<default_opset::Constant>(frame_step_node.get_node_shared_ptr())->cast_vector<int64_t>()[0];
    const auto signal_param_shape = signal.get_partial_shape();
    CHECK_VALID_NODE(node,
                     signal_param_shape.rank().is_static() && signal_param_shape.rank().get_length() > axis &&
                         signal_param_shape[axis].is_static(),
                     "Shape of DFT axis must be known.");  // TODO: CHECK IF SCALAR

    const int64_t frame_length = 16; // TODO: Read from param or calculate
    const int64_t nstfts = std::floor((signal_param_shape[axis].get_length() - frame_length) / frame_step) + 1;
    //std::cout << "nstfts: " << nstfts  << ", len: " << signal_param_shape[axis].get_length() << ", frame_length: " << frame_length << ", frame_step: " << frame_step << std::endl;
    const auto axis_const = default_opset::Constant::create(element::i64, {}, {axis});
    const auto zero_const = default_opset::Constant::create(element::i64, {}, {0});
    ov::OutputVector concatenated_dft;
    for (int i = 0; i < nstfts; ++i) {
        std::vector<int64_t> indices(frame_length);
        std::iota(std::begin(indices), std::end(indices), i * frame_step);
        const auto indices_const = default_opset::Constant::create(element::i64, Shape{indices.size()}, indices);
        const auto gather = std::make_shared<default_opset::Reshape>(std::make_shared<default_opset::Gather>(signal, indices_const, axis_const), default_opset::Constant::create(element::i64, {1}, {-1}), false);
        //std::cout << "fft input shape: " << gather->get_output_partial_shape(0) << std::endl;
        const auto dft = dft::make_dft(gather,
                      dft_length_provided ? ng_inputs.at(1) : std::make_shared<NullNode>(),
                      0,
                      false,
                      onesided == 1);
        concatenated_dft.push_back(std::make_shared<default_opset::Unsqueeze>(dft, zero_const));
        //std::cout << "fft output shape: " << concatenated_dft.back().get_partial_shape() << std::endl;
    }
    return {std::make_shared<default_opset::Unsqueeze>(std::make_shared<default_opset::Concat>(concatenated_dft, 0), zero_const)}; // TODO: Try to optimize this with previous Unsqueeze
}

}  // namespace set_17

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
