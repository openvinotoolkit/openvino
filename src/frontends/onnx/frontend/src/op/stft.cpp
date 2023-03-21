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
    const auto tensor_iterator = std::make_shared<default_opset::TensorIterator>();
    const OutputVector ng_inputs{node.get_ng_inputs()};
    const auto& signal = ng_inputs.at(0);
    const auto dft_length_provided = ng_inputs.size() > 3 && !ngraph::op::is_null(ng_inputs[3]);
    const auto onesided = node.get_attribute_value<int64_t>("onesided", 0);
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
                     "Non-constant frame_step input is not supported.");  // TODO: CHECK IF SCALAR

    const int64_t nstfts =
        std::floor(signal_param_shape[axis].get_length() / frame_step) + 1;  // TODO: add support of length
    const int64_t length = nstfts;
    const auto axis_const = default_opset::Constant::create(element::i64, {}, {axis});
    ov::OutputVector concatenated_dft;
    for (int i = 0; i < nstfts; ++i) {
        std::vector<int64_t> indices(length);
        std::iota(std::begin(indices), std::end(indices), i * frame_step);
        const auto indices_const = default_opset::Constant::create(element::i64, Shape{indices.size()}, indices);
        const auto gather = std::make_shared<default_opset::Gather>(signal, indices_const, axis_const);
        concatenated_dft.push_back(dft::make_dft(gather,
                                                 dft_length_provided ? ng_inputs.at(1) : std::make_shared<NullNode>(),
                                                 axis,
                                                 false,
                                                 onesided == 1));
    }
    return {std::make_shared<default_opset::Concat>(concatenated_dft, axis)};
}

}  // namespace set_17

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
