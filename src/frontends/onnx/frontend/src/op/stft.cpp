// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/stft.hpp"

#include "default_opset.hpp"
#include "onnx_import/core/null_node.hpp"
#include "utils/dft.hpp"
#include "utils/common.hpp"
#include "exceptions.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_17 {

OutputVector stft(const Node& node) {
    const auto tensor_iterator = std::make_shared<default_opset::TensorIterator>();
    const OutputVector ng_inputs{node.get_ng_inputs()};
    const auto& signal = ng_inputs.at(0);
    std::cout << "signal: " << signal << "\n";
    const auto dft_length_provided = ng_inputs.size() > 3 && !ngraph::op::is_null(ng_inputs[3]);
    const auto onesided = node.get_attribute_value<int64_t>("onesided", 0);
    const int64_t axis = 1;

    const auto& frame_step_node = ng_inputs.at(1);
    CHECK_VALID_NODE(node,
                    ngraph::op::is_constant(frame_step_node.get_node_shared_ptr()),
                    "Non-constant frame_step input is not supported."); // TODO: CHECK IF SCALAR
    const auto frame_step = ov::as_type_ptr<default_opset::Constant>(frame_step_node.get_node_shared_ptr())->cast_vector<int64_t>()[0];
    auto signal_param_shape = signal.get_partial_shape();
    CHECK_VALID_NODE(node,
                    signal_param_shape.rank().is_static() && signal_param_shape.rank().get_length() > axis && signal_param_shape[axis].is_static(),
                    "Non-constant frame_step input is not supported."); // TODO: CHECK IF SCALAR
    std::cout << "shape before: " << signal_param_shape << "\n";
    signal_param_shape[axis] /= frame_step;
    std::cout << "shape after: " << signal_param_shape << "\n";
    const auto signal_param = std::make_shared<default_opset::Parameter>(signal.get_element_type(), signal_param_shape);

    const auto dft = dft::make_dft(signal_param, dft_length_provided ? ng_inputs.at(1) : std::make_shared<NullNode>(), axis, false, onesided == 1);
    const auto dft_res = std::make_shared<default_opset::Result>(dft);
    const auto body = std::make_shared<ov::Model>(ResultVector{dft_res}, ParameterVector{signal_param});

    tensor_iterator->set_body(body);
    tensor_iterator->set_sliced_input(signal_param, signal, 0, 1, frame_step, -1, axis);

    return {tensor_iterator->get_concatenated_slices(dft_res, 0, 1, frame_step, -1, axis)};
}

}  // namespace set_17

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
