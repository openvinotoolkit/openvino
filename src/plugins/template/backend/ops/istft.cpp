// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/istft.hpp"

#include "evaluate_node.hpp"
#include "evaluates_map.hpp"
#include "istft_shape_inference.hpp"
#include "openvino/reference/fft.hpp"
#include "openvino/runtime/tensor.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v16::ISTFT>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto input_shapes = ov::util::get_tensors_partial_shapes(inputs);
    const auto output_shape =
        ov::op::v16::shape_infer(op.get(), input_shapes, make_tensor_accessor(inputs)).front().to_shape();

    outputs[0].set_shape(output_shape);

    const auto frame_size = ov::get_tensor_data_as<int64_t>(inputs[2]).front();
    const auto frame_step = ov::get_tensor_data_as<int64_t>(inputs[3]).front();
    int64_t length = -1;
    if (inputs.size() == 5) {
        length = ov::get_tensor_data_as<int64_t>(inputs[4]).front();
    }

    const std::vector<float> data_f32 = get_floats(inputs[0], inputs[0].get_shape());
    const std::vector<float> window_f32 = get_floats(inputs[1], inputs[1].get_shape());
    std::vector<float> result_f32(ov::shape_size(output_shape), 0.f);

    ov::reference::istft(data_f32.data(),
                         window_f32.data(),
                         result_f32.data(),
                         inputs[0].get_shape(),
                         inputs[1].get_shape(),
                         frame_size,
                         frame_step,
                         length,
                         op->get_center(),
                         op->get_normalized());

    const auto& output_type = op->get_input_element_type(0);
    ov::reference::fft_postprocessing(outputs, output_type, result_f32);
    return true;
}

template <>
bool evaluate_node<ov::op::v16::ISTFT>(std::shared_ptr<ov::Node> node,
                                       ov::TensorVector& outputs,
                                       const ov::TensorVector& inputs) {
    switch (node->get_input_element_type(0)) {
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v16::ISTFT>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v16::ISTFT>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v16::ISTFT>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_input_element_type(0), " in evaluate_node()");
    }
}
