// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/stft.hpp"

#include "evaluate_node.hpp"
#include "evaluates_map.hpp"
#include "openvino/reference/fft.hpp"
#include "openvino/runtime/tensor.hpp"
#include "stft_shape_inference.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v15::STFT>& op, ov::TensorVector& outputs, const ov::TensorVector& inputs) {
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto input_shapes = ov::util::get_tensors_partial_shapes(inputs);
    const auto output_shape =
        ov::op::v15::shape_infer(op.get(), input_shapes, make_tensor_accessor(inputs)).front().to_shape();

    outputs[0].set_shape(output_shape);

    const auto frame_size = ov::get_tensor_data_as<int64_t>(inputs[2]).front();
    const auto frame_step = ov::get_tensor_data_as<int64_t>(inputs[3]).front();

    const std::vector<float> signal_f32 = get_floats(inputs[0], inputs[0].get_shape());
    const std::vector<float> window_f32 = get_floats(inputs[1], inputs[1].get_shape());
    std::vector<float> result_f32(shape_size(output_shape), 0.f);

    ov::reference::stft(signal_f32.data(),
                        window_f32.data(),
                        result_f32.data(),
                        inputs[0].get_shape(),
                        inputs[1].get_shape(),
                        frame_size,
                        frame_step,
                        op->get_transpose_frames());

    const auto& output_type = op->get_input_element_type(0);
    ov::reference::fft_postprocessing(outputs, output_type, result_f32);
    return true;
}

template <>
bool evaluate_node<ov::op::v15::STFT>(std::shared_ptr<ov::Node> node,
                                      ov::TensorVector& outputs,
                                      const ov::TensorVector& inputs) {
    switch (node->get_input_element_type(0)) {
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v15::STFT>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v15::STFT>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v15::STFT>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_input_element_type(0), " in evaluate_node()");
    }
}
