// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/stft.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/reference/stft.hpp"
#include "stft_shape_inference.hpp"

namespace ov {
namespace op {
namespace v15 {

STFT::STFT(const Output<Node>& data,
           const Output<Node>& window,
           const Output<Node>& frame_size,
           const Output<Node>& frame_step,
           const bool transpose_frames)
    : Op({data, window, frame_size, frame_step}),
      m_transpose_frames(transpose_frames) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> STFT::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v15_STFT_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    return std::make_shared<STFT>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_transpose_frames);
}

bool STFT::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v15_STFT_visit_attributes);
    visitor.on_attribute("transpose_frames", m_transpose_frames);
    return true;
}

void STFT::validate_and_infer_types() {
    OV_OP_SCOPE(v15_STFT_validate_and_infer_types);

    // TODO: Add input types validation

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

bool STFT::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v15_STFT_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 4);

    const auto frame_size = ov::get_tensor_data_as<int64_t>(inputs[2]).front();
    const auto frame_step = ov::get_tensor_data_as<int64_t>(inputs[3]).front();

    // TODO: Reuse shape_infer to set shape of output tensor
    const auto& data_shape = inputs[0].get_shape();

    NODE_VALIDATION_CHECK(this,
                          0 < frame_size && static_cast<size_t>(frame_size) < data_shape[1],
                          "Provided frame size is ",
                          frame_size,
                          " but must be in range {0, ",
                          data_shape[1],
                          "}");

    Shape output_shape;
    const size_t frame_size_dim = static_cast<size_t>(frame_size);
    const size_t frames_dim = ((data_shape[1] - frame_size_dim) / frame_step) + 1;
    const size_t fft_samples_dim = (frame_size / 2) + 1;
    constexpr size_t complex_dim = 2;
    if (!m_transpose_frames) {
        output_shape = Shape{data_shape[0], frames_dim, fft_samples_dim, complex_dim};
    } else {
        output_shape = Shape{data_shape[0], fft_samples_dim, frames_dim, complex_dim};
    }
    outputs[0].set_shape(output_shape);

    ov::reference::stft(inputs[0].data<const float>(),
                        inputs[1].data<const float>(),
                        outputs[0].data<float>(),
                        inputs[0].get_shape(),
                        inputs[1].get_shape(),
                        frame_size,
                        frame_step,
                        m_transpose_frames);
    return true;
}

bool STFT::has_evaluate() const {
    OV_OP_SCOPE(v15_STFT_has_evaluate);
    const auto& input_0_et = get_input_element_type(0);
    return input_0_et == element::f32;
}

bool STFT::get_transpose_frames() const {
    return m_transpose_frames;
}

}  // namespace v15
}  // namespace op
}  // namespace ov
