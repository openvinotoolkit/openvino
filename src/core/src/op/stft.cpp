// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/stft.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/reference/stft.hpp"
#include "rdft_shape_inference.hpp"

namespace ov {
namespace op {
namespace v15 {

STFT::STFT(const Output<Node>& data,
           const Output<Node>& window,
           const Output<Node>& signal_size,
           int64_t frame_step,
           bool frames_first)
    : Op({data, window, signal_size}),
      m_frame_step(frame_step),
      m_frames_first(frames_first) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> STFT::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v15_STFT_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    return std::make_shared<STFT>(new_args.at(0), new_args.at(1), new_args.at(2), m_frame_step, m_frames_first);
}

bool STFT::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v15_STFT_visit_attributes);
    visitor.on_attribute("frame_step", m_frame_step);
    visitor.on_attribute("frames_first", m_frames_first);
    return true;
}

void STFT::validate_and_infer_types() {
    OV_OP_SCOPE(v15_STFT_validate_and_infer_types);

    // TODO: Add validation for input shapes
    // TODO: Move to shape_infer

    const auto& data_shape = get_input_partial_shape(0);
    const ITensorAccessor& ta = make_tensor_accessor();
    constexpr auto signal_size_port = 2;
    auto signal_size = get_input_const_data_as<ov::PartialShape, int64_t>(this, signal_size_port, ta);

    if (!signal_size) {
        ov::PartialShape output_shape{data_shape[0], {1, -1}, {1, -1}, 2};
        set_output_type(0, get_input_element_type(0), output_shape);
        return;
    }

    // InShape:  [Batch, L]
    // OutShape: [Batch, floor(signal_size//2) + 1, T => floor(L-signal_size)//frame_step) + 1, 2]
    // Requirements: L >= signal_size

    // Torch out shape
    // ov::PartialShape output_shape{data_shape[0], ((*signal_size)[0] / 2) + 1, ((data_shape[1] - (*signal_size)[0]) /
    // m_frame_step) + 1, 2}; ONNX out shape (transposed)

    ov::PartialShape output_shape;
    if (m_frames_first) {  // [batch, frames, fft_samples, 2]
        output_shape = ov::PartialShape{data_shape[0],
                                        ((data_shape[1] - (*signal_size)[0]) / m_frame_step) + 1,
                                        ((*signal_size)[0] / 2) + 1,
                                        2};
    } else {  // [batch, fft_samples, frames, 2]
        output_shape = ov::PartialShape{data_shape[0],
                                        ((*signal_size)[0] / 2) + 1,
                                        ((data_shape[1] - (*signal_size)[0]) / m_frame_step) + 1,
                                        2};
    }

    set_output_type(0, get_input_element_type(0), output_shape);
}

bool STFT::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v15_STFT_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 3);

    const auto signal_size = ov::get_tensor_data_as<int64_t>(inputs[2]).front();

    ov::reference::stft(inputs[0].data<const float>(),
                        inputs[1].data<const float>(),
                        outputs[0].data<float>(),
                        inputs[0].get_shape(),
                        inputs[1].get_shape(),
                        signal_size,
                        m_frame_step,
                        m_frames_first);
    return true;
}

bool STFT::has_evaluate() const {
    OV_OP_SCOPE(v15_STFT_has_evaluate);
    const auto& input_0_et = get_input_element_type(0);
    return input_0_et == element::f32;
}

}  // namespace v15
}  // namespace op
}  // namespace ov
