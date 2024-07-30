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
namespace {
void check_int_input_at(const Node* op, size_t input_idx) {
    const auto& in_type = op->get_input_element_type(input_idx);
    const auto has_valid_type = in_type.is_dynamic() || in_type == element::i32 || in_type == element::i64;
    NODE_VALIDATION_CHECK(op, has_valid_type, "Expected i32 or i64 type of the input at port: ", input_idx);
}
}  // namespace
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
    NODE_VALIDATION_CHECK(this, get_input_size() == 4, "Expected 4 inputs to be provided.");

    auto signal_type = get_input_element_type(0);
    const auto& window_type = get_input_element_type(1);

    const auto has_valid_signal_type = signal_type.is_dynamic() || signal_type.is_real();
    NODE_VALIDATION_CHECK(this, has_valid_signal_type, "Expected floating point type of the 'signal' input.");

    const auto has_valid_window_type =
        window_type.is_dynamic() ||
        (window_type.is_real() && element::Type::merge(signal_type, window_type, signal_type));
    NODE_VALIDATION_CHECK(this,
                          has_valid_window_type,
                          "Expected floating point type of the 'window' input, matching the type of `signal` input.");

    check_int_input_at(this, 2);
    check_int_input_at(this, 3);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, signal_type, output_shapes[0]);
}

bool STFT::get_transpose_frames() const {
    return m_transpose_frames;
}

void STFT::set_transpose_frames(const bool transpose_frames) {
    m_transpose_frames = transpose_frames;
}

}  // namespace v15
}  // namespace op
}  // namespace ov
