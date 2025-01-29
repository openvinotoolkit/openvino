// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/istft.hpp"

#include <memory>

#include "istft_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/reference/istft.hpp"

namespace ov {
namespace op {
namespace v16 {
namespace {
void check_int_input_at(const Node* op, size_t input_idx) {
    const auto& in_type = op->get_input_element_type(input_idx);
    const auto has_valid_type = in_type.is_dynamic() || in_type == element::i32 || in_type == element::i64;
    NODE_VALIDATION_CHECK(op, has_valid_type, "Expected i32 or i64 type of the input at port: ", input_idx);
}
}  // namespace
ISTFT::ISTFT(const Output<Node>& data,
             const Output<Node>& window,
             const Output<Node>& frame_size,
             const Output<Node>& frame_step,
             const bool center,
             const bool normalized)
    : Op({data, window, frame_size, frame_step}),
      m_center(center),
      m_normalized(normalized) {
    constructor_validate_and_infer_types();
}

ISTFT::ISTFT(const Output<Node>& data,
             const Output<Node>& window,
             const Output<Node>& frame_size,
             const Output<Node>& frame_step,
             const Output<Node>& signal_length,
             const bool center,
             const bool normalized)
    : Op({data, window, frame_size, frame_step, signal_length}),
      m_center(center),
      m_normalized(normalized) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> ISTFT::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v16_ISTFT_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    if (new_args.size() == 4) {
        return std::make_shared<ISTFT>(new_args.at(0),
                                       new_args.at(1),
                                       new_args.at(2),
                                       new_args.at(3),
                                       m_center,
                                       m_normalized);
    }
    return std::make_shared<ISTFT>(new_args.at(0),
                                   new_args.at(1),
                                   new_args.at(2),
                                   new_args.at(3),
                                   new_args.at(4),
                                   m_center,
                                   m_normalized);
}

bool ISTFT::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v16_ISTFT_visit_attributes);
    visitor.on_attribute("center", m_center);
    visitor.on_attribute("normalized", m_normalized);
    return true;
}

void ISTFT::validate_and_infer_types() {
    OV_OP_SCOPE(v16_ISTFT_validate_and_infer_types);
    const auto input_size = get_input_size();
    const auto is_in_count_correct = input_size == 4 || input_size == 5;
    NODE_VALIDATION_CHECK(this, is_in_count_correct, "Expected 4 or 5 inputs to be provided.");

    auto data_type = get_input_element_type(0);
    const auto& window_type = get_input_element_type(1);

    const auto has_valid_data_type = data_type.is_dynamic() || data_type.is_real();
    NODE_VALIDATION_CHECK(this, has_valid_data_type, "Expected floating point type of the 'data' input.");

    const auto has_valid_window_type =
        window_type.is_dynamic() || (window_type.is_real() && element::Type::merge(data_type, window_type, data_type));
    NODE_VALIDATION_CHECK(this,
                          has_valid_window_type,
                          "Expected floating point type of the 'window' input, matching the type of `data` input.");

    check_int_input_at(this, 2);
    check_int_input_at(this, 3);
    if (input_size == 5) {
        check_int_input_at(this, 4);
    }

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, data_type, output_shapes[0]);
}

bool ISTFT::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v16_ISTFT_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto input_shapes = ov::util::get_tensors_partial_shapes(inputs);
    const auto output_shape = shape_infer(this, input_shapes, make_tensor_accessor(inputs)).front().to_shape();

    outputs[0].set_shape(output_shape);

    const auto frame_size = ov::get_tensor_data_as<int64_t>(inputs[2]).front();
    const auto frame_step = ov::get_tensor_data_as<int64_t>(inputs[3]).front();
    int64_t length = -1;
    if (inputs.size() == 5) {
        length = ov::get_tensor_data_as<int64_t>(inputs[4]).front();
    }

    ov::reference::istft(inputs[0].data<const float>(),
                         inputs[1].data<const float>(),
                         outputs[0].data<float>(),
                         inputs[0].get_shape(),
                         inputs[1].get_shape(),
                         frame_size,
                         frame_step,
                         length,
                         m_center,
                         m_normalized);
    return true;
}

bool ISTFT::has_evaluate() const {
    OV_OP_SCOPE(v16_ISTFT_has_evaluate);
    const auto& input_0_et = get_input_element_type(0);
    return input_0_et == element::f32;
}

bool ISTFT::get_center() const {
    OV_OP_SCOPE(v16_ISTFT_get_center);
    return m_center;
}

void ISTFT::set_center(const bool center) {
    OV_OP_SCOPE(v16_ISTFT_set_center);
    m_center = center;
}

bool ISTFT::get_normalized() const {
    OV_OP_SCOPE(v16_ISTFT_get_normalized);
    return m_normalized;
}

void ISTFT::set_normalized(const bool normalized) {
    OV_OP_SCOPE(v16_ISTFT_set_normalized);
    m_normalized = normalized;
}

}  // namespace v16
}  // namespace op
}  // namespace ov
