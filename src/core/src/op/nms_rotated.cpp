// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/nms_rotated.hpp"

#include "itt.hpp"
#include "nms_shape_inference.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/op/util/op_types.hpp"

namespace ov {

namespace op {
namespace nms_rotated {
namespace validate {
namespace {
void input_types(const Node* op) {
    const auto inputs_size = op->get_input_size();

    NODE_VALIDATION_CHECK(op, inputs_size == 5, "Expected 5 inputs to be provided.");
    constexpr size_t integer_input_idx = 2;
    for (size_t i = 0; i < inputs_size; ++i) {
        if (i == integer_input_idx) {
            NODE_VALIDATION_CHECK(op,
                                  op->get_input_element_type(integer_input_idx).is_integral_number() ||
                                      op->get_input_element_type(integer_input_idx).is_dynamic(),
                                  "Expected integer type as element type for the input at: 2");
        } else {
            NODE_VALIDATION_CHECK(op,
                                  op->get_input_element_type(i).is_real() || op->get_input_element_type(i).is_dynamic(),
                                  "Expected floating point type as element type for the input at: ",
                                  i);
        }
    }
}
}  // namespace
}  // namespace validate
}  // namespace nms_rotated
}  // namespace op
// ------------------------------ v13 ------------------------------

op::v13::NMSRotated::NMSRotated(const Output<Node>& boxes,
                                const Output<Node>& scores,
                                const Output<Node>& max_output_boxes_per_class,
                                const Output<Node>& iou_threshold,
                                const Output<Node>& score_threshold,
                                const bool sort_result_descending,
                                const element::Type& output_type,
                                const bool clockwise)
    : Op({boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold}),
      m_sort_result_descending{sort_result_descending},
      m_output_type{output_type},
      m_clockwise{clockwise} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v13::NMSRotated::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v13_NMSRotated_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() == 5, "Number of inputs must be 5");

    return std::make_shared<op::v13::NMSRotated>(new_args.at(0),
                                                 new_args.at(1),
                                                 new_args.at(2),
                                                 new_args.at(3),
                                                 new_args.at(4),
                                                 m_sort_result_descending,
                                                 m_output_type,
                                                 m_clockwise);
}

bool op::v13::NMSRotated::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v13_NMSRotated_visit_attributes);
    visitor.on_attribute("sort_result_descending", m_sort_result_descending);
    visitor.on_attribute("output_type", m_output_type);
    visitor.on_attribute("clockwise", m_clockwise);
    return true;
}

void op::v13::NMSRotated::validate_and_infer_types() {
    OV_OP_SCOPE(v13_NMSRotated_validate_and_infer_types);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    const auto output_shapes = shape_infer(this, input_shapes);

    nms_rotated::validate::input_types(this);
    NODE_VALIDATION_CHECK(this,
                          m_output_type == element::i64 || m_output_type == element::i32,
                          "The `output_type` attribute (related to the first and third output) must be i32 or i64.");

    set_output_type(0, m_output_type, output_shapes[0]);
    set_output_type(1, element::f32, output_shapes[1]);
    set_output_type(2, m_output_type, output_shapes[2]);
}

bool op::v13::NMSRotated::get_sort_result_descending() const {
    return m_sort_result_descending;
}
void op::v13::NMSRotated::set_sort_result_descending(const bool sort_result_descending) {
    m_sort_result_descending = sort_result_descending;
}

element::Type op::v13::NMSRotated::get_output_type_attr() const {
    return m_output_type;
}
void op::v13::NMSRotated::set_output_type_attr(const element::Type& output_type) {
    m_output_type = output_type;
}

bool op::v13::NMSRotated::get_clockwise() const {
    return m_clockwise;
}
void op::v13::NMSRotated::set_clockwise(const bool clockwise) {
    m_clockwise = clockwise;
}

}  // namespace ov
