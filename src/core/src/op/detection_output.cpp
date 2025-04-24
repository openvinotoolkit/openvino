// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/detection_output.hpp"

#include "detection_output_shape_inference.hpp"
#include "itt.hpp"

// ------------------------------ V0 ------------------------------
ov::op::v0::DetectionOutput::DetectionOutput(const Output<Node>& box_logits,
                                             const Output<Node>& class_preds,
                                             const Output<Node>& proposals,
                                             const Output<Node>& aux_class_preds,
                                             const Output<Node>& aux_box_preds,
                                             const Attributes& attrs)
    : DetectionOutputBase({box_logits, class_preds, proposals, aux_class_preds, aux_box_preds}),
      m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

ov::op::v0::DetectionOutput::DetectionOutput(const Output<Node>& box_logits,
                                             const Output<Node>& class_preds,
                                             const Output<Node>& proposals,
                                             const Attributes& attrs)
    : DetectionOutputBase({box_logits, class_preds, proposals}),
      m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

void ov::op::v0::DetectionOutput::validate_and_infer_types() {
    OV_OP_SCOPE(v0_DetectionOutput_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, m_attrs.num_classes > 0, "Number of classes must be greater than zero");
    validate_base(m_attrs);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

std::shared_ptr<ov::Node> ov::op::v0::DetectionOutput::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_DetectionOutput_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    auto num_args = new_args.size();

    NODE_VALIDATION_CHECK(this, num_args == 3 || num_args == 5, "DetectionOutput accepts 3 or 5 inputs.");

    if (num_args == 3) {
        return std::make_shared<DetectionOutput>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
    } else {
        return std::make_shared<DetectionOutput>(new_args.at(0),
                                                 new_args.at(1),
                                                 new_args.at(2),
                                                 new_args.at(3),
                                                 new_args.at(4),
                                                 m_attrs);
    }
}

bool ov::op::v0::DetectionOutput::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_DetectionOutput_visit_attributes);
    visitor.on_attribute("num_classes", m_attrs.num_classes);
    visit_attributes_base(visitor, m_attrs);
    return true;
}

// ------------------------------ V8 ------------------------------
ov::op::v8::DetectionOutput::DetectionOutput(const Output<Node>& box_logits,
                                             const Output<Node>& class_preds,
                                             const Output<Node>& proposals,
                                             const Output<Node>& aux_class_preds,
                                             const Output<Node>& aux_box_preds,
                                             const Attributes& attrs)
    : DetectionOutputBase({box_logits, class_preds, proposals, aux_class_preds, aux_box_preds}),
      m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

ov::op::v8::DetectionOutput::DetectionOutput(const Output<Node>& box_logits,
                                             const Output<Node>& class_preds,
                                             const Output<Node>& proposals,
                                             const Attributes& attrs)
    : DetectionOutputBase({box_logits, class_preds, proposals}),
      m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

void ov::op::v8::DetectionOutput::validate_and_infer_types() {
    OV_OP_SCOPE(v8_DetectionOutput_validate_and_infer_types);
    validate_base(m_attrs);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

std::shared_ptr<ov::Node> ov::op::v8::DetectionOutput::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_DetectionOutput_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    auto num_args = new_args.size();

    NODE_VALIDATION_CHECK(this, num_args == 3 || num_args == 5, "DetectionOutput accepts 3 or 5 inputs.");

    if (num_args == 3) {
        return std::make_shared<DetectionOutput>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
    } else {
        return std::make_shared<DetectionOutput>(new_args.at(0),
                                                 new_args.at(1),
                                                 new_args.at(2),
                                                 new_args.at(3),
                                                 new_args.at(4),
                                                 m_attrs);
    }
}

bool ov::op::v8::DetectionOutput::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_DetectionOutput_visit_attributes);
    visit_attributes_base(visitor, m_attrs);
    return true;
}
