// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/multiclass_nms.hpp"

#include "itt.hpp"
#include "multiclass_nms_shape_inference.hpp"

namespace ov {
namespace op {
// ------------------------------ V8 ------------------------------
namespace v8 {

MulticlassNms::MulticlassNms(const Output<Node>& boxes, const Output<Node>& scores, const Attributes& attrs)
    : MulticlassNmsBase({boxes, scores}, attrs) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> MulticlassNms::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(MulticlassNms_v8_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() >= 2, "Number of inputs must be 2 at least");

    return std::make_shared<MulticlassNms>(new_args.at(0), new_args.at(1), m_attrs);
}

void MulticlassNms::validate_and_infer_types() {
    OV_OP_SCOPE(MulticlassNms_v9_validate_and_infer_types);
    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    const auto output_shapes = shape_infer(this, input_shapes, false);

    validate();

    const auto& output_type = get_attrs().output_type;
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
    set_output_type(1, output_type, output_shapes[1]);
    set_output_type(2, output_type, output_shapes[2]);
}
}  // namespace v8

// ------------------------------ V9 ------------------------------
namespace v9 {
MulticlassNms::MulticlassNms(const Output<Node>& boxes, const Output<Node>& scores, const Attributes& attrs)
    : MulticlassNmsBase({boxes, scores}, attrs) {
    constructor_validate_and_infer_types();
}

MulticlassNms::MulticlassNms(const Output<Node>& boxes,
                             const Output<Node>& scores,
                             const Output<Node>& roisnum,
                             const Attributes& attrs)
    : MulticlassNmsBase({boxes, scores, roisnum}, attrs) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> MulticlassNms::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(MulticlassNms_v9_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() == 2 || new_args.size() == 3, "Number of inputs must be 2 or 3");

    switch (new_args.size()) {
    case 3:
        return std::make_shared<MulticlassNms>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
    default:
        return std::make_shared<MulticlassNms>(new_args.at(0), new_args.at(1), m_attrs);
    }
}

void MulticlassNms::validate_and_infer_types() {
    OV_OP_SCOPE(MulticlassNms_v9_validate_and_infer_types);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    const auto output_shapes = shape_infer(this, input_shapes, false);

    validate();

    const auto& output_type = get_attrs().output_type;
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
    set_output_type(1, output_type, output_shapes[1]);
    set_output_type(2, output_type, output_shapes[2]);
}
}  // namespace v9
}  // namespace op
}  // namespace ov
