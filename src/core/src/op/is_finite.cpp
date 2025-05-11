// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/is_finite.hpp"

#include "itt.hpp"

ov::op::v10::IsFinite::IsFinite(const Output<Node>& data) : op::Op{{data}} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::op::v10::IsFinite::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v10_IsFinite_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<IsFinite>(new_args.at(0));
}

void ov::op::v10::IsFinite::validate_and_infer_types() {
    OV_OP_SCOPE(v10_IsFinite_validate_and_infer_types);
    element::Type input_element_type = get_input_element_type(0);
    element::Type output_element_type = ov::element::boolean;
    ov::PartialShape input_pshape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "The element type of the input tensor must be a floating point number or dynamic (got ",
                          input_element_type,
                          ").");
    set_output_type(0, output_element_type, input_pshape);
}

bool ov::op::v10::IsFinite::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v10_IsFinite_visit_attributes);
    return true;
}
