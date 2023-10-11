// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

#include "itt.hpp"

ov::op::util::UnaryElementwiseArithmetic::UnaryElementwiseArithmetic() : Op() {}

ov::op::util::UnaryElementwiseArithmetic::UnaryElementwiseArithmetic(const Output<Node>& arg) : Op({arg}) {}

void ov::op::util::UnaryElementwiseArithmetic::validate_and_infer_elementwise_arithmetic() {
    const auto& element_type = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          element_type.is_dynamic() || element_type != element::boolean,
                          "Arguments cannot have boolean element type (argument element type: ",
                          element_type,
                          ").");

    const auto& arg_pshape = get_input_partial_shape(0);
    set_output_type(0, element_type, arg_pshape);
}

void ov::op::util::UnaryElementwiseArithmetic::validate_and_infer_types() {
    OV_OP_SCOPE(util_UnaryElementwiseArithmetic_validate_and_infer_types);
    validate_and_infer_elementwise_arithmetic();
}

bool ov::op::util::UnaryElementwiseArithmetic::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(util_UnaryElementwiseArithmetic_visit_attributes);
    return true;
}
