// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

#include "itt.hpp"

ov::op::util::UnaryElementwiseArithmetic::UnaryElementwiseArithmetic() : Op() {}

ov::op::util::UnaryElementwiseArithmetic::UnaryElementwiseArithmetic(const Output<Node>& arg) : Op({arg}) {}

void ov::op::util::UnaryElementwiseArithmetic::validate_and_infer_elementwise_arithmetic() {
    const auto& element_type = get_input_element_type(0);
    const auto is_supported_et = (element_type != element::boolean && element_type != element::string);
    NODE_VALIDATION_CHECK(this,
                          is_supported_et,
                          "This operation does not support input with element type: ",
                          element_type);

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
