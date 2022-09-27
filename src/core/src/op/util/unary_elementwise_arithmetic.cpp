// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"

#include "itt.hpp"
#include "ngraph/op/util/elementwise_args.hpp"

BWDCMP_RTTI_DEFINITION(ov::op::util::UnaryElementwiseArithmetic);

ov::op::util::UnaryElementwiseArithmetic::UnaryElementwiseArithmetic() : Op() {}

ov::op::util::UnaryElementwiseArithmetic::UnaryElementwiseArithmetic(const Output<Node>& arg) : Op({arg}) {}

void ov::op::util::UnaryElementwiseArithmetic::validate_and_infer_elementwise_arithmetic() {
    auto args_et_pshape = op::util::validate_and_infer_elementwise_args(this);
    element::Type& args_et = std::get<0>(args_et_pshape);
    PartialShape& args_pshape = std::get<1>(args_et_pshape);

    NODE_VALIDATION_CHECK(this,
                          args_et.is_dynamic() || args_et != element::boolean,
                          "Arguments cannot have boolean element type (argument element type: ",
                          args_et,
                          ").");

    set_output_type(0, args_et, args_pshape);
}

void ov::op::util::UnaryElementwiseArithmetic::validate_and_infer_types() {
    OV_OP_SCOPE(util_UnaryElementwiseArithmetic_validate_and_infer_types);
    validate_and_infer_elementwise_arithmetic();
}

bool ov::op::util::UnaryElementwiseArithmetic::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(util_UnaryElementwiseArithmetic_visit_attributes);
    return true;
}
