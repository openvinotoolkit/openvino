// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "itt.hpp"
#include "ngraph/op/util/elementwise_args.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::util::UnaryElementwiseArithmetic, "UnaryElementwiseArithmetic", 0);

op::util::UnaryElementwiseArithmetic::UnaryElementwiseArithmetic()
    : Op()
{
}

op::util::UnaryElementwiseArithmetic::UnaryElementwiseArithmetic(const Output<Node>& arg)
    : Op({arg})
{
}

void op::util::UnaryElementwiseArithmetic::validate_and_infer_elementwise_arithmetic()
{
    const auto& element_type = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          element_type.is_dynamic() || element_type != element::boolean,
                          "Arguments cannot have boolean element type (argument element type: ",
                          element_type,
                          ").");
    set_output_type(0, element_type, get_input_partial_shape(0));
}

void op::util::UnaryElementwiseArithmetic::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(util_UnaryElementwiseArithmetic_validate_and_infer_types);
    validate_and_infer_elementwise_arithmetic();
}

bool op::util::UnaryElementwiseArithmetic::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(util_UnaryElementwiseArithmetic_visit_attributes);
    return true;
}
