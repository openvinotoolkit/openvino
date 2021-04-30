// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/selu.hpp"
#include "itt.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v0::Selu, "Selu", 0);

op::v0::Selu::Selu(const Output<Node>& data, const Output<Node>& alpha, const Output<Node>& lambda)
    : Op({data, alpha, lambda})
{
    constructor_validate_and_infer_types();
}

void op::v0::Selu::validate_and_infer_types()

{
    NGRAPH_OP_SCOPE(v0_Selu_validate_and_infer_types);
    auto data_et = get_input_element_type(0);
    auto alpha_et = get_input_element_type(1);
    auto lambda_et = get_input_element_type(2);

    // set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    NODE_VALIDATION_CHECK(
        this,
        data_et.is_real() && alpha_et.is_real() && lambda_et.is_real(),
        "The data type for input, alpha and lambda is expected to be a floating point type. Got data: ",
        data_et,
        ", alpha: ",
        alpha_et,
        ", lambda: ",
        lambda_et);

    NODE_VALIDATION_CHECK(
        this,
        data_et == alpha_et && alpha_et == lambda_et,
        "Type of data (inputs), alpha and lambda is expected to be the same. Got: data: ",
        data_et,
        ", alpha: ",
        alpha_et,
        ", lambda: ",
        lambda_et);
    
    set_output_type(0, data_et, get_input_partial_shape(0));
}

bool op::v0::Selu::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_Selu_visit_attributes);
    return true;
}

shared_ptr<Node> op::v0::Selu::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_Selu_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v0::Selu>(new_args.at(0), new_args.at(1), new_args.at(2));
}
