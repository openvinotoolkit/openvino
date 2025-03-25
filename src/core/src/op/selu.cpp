// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/selu.hpp"

#include "itt.hpp"

namespace ov {

op::v0::Selu::Selu(const Output<Node>& data, const Output<Node>& alpha, const Output<Node>& lambda)
    : Op({data, alpha, lambda}) {
    constructor_validate_and_infer_types();
}

void op::v0::Selu::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Selu_validate_and_infer_types);
    auto data_et = get_input_element_type(0);
    auto alpha_et = get_input_element_type(1);
    auto lambda_et = get_input_element_type(2);
    auto result_et = element::dynamic;

    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, result_et, data_et) &&
                              element::Type::merge(result_et, result_et, alpha_et) &&
                              element::Type::merge(result_et, result_et, lambda_et),
                          "Input element types do not match : ",
                          data_et,
                          " and ",
                          alpha_et,
                          " and ",
                          lambda_et);

    NODE_VALIDATION_CHECK(this,
                          result_et.is_dynamic() || result_et.is_real(),
                          "Input element types must be floating-point. Got: ",
                          result_et);

    set_output_type(0, result_et, get_input_partial_shape(0));
}

std::shared_ptr<Node> op::v0::Selu::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Selu_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v0::Selu>(new_args.at(0), new_args.at(1), new_args.at(2));
}
}  // namespace ov
