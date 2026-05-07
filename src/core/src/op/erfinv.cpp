// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/erfinv.hpp"

#include "itt.hpp"

namespace ov::op::v17 {

ErfInv::ErfInv(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

void ErfInv::validate_and_infer_types() {
    OV_OP_SCOPE(v17_ErfInv_validate_and_infer_types);
    const auto& input_et = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_et.is_dynamic() || input_et.is_real(),
                          "Input element type must be floating-point, instead got: ",
                          input_et);

    UnaryElementwiseArithmetic::validate_and_infer_types();
}

std::shared_ptr<Node> ErfInv::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v17_ErfInv_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ErfInv>(new_args.at(0));
}

}  // namespace ov::op::v17
