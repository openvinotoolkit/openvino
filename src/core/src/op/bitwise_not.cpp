// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/bitwise_not.hpp"

#include "itt.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v13 {
BitwiseNot::BitwiseNot(const Output<Node>& arg) : op::Op({arg}) {
    constructor_validate_and_infer_types();
}
void BitwiseNot::validate_and_infer_types() {
    OV_OP_SCOPE(v13_BitwiseNot_validate_and_infer_types);
    const auto& element_type = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          element_type.is_dynamic() || element_type.is_integral(),
                          "The element type of the input tensor must be integer or boolean.");
    set_output_type(0, element_type, get_input_partial_shape(0));
}

std::shared_ptr<Node> BitwiseNot::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v13_BitwiseNot_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BitwiseNot>(new_args.at(0));
}

}  // namespace v13
}  // namespace op
}  // namespace ov
