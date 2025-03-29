// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/roll.hpp"

#include "itt.hpp"
#include "roll_shape_inference.hpp"

namespace ov {
namespace op {
namespace v7 {

Roll::Roll(const Output<Node>& data, const Output<Node>& shift, const Output<Node>& axes) : Op({data, shift, axes}) {
    constructor_validate_and_infer_types();
}

void Roll::validate_and_infer_types() {
    OV_OP_SCOPE(v7_Roll_validate_and_infer_types);

    const auto& shift_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          shift_et.is_dynamic() || shift_et == element::i32 || shift_et == element::i64,
                          "Shift must have int32 or int64 element type.");

    const auto& axes_et = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          axes_et.is_dynamic() || axes_et == element::i32 || axes_et == element::i64,
                          "Axes must have int32 or int64 element type.");

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

bool Roll::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v7_Roll_visit_attributes);
    return true;
}

std::shared_ptr<Node> Roll::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v7_Roll_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Roll>(new_args[0], new_args[1], new_args[2]);
}

}  // namespace v7
}  // namespace op
}  // namespace ov
