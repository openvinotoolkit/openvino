// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/roll.hpp"

#include <ngraph/validation_util.hpp>
#include <roll_shape_inference.hpp>

#include "itt.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v7::Roll);

op::v7::Roll::Roll(const Output<Node>& data, const Output<Node>& shift, const Output<Node>& axes)
    : Op({data, shift, axes}) {
    constructor_validate_and_infer_types();
}

void op::v7::Roll::validate_and_infer_types() {
    OV_OP_SCOPE(v7_Roll_validate_and_infer_types);

    const auto& shift_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          shift_et.is_dynamic() || shift_et == element::i32 || shift_et == element::i64,
                          "Shift must have int32 or int64 element type.");

    const auto& axes_et = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          axes_et.is_dynamic() || axes_et == element::i32 || axes_et == element::i64,
                          "Axes must have int32 or int64 element type.");

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    const std::vector<ov::PartialShape> input_shapes = {get_input_partial_shape(0),
                                                        get_input_partial_shape(1),
                                                        get_input_partial_shape(2)};
    shape_infer(this, input_shapes, output_shapes);

    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

bool op::v7::Roll::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v7_Roll_visit_attributes);
    return true;
}

shared_ptr<Node> op::v7::Roll::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v7_Roll_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v7::Roll>(new_args[0], new_args[1], new_args[2]);
}
