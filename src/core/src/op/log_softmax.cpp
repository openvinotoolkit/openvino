// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/log_softmax.hpp"

#include "itt.hpp"

namespace ov {

op::v5::LogSoftmax::LogSoftmax(const Output<Node>& arg, const int64_t axis) : Op({arg}), m_axis(axis) {
    constructor_validate_and_infer_types();
}

bool op::v5::LogSoftmax::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v5_LogSoftmax_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    return true;
}

void op::v5::LogSoftmax::validate_and_infer_types() {
    OV_OP_SCOPE(v5_LogSoftmax_validate_and_infer_types);
    const ov::PartialShape& input_shape = get_input_partial_shape(0);
    if (input_shape.rank().is_static())
        NODE_VALIDATION_CHECK(this,
                              m_axis < input_shape.rank().get_length() && m_axis >= -input_shape.rank().get_length(),
                              "Reduction axis (",
                              m_axis,
                              ") is out of bounds (argument shape: ",
                              input_shape,
                              ").");

    set_output_type(0, get_input_element_type(0), input_shape);
}

std::shared_ptr<Node> op::v5::LogSoftmax::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v5_LogSoftmax_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v5::LogSoftmax>(new_args.at(0), m_axis);
}
}  // namespace ov
