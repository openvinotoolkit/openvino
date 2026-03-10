// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/log_softmax.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"

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
    if (const auto& rank = input_shape.rank(); rank.is_static()) {
        ov::util::validate_axis(m_axis, rank, *this);
    }

    set_output_type(0, get_input_element_type(0), input_shape);
}

std::shared_ptr<Node> op::v5::LogSoftmax::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v5_LogSoftmax_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v5::LogSoftmax>(new_args.at(0), m_axis);
}
}  // namespace ov
