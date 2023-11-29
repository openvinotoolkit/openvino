// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "snippets/op/reduce.hpp"


namespace ov {
namespace snippets {
namespace op {

ReduceBase::ReduceBase(const Output<Node>& x, size_t axis) : Op({x}), m_axis(axis) {
    constructor_validate_and_infer_types();
}

bool ReduceBase::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("axis", m_axis);
    return true;
}

void ReduceBase::validate_and_infer_types() {
    auto result_shape = get_input_partial_shape(0);
    result_shape[m_axis] = 1;
    set_output_type(0, get_input_element_type(0), result_shape);
}

std::shared_ptr<Node> ReduceSum::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(ReduceSum);
    check_new_args_count(this, new_args);
    return std::make_shared<ReduceSum>(new_args.at(0), m_axis);
}

std::shared_ptr<Node> ReduceMax::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(ReduceMax);
    check_new_args_count(this, new_args);
    return std::make_shared<ReduceMax>(new_args.at(0), m_axis);
}

} // namespace op
} // namespace snippets
} // namespace ov
