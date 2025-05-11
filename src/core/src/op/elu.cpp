// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/elu.hpp"

#include "itt.hpp"

ov::op::v0::Elu::Elu(const Output<Node>& data, const double alpha)
    : util::UnaryElementwiseArithmetic(data),
      m_alpha{alpha} {
    constructor_validate_and_infer_types();
}

bool ov::op::v0::Elu::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Elu_visit_attributes);
    visitor.on_attribute("alpha", m_alpha);
    return true;
}

void ov::op::v0::Elu::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Elu_validate_and_infer_types);
    set_output_size(1);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<ov::Node> ov::op::v0::Elu::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Elu_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Elu>(new_args.at(0), m_alpha);
}
