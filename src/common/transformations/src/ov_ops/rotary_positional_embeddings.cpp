// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/rotary_positional_embeddings.hpp"

#include "itt.hpp"

using namespace std;
using namespace ov::op::internal;

RPE::RPE(const Output<Node>& data, const Output<Node>& sin, const Output<Node>& cos, const int64_t& axis)
    : Op({data, sin, cos}),
      m_axis{axis} {
    constructor_validate_and_infer_types();
}

void RPE::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(internal_RoPE_validate_and_infer_types);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool RPE::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(internal_RoPE_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    return true;
}

shared_ptr<ov::Node> RPE::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(internal_RoPE_clone_with_new_inputs);
    return make_shared<RPE>(new_args.at(0), new_args.at(1), new_args.at(2), m_axis);
}
