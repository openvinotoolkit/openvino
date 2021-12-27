// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/elu.hpp"

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;

BWDCMP_RTTI_DEFINITION(ov::op::v1::Elu);

ov::op::v1::Elu::Elu(const Output<Node>& data, const double alpha) : Op({data}), m_alpha{alpha} {
    constructor_validate_and_infer_types();
}

bool ov::op::v1::Elu::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v1_Elu_visit_attributes);
    visitor.on_attribute("alpha", m_alpha);
    return true;
}

void ov::op::v1::Elu::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v1_Elu_validate_and_infer_types);
    set_output_size(1);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<ov::Node> ov::op::v1::Elu::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v1_Elu_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Elu>(new_args.at(0), m_alpha);
}
