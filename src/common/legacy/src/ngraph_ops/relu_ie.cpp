// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/ngraph_ops/relu_ie.hpp"

#include <algorithm>
#include <memory>

#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::ReLUIE);

op::ReLUIE::ReLUIE(const Output<Node>& data, const float& negative_slope, const element::Type output_type)
    : Op(OutputVector {data}), m_negative_slope(negative_slope), m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::ReLUIE::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<ReLUIE>(new_args.at(0), m_negative_slope, m_output_type);
}

void op::ReLUIE::validate_and_infer_types() {
    set_output_type(
        0,
        m_output_type == element::undefined ? get_input_element_type(0) : m_output_type,
        get_input_partial_shape(0));
}

bool op::ReLUIE::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("negative_slope", m_negative_slope);
    return true;
}
