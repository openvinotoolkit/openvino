// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/ngraph_ops/power.hpp"

#include <memory>

#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::PowerIE);

op::PowerIE::PowerIE(const Output<ngraph::Node>& data_batch, const float power, const float scale, const float shift, const element::Type output_type)
    : Op({data_batch}), scale(scale), power(power), shift(shift), m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::PowerIE::clone_with_new_inputs(const OutputVector& new_args) const {
    if (new_args.size() != 1) {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<PowerIE>(new_args.at(0), this->power, this->scale, this->shift, this->m_output_type);
}

void op::PowerIE::validate_and_infer_types() {
    set_output_type(0, m_output_type == element::undefined ? get_input_element_type(0) : m_output_type, get_input_partial_shape(0));
}

bool op::PowerIE::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("scale", scale);
    visitor.on_attribute("power", power);
    visitor.on_attribute("shift", shift);
    return true;
}
