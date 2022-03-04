// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/ngraph_ops/swish_ie.hpp"

#include <algorithm>
#include <memory>

#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::SwishIE);

op::SwishIE::SwishIE(const Output<Node> & input, const float alpha)
        : Op({input}), m_alpha(alpha) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::SwishIE::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<SwishIE>(new_args.at(0), m_alpha);
}

bool op::SwishIE::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("alpha", m_alpha);
    return true;
}

void op::SwishIE::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

void op::SwishIE::set_alpha(float alpha)  {
    m_alpha = alpha;
}

float op::SwishIE::get_alpha() const {
    return m_alpha;
}

