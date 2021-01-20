// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/ngraph_ops/swish_ie.hpp"

#include <algorithm>
#include <memory>

#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"
#include "../itt.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::SwishIE::type_info;

op::SwishIE::SwishIE(const Output<Node> & input, const float alpha)
        : Op({input}), m_alpha(alpha) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::SwishIE::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(SwishIE_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<SwishIE>(new_args.at(0), m_alpha);
}

bool op::SwishIE::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(SwishIE_visit_attributes);
    visitor.on_attribute("alpha", m_alpha);
    return true;
}

void op::SwishIE::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(SwishIE_validate_and_infer_types);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

void op::SwishIE::set_alpha(float alpha)  {
    INTERNAL_OP_SCOPE(SwishIE_set_alpha);
    m_alpha = alpha;
}

float op::SwishIE::get_alpha() const {
    INTERNAL_OP_SCOPE(SwishIE_get_alpha);
    return m_alpha;
}

