// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>

#include "legacy/ngraph_ops/hard_sigmoid_ie.hpp"

#include "ngraph/op/hard_sigmoid.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/multiply.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::HardSigmoid_IE::type_info;

op::HardSigmoid_IE::HardSigmoid_IE(const ngraph::Output<ngraph::Node> &arg,
                   float alpha,
                   float beta)
        : Op({arg})
        , m_alpha(alpha)
        , m_beta(beta) {
    constructor_validate_and_infer_types();
}

void op::HardSigmoid_IE::validate_and_infer_types() {
    element::Type arg_type = get_input_element_type(0);
    PartialShape arg_shape = get_input_partial_shape(0);
    set_output_type(0, arg_type, arg_shape);
}

shared_ptr<Node> op::HardSigmoid_IE::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<op::HardSigmoid_IE>(new_args.at(0), m_alpha, m_beta);
}

bool op::HardSigmoid_IE::visit_attributes(AttributeVisitor& visitor) {
    return true;
}
