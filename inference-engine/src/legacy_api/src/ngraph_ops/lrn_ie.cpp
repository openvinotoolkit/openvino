// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/ngraph_ops/lrn_ie.hpp"

#include <memory>
#include <string>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/multiply.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::LRN_IE::type_info;

op::LRN_IE::LRN_IE(const ngraph::Output<ngraph::Node>& arg, double alpha, double beta, double bias, size_t size,
                   std::string region)
    : Op({arg}), m_alpha(alpha), m_beta(beta), m_bias(bias), m_size(size), m_region(region) {
    constructor_validate_and_infer_types();
}

void op::LRN_IE::validate_and_infer_types() {
    element::Type arg_type = get_input_element_type(0);
    PartialShape arg_shape = get_input_partial_shape(0);
    set_output_type(0, arg_type, arg_shape);
}

shared_ptr<Node> op::LRN_IE::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<op::LRN_IE>(new_args.at(0), m_alpha, m_beta, m_bias, m_size, m_region);
}

bool op::LRN_IE::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("alpha", m_alpha);
    visitor.on_attribute("beta", m_beta);
    visitor.on_attribute("k", m_bias);
    visitor.on_attribute("local-size", m_size);
    visitor.on_attribute("region", m_region);
    return true;
}
