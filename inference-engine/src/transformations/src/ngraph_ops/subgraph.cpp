// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/subgraph.hpp"

#include <algorithm>
#include <memory>

#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Subgraph::type_info;

op::Subgraph::Subgraph(const OutputVector& args, std::shared_ptr<Function> body)
    : Op(args), m_body(body) {
    constructor_validate_and_infer_types();
}

op::Subgraph::Subgraph(const NodeVector& args, std::shared_ptr<Function> body)
    : Subgraph(as_output_vector(args), body) {}

std::shared_ptr<Node> op::Subgraph::copy_with_new_args(const NodeVector& new_args) const {
    return make_shared<Subgraph>(new_args, m_body);
}

void op::Subgraph::validate_and_infer_types() {
    // Go over all inputs in the node and replace parameters in m_body with new shape/type
    // FIXME: Check if shape/type is changed before replacement?
    for (size_t i = 0; i < get_input_size(); ++i) {
        m_body->replace_parameter(i, std::make_shared<Parameter>(get_input_element_type(i), get_input_partial_shape(i)));
    }

    m_body->validate_nodes_and_infer_types();

    // Go over all outputs and update shape/type from m_body
    set_output_size(m_body->get_output_size());
    for (size_t i = 0; i < get_output_size(); ++i) {
        set_output_type(0, m_body->get_output_element_type(i), m_body->get_output_partial_shape(i));
    }
}

bool op::Subgraph::visit_attributes(AttributeVisitor& visitor) {
    return true;
}