// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/rnn_cell_ie.hpp"

#include <memory>
#include <string>
#include <vector>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::RNNCellIE::type_info;

op::RNNCellIE::RNNCellIE(const Output<Node>& X, const Output<Node>& H_t,
                           const Output<Node>& WR, const Output<Node>& B, std::size_t hidden_size,
                           const std::vector<std::string>& activations, const std::vector<float>& activations_alpha,
                           const std::vector<float>& activations_beta, float clip)
    : Op({X, H_t, WR, B}),
      m_hidden_size(hidden_size),
      m_activations(activations),
      m_activations_alpha(activations_alpha),
      m_activations_beta(activations_beta),
      m_clip(clip) {
    constructor_validate_and_infer_types();
}

void op::RNNCellIE::validate_and_infer_types() {
    element::Type arg_type = get_input_element_type(0);

    PartialShape output_shape{PartialShape::dynamic(2)};
    if (get_input_partial_shape(0).is_static()) {
        int64_t batch_size = get_input_partial_shape(0).get_shape()[0];
        output_shape = {batch_size, m_hidden_size};
    }
    set_output_type(0, arg_type, output_shape);
}

shared_ptr<Node> op::RNNCellIE::copy_with_new_args(const NodeVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<op::RNNCellIE>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3),
                                       m_hidden_size, m_activations, m_activations_alpha, m_activations_beta, m_clip);
}

bool op::RNNCellIE::visit_attributes(AttributeVisitor &visitor) {
    visitor.on_attribute("hidden_size", m_hidden_size);
    visitor.on_attribute("activations", m_activations);
    visitor.on_attribute("activations_alpha", m_activations_alpha);
    visitor.on_attribute("activations_beta", m_activations_beta);
    visitor.on_attribute("clip", m_clip);
    return true;
}