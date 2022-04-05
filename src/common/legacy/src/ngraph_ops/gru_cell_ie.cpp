// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/ngraph_ops/gru_cell_ie.hpp"

#include <memory>
#include <string>
#include <vector>

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::GRUCellIE);

op::GRUCellIE::GRUCellIE(const Output<Node>& X, const Output<Node>& H_t,
                           const Output<Node>& WR, const Output<Node>& B, std::size_t hidden_size,
                           const std::vector<std::string>& activations, const std::vector<float>& activations_alpha,
                           const std::vector<float>& activations_beta, float clip, bool linear_before_reset)
    : Op({X, H_t, WR, B}),
      m_hidden_size(hidden_size),
      m_activations(activations),
      m_activations_alpha(activations_alpha),
      m_activations_beta(activations_beta),
      m_clip(clip),
      m_linear_before_reset(linear_before_reset) {
    constructor_validate_and_infer_types();
}

void op::GRUCellIE::validate_and_infer_types() {
    element::Type arg_type = get_input_element_type(0);

    PartialShape output_shape{PartialShape::dynamic(2)};

    if (get_input_partial_shape(0).is_static()) {
        int64_t batch_size = get_input_partial_shape(0).get_shape()[0];
        output_shape = {batch_size, m_hidden_size};
    }

    set_output_type(0, arg_type, output_shape);
}

shared_ptr<Node> op::GRUCellIE::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<op::GRUCellIE>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3),
                                      m_hidden_size, m_activations, m_activations_alpha, m_activations_beta, m_clip,
                                      m_linear_before_reset);
}

bool op::GRUCellIE::visit_attributes(AttributeVisitor &visitor) {
    visitor.on_attribute("hidden_size", m_hidden_size);
    visitor.on_attribute("activations", m_activations);
    visitor.on_attribute("activations_alpha", m_activations_alpha);
    visitor.on_attribute("activations_beta", m_activations_beta);
    visitor.on_attribute("clip", m_clip);
    visitor.on_attribute("linear_before_reset", m_linear_before_reset);
    return true;
}
