// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/augru_cell.hpp"

#include <cmath>

#include "augru_cell_shape_inference.hpp"
#include "itt.hpp"

using namespace std;

ov::op::internal::AUGRUCell::AUGRUCell() : m_linear_before_reset(false) {
    m_activations = {"sigmoid", "tanh"};
    m_activation_f = get_activation_function(0);
    m_activation_g = get_activation_function(1);
}

ov::op::internal::AUGRUCell::AUGRUCell(const Output<Node>& X,
                                       const Output<Node>& H_t,
                                       const Output<Node>& W,
                                       const Output<Node>& R,
                                       const Output<Node>& B,
                                       const Output<Node>& A,
                                       size_t hidden_size)
    : RNNCellBase({X, H_t, W, R, B, A}, hidden_size, 0.f, std::vector<std::string>{"sigmoid", "tanh"}, {}, {}),
      m_activation_f{get_activation_function(0)},
      m_activation_g{get_activation_function(1)},
      m_linear_before_reset{false} {
    constructor_validate_and_infer_types();
}

bool ov::op::internal::AUGRUCell::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(internal_AUGRUCell_visit_attributes);
    visitor.on_attribute("linear_before_reset", m_linear_before_reset);
    return op::util::RNNCellBase::visit_attributes(visitor);
}

void ov::op::internal::AUGRUCell::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(internal_AUGRUCell_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this, m_clip == 0.f, "AUGRUCell doesn't support clip other than 0.");
    NODE_VALIDATION_CHECK(this,
                          m_activations.size() == 2 && m_activations[0] == "sigmoid" && m_activations[1] == "tanh",
                          "AUGRUCell supports only sigmoid for f and tanh for g activation functions.");
    NODE_VALIDATION_CHECK(this,
                          m_activations_alpha.empty() && m_activations_beta.empty(),
                          "AUGRUCell doesn't support activations_alpha and activations_beta.");
    NODE_VALIDATION_CHECK(this,
                          m_linear_before_reset == false,
                          "AUGRUCell supports only linear_before_reset equals false.");

    // Validate input types and save result for output type
    auto result_et = element::dynamic;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(2)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(3)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(4)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(5)),
                          "Element types for inputs do not match.");

    // Get input partial shape for all inputs
    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    std::vector<ov::PartialShape> output_shapes = shape_infer(this, input_shapes);

    // Set output type and shape
    set_output_type(0, result_et, output_shapes[0]);
}

shared_ptr<ov::Node> ov::op::internal::AUGRUCell::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(internal_AUGRUCell_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<AUGRUCell>(new_args.at(0),
                                  new_args.at(1),
                                  new_args.at(2),
                                  new_args.at(3),
                                  new_args.at(4),
                                  new_args.at(5),
                                  get_hidden_size());
}
