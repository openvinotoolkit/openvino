// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gru_cell.hpp"

#include <cmath>

#include "gru_cell_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {

op::v3::GRUCell::GRUCell() : m_linear_before_reset(false) {
    m_activations = {"sigmoid", "tanh"};
    m_activation_f = get_activation_function(0);
    m_activation_g = get_activation_function(1);
}

op::v3::GRUCell::GRUCell(const Output<Node>& X,
                         const Output<Node>& initial_hidden_state,
                         const Output<Node>& W,
                         const Output<Node>& R,
                         size_t hidden_size)
    : GRUCell(X,
              initial_hidden_state,
              W,
              R,
              hidden_size,
              std::vector<std::string>{"sigmoid", "tanh"},
              std::vector<float>{},
              std::vector<float>{},
              0.f,
              false) {}

op::v3::GRUCell::GRUCell(const Output<Node>& X,
                         const Output<Node>& initial_hidden_state,
                         const Output<Node>& W,
                         const Output<Node>& R,
                         size_t hidden_size,
                         const std::vector<std::string>& activations,
                         const std::vector<float>& activations_alpha,
                         const std::vector<float>& activations_beta,
                         float clip,
                         bool linear_before_reset)
    : RNNCellBase({X, initial_hidden_state, W, R}, hidden_size, clip, activations, activations_alpha, activations_beta),
      m_activation_f{get_activation_function(0)},
      m_activation_g{get_activation_function(1)},
      m_linear_before_reset{linear_before_reset} {
    add_default_bias_input();
    constructor_validate_and_infer_types();
}

op::v3::GRUCell::GRUCell(const Output<Node>& X,
                         const Output<Node>& initial_hidden_state,
                         const Output<Node>& W,
                         const Output<Node>& R,
                         const Output<Node>& B,
                         size_t hidden_size,
                         const std::vector<std::string>& activations,
                         const std::vector<float>& activations_alpha,
                         const std::vector<float>& activations_beta,
                         float clip,
                         bool linear_before_reset)
    : RNNCellBase({X, initial_hidden_state, W, R, B},
                  hidden_size,
                  clip,
                  activations,
                  activations_alpha,
                  activations_beta),
      m_activation_f{get_activation_function(0)},
      m_activation_g{get_activation_function(1)},
      m_linear_before_reset{linear_before_reset} {
    constructor_validate_and_infer_types();
}

bool op::v3::GRUCell::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v3_GRUCell_visit_attributes);
    visitor.on_attribute("linear_before_reset", m_linear_before_reset);
    return op::util::RNNCellBase::visit_attributes(visitor);
}

void op::v3::GRUCell::validate_and_infer_types() {
    OV_OP_SCOPE(v3_GRUCell_validate_and_infer_types);

    // Validate input types and save result for output type
    auto result_et = element::dynamic;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(3)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(4)),
                          "Element types for X, initial_hidden_state, W, R and B inputs do not "
                          "match.");

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, result_et, output_shapes[0]);
}

void op::v3::GRUCell::add_default_bias_input() {
    Output<Node> B =
        op::v0::Constant::create(get_input_element_type(0),
                                 ov::Shape{(s_gates_count + m_linear_before_reset) * get_hidden_size()},
                                 std::vector<float>((s_gates_count + m_linear_before_reset) * get_hidden_size(), 0.f));
    set_argument(4, B);
}

std::shared_ptr<Node> op::v3::GRUCell::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_GRUCell_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 4) {
        return std::make_shared<GRUCell>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         new_args.at(3),
                                         get_hidden_size(),
                                         get_activations(),
                                         get_activations_alpha(),
                                         get_activations_beta(),
                                         get_clip(),
                                         m_linear_before_reset);
    } else if (new_args.size() == 5) {
        return std::make_shared<GRUCell>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         new_args.at(3),
                                         new_args.at(4),
                                         get_hidden_size(),
                                         get_activations(),
                                         get_activations_alpha(),
                                         get_activations_beta(),
                                         get_clip(),
                                         m_linear_before_reset);
    } else {
        OPENVINO_THROW("Incorrect number of new arguments");
    }
}
}  // namespace ov
