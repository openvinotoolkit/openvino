// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/rnn_cell.hpp"

#include <cmath>

#include "itt.hpp"
#include "openvino/core/type/element_type.hpp"
#include "rnn_cell_shape_inference.hpp"

namespace ov {

op::v0::RNNCell::RNNCell() {
    m_activations = {"tanh"};
    m_activation_f = get_activation_function(0);
}

op::v0::RNNCell::RNNCell(const Output<Node>& X,
                         const Output<Node>& initial_hidden_state,
                         const Output<Node>& W,
                         const Output<Node>& R,
                         size_t hidden_size,
                         const std::vector<std::string>& activations,
                         const std::vector<float>& activations_alpha,
                         const std::vector<float>& activations_beta,
                         float clip)
    : RNNCellBase({X, initial_hidden_state, W, R}, hidden_size, clip, activations, activations_alpha, activations_beta),
      m_activation_f{get_activation_function(0)} {
    set_argument(4, get_default_bias_input());
    constructor_validate_and_infer_types();
}

op::v0::RNNCell::RNNCell(const Output<Node>& X,
                         const Output<Node>& initial_hidden_state,
                         const Output<Node>& W,
                         const Output<Node>& R,
                         const Output<Node>& B,
                         size_t hidden_size,
                         const std::vector<std::string>& activations,
                         const std::vector<float>& activations_alpha,
                         const std::vector<float>& activations_beta,
                         float clip)
    : RNNCellBase({X, initial_hidden_state, W, R, B},
                  hidden_size,
                  clip,
                  activations,
                  activations_alpha,
                  activations_beta),
      m_activation_f{get_activation_function(0)} {
    constructor_validate_and_infer_types();
}

bool op::v0::RNNCell::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_RNNCell_visit_attributes);
    return op::util::RNNCellBase::visit_attributes(visitor);
}

void op::v0::RNNCell::validate_and_infer_types() {
    OV_OP_SCOPE(v0_RNNCell_validate_and_infer_types);
    auto result_et = element::dynamic;

    // Validate input types and save result for output type
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(2)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(3)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(4)),
                          "Element types for X, initial_hidden_state, W, R and B inputs do not match.");

    // Mark inputs which are relevant to output parameters
    for (size_t i = 0; i <= 4; ++i)
        set_input_is_relevant_to_shape(i);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, result_et, output_shapes[0]);
}

Output<Node> op::v0::RNNCell::get_default_bias_input() const {
    return Output<Node>{op::v0::Constant::create(get_input_element_type(0),
                                                 ov::Shape{s_gates_count * get_hidden_size()},
                                                 std::vector<float>(s_gates_count * get_hidden_size(), 0.f))};
}

std::shared_ptr<Node> op::v0::RNNCell::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_RNNCell_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 4) {
        return std::make_shared<RNNCell>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         new_args.at(3),
                                         get_hidden_size(),
                                         get_activations(),
                                         get_activations_alpha(),
                                         get_activations_beta(),
                                         get_clip());
    } else if (new_args.size() == 5) {
        return std::make_shared<RNNCell>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         new_args.at(3),
                                         new_args.at(4),
                                         get_hidden_size(),
                                         get_activations(),
                                         get_activations_alpha(),
                                         get_activations_beta(),
                                         get_clip());
    } else {
        OPENVINO_THROW("Incorrect number of new arguments");
    }
}
}  // namespace ov
