// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/lstm_sequence.hpp"

#include "itt.hpp"
#include "lstm_sequence_shape_inference.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/op/util/recurrent_sequence.hpp"

namespace ov {
bool op::v5::LSTMSequence::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v5_LSTMSequence_visit_attributes);
    visitor.on_attribute("direction", m_direction);
    return op::util::RNNCellBase::visit_attributes(visitor);
}

std::shared_ptr<Node> op::v5::LSTMSequence::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v5_LSTMSequence_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 7) {
        return std::make_shared<op::v5::LSTMSequence>(new_args.at(0),  // X
                                                      new_args.at(1),  // initial_hidden_state
                                                      new_args.at(2),  // initial_cell_state
                                                      new_args.at(3),  // sequence_lengths
                                                      new_args.at(4),  // W
                                                      new_args.at(5),  // R
                                                      new_args.at(6),  // B
                                                      m_hidden_size,
                                                      m_direction,
                                                      m_activations_alpha,
                                                      m_activations_beta,
                                                      m_activations,
                                                      m_clip);
    } else {
        OPENVINO_THROW("Incorrect number of new arguments");
    }
}

void op::v5::LSTMSequence::validate_and_infer_types() {
    OV_OP_SCOPE(v5_LSTMSequence_validate_and_infer_types);

    auto result_et = element::dynamic;

    // Validate input types and save result for output type
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(2)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(4)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(5)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(6)),
                          "Element types for X, initial_hidden_state, initial_cell_state, W, R and B inputs do "
                          "not match.");

    // Mark inputs which are relevant to output parameters
    for (size_t i = 0; i <= 6; ++i)
        set_input_is_relevant_to_shape(i);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    auto output_shapes = shape_infer(this, input_shapes);

    // Set output size, type and shape
    set_output_type(0, result_et, output_shapes[0]);
    set_output_type(1, result_et, output_shapes[1]);
    set_output_type(2, result_et, output_shapes[2]);
}
}  // namespace ov
