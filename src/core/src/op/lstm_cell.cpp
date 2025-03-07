// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/lstm_cell.hpp"

#include <cmath>
#include <functional>

#include "itt.hpp"
#include "lstm_cell_shape_inference.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"

namespace ov {

op::v0::LSTMCell::LSTMCell() : m_input_forget(false), m_weights_format(LSTMWeightsFormat::IFCO) {
    m_activations = {"sigmoid", "tanh", "tanh"};
    m_activation_f = get_activation_function(0);
    m_activation_g = get_activation_function(1);
    m_activation_h = get_activation_function(2);
}

op::v0::LSTMCell::LSTMCell(const Output<Node>& X,
                           const Output<Node>& initial_hidden_state,
                           const Output<Node>& initial_cell_state,
                           const Output<Node>& W,
                           const Output<Node>& R,
                           size_t hidden_size,
                           op::LSTMWeightsFormat weights_format,
                           const std::vector<std::string>& activations,
                           const std::vector<float>& activations_alpha,
                           const std::vector<float>& activations_beta,
                           float clip,
                           bool input_forget)
    : RNNCellBase({X, initial_hidden_state, initial_cell_state, W, R},
                  hidden_size,
                  clip,
                  activations,
                  activations_alpha,
                  activations_beta),
      m_activation_f{get_activation_function(0)},
      m_activation_g{get_activation_function(1)},
      m_activation_h{get_activation_function(2)},
      m_input_forget{input_forget},
      m_weights_format{weights_format} {
    set_argument(5, get_default_bias_input());
    set_argument(6, get_default_peepholes_input());
    constructor_validate_and_infer_types();
}

op::v0::LSTMCell::LSTMCell(const Output<Node>& X,
                           const Output<Node>& initial_hidden_state,
                           const Output<Node>& initial_cell_state,
                           const Output<Node>& W,
                           const Output<Node>& R,
                           const Output<Node>& B,
                           size_t hidden_size,
                           op::LSTMWeightsFormat weights_format,
                           const std::vector<std::string>& activations,
                           const std::vector<float>& activations_alpha,
                           const std::vector<float>& activations_beta,
                           float clip,
                           bool input_forget)
    : RNNCellBase({X, initial_hidden_state, initial_cell_state, W, R, B},
                  hidden_size,
                  clip,
                  activations,
                  activations_alpha,
                  activations_beta),
      m_activation_f{get_activation_function(0)},
      m_activation_g{get_activation_function(1)},
      m_activation_h{get_activation_function(2)},
      m_input_forget{input_forget},
      m_weights_format{weights_format} {
    set_argument(6, get_default_peepholes_input());
    constructor_validate_and_infer_types();
}

op::v0::LSTMCell::LSTMCell(const Output<Node>& X,
                           const Output<Node>& initial_hidden_state,
                           const Output<Node>& initial_cell_state,
                           const Output<Node>& W,
                           const Output<Node>& R,
                           const Output<Node>& B,
                           const Output<Node>& P,
                           size_t hidden_size,
                           op::LSTMWeightsFormat weights_format,
                           const std::vector<std::string>& activations,
                           const std::vector<float>& activations_alpha,
                           const std::vector<float>& activations_beta,
                           float clip,
                           bool input_forget)
    : RNNCellBase({X, initial_hidden_state, initial_cell_state, W, R, B, P},
                  hidden_size,
                  clip,
                  activations,
                  activations_alpha,
                  activations_beta),
      m_activation_f{get_activation_function(0)},
      m_activation_g{get_activation_function(1)},
      m_activation_h{get_activation_function(2)},
      m_input_forget{input_forget},
      m_weights_format{weights_format} {
    constructor_validate_and_infer_types();
}

bool op::v0::LSTMCell::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_LSTMCell_visit_attributes);
    visitor.on_attribute("hidden_size", m_hidden_size);
    visitor.on_attribute("activations", m_activations);
    visitor.on_attribute("activations_alpha", m_activations_alpha);
    visitor.on_attribute("activations_beta", m_activations_beta);
    visitor.on_attribute("clip", m_clip);

    visitor.on_attribute("input_forget", m_input_forget);
    visitor.on_attribute("weights_format", m_weights_format);
    return true;
}

void op::v0::LSTMCell::validate_and_infer_types() {
    OV_OP_SCOPE(v0_LSTMCell_validate_and_infer_types);

    // There should be 7 inputs, if no, it's possible the op can be fixed by
    // generating default ones for input 6 and 7 (bias input, peepholes)

    // missing both bias_input and peepholes
    if (get_input_size() == 5) {
        set_argument(5, get_default_bias_input());
    }
    // missing peepholes
    if (get_input_size() == 6) {
        set_argument(6, get_default_peepholes_input());
    }

    auto result_et = element::dynamic;
    // Get input partial shape for all inputs
    const auto& x_pshape = get_input_partial_shape(0);
    const auto& ht_pshape = get_input_partial_shape(1);
    const auto& ct_pshape = get_input_partial_shape(2);
    const auto& w_pshape = get_input_partial_shape(3);
    const auto& r_pshape = get_input_partial_shape(4);
    const auto& b_pshape = get_input_partial_shape(5);
    const auto& p_pshape = get_input_partial_shape(6);

    // Validate input element types and save result for output type
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(2)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(3)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(4)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(5)),
                          "Element types for X, initial_hidden_state, initial_cell_state, W, R and B do not "
                          "match.");

    std::vector<ov::PartialShape> input_shapes =
        {x_pshape, ht_pshape, ct_pshape, w_pshape, r_pshape, b_pshape, p_pshape};
    std::vector<ov::PartialShape> output_shapes = shape_infer(this, input_shapes);
    // Mark inputs which are relevant to output parameters
    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);
    set_input_is_relevant_to_shape(4);

    // Set output size, type and shape
    set_output_size(2);
    set_output_type(0, result_et, output_shapes[0]);
    set_output_type(1, result_et, output_shapes[1]);
}

Output<Node> op::v0::LSTMCell::get_default_bias_input() const {
    return Output<Node>{op::v0::Constant::create(get_input_element_type(0),
                                                 Shape{lstm_cell::gates_count * get_hidden_size()},
                                                 std::vector<float>{0.f})};
}

Output<Node> op::v0::LSTMCell::get_default_peepholes_input() const {
    return Output<Node>{op::v0::Constant::create(get_input_element_type(0),
                                                 Shape{lstm_cell::peepholes_count * get_hidden_size()},
                                                 std::vector<float>{0.f})};
}

std::shared_ptr<Node> op::v0::LSTMCell::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_LSTMCell_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 5) {
        return std::make_shared<op::v0::LSTMCell>(new_args.at(0),
                                                  new_args.at(1),
                                                  new_args.at(2),
                                                  new_args.at(3),
                                                  new_args.at(4),
                                                  get_hidden_size(),
                                                  get_weights_format(),
                                                  get_activations(),
                                                  get_activations_alpha(),
                                                  get_activations_beta(),
                                                  get_clip(),
                                                  m_input_forget);
    } else if (new_args.size() == 6) {
        return std::make_shared<op::v0::LSTMCell>(new_args.at(0),
                                                  new_args.at(1),
                                                  new_args.at(2),
                                                  new_args.at(3),
                                                  new_args.at(4),
                                                  new_args.at(5),
                                                  get_hidden_size(),
                                                  get_weights_format(),
                                                  get_activations(),
                                                  get_activations_alpha(),
                                                  get_activations_beta(),
                                                  get_clip(),
                                                  m_input_forget);
    } else if (new_args.size() == 7) {
        return std::make_shared<op::v0::LSTMCell>(new_args.at(0),
                                                  new_args.at(1),
                                                  new_args.at(2),
                                                  new_args.at(3),
                                                  new_args.at(4),
                                                  new_args.at(5),
                                                  new_args.at(6),
                                                  get_hidden_size(),
                                                  get_weights_format(),
                                                  get_activations(),
                                                  get_activations_alpha(),
                                                  get_activations_beta(),
                                                  get_clip(),
                                                  m_input_forget);
    } else {
        OPENVINO_THROW("Incorrect number of new arguments");
    }
}

AttributeAdapter<op::LSTMWeightsFormat>::~AttributeAdapter() = default;

template <>
OPENVINO_API EnumNames<op::LSTMWeightsFormat>& EnumNames<op::LSTMWeightsFormat>::get() {
    static auto enum_names = EnumNames<op::LSTMWeightsFormat>("op::LSTMWeightsFormat",
                                                              {{"fico", op::LSTMWeightsFormat::FICO},
                                                               {"icof", op::LSTMWeightsFormat::ICOF},
                                                               {"ifco", op::LSTMWeightsFormat::IFCO},
                                                               {"ifoc", op::LSTMWeightsFormat::IFOC},
                                                               {"iofc", op::LSTMWeightsFormat::IOFC}});
    return enum_names;
}

ov::op::util::LSTMWeightsFormat op::convert_lstm_weights_enums(op::LSTMWeightsFormat format) {
    switch (format) {
    case LSTMWeightsFormat::FICO:
        return ov::op::util::LSTMWeightsFormat::FICO;
    case LSTMWeightsFormat::ICOF:
        return ov::op::util::LSTMWeightsFormat::ICOF;
    case LSTMWeightsFormat::IFCO:
        return ov::op::util::LSTMWeightsFormat::IFCO;
    case LSTMWeightsFormat::IFOC:
        return ov::op::util::LSTMWeightsFormat::IFOC;
    case LSTMWeightsFormat::IOFC:
        return ov::op::util::LSTMWeightsFormat::IOFC;
    default:
        OPENVINO_ASSERT(false, "Incorrect LSTM weights format");
    }
}

std::ostream& operator<<(std::ostream& s, const op::LSTMWeightsFormat& type) {
    return s << as_string(type);
}

op::v4::LSTMCell::LSTMCell() {
    m_activations = {"sigmoid", "tanh", "tanh"};
    m_activation_f = get_activation_function(0);
    m_activation_g = get_activation_function(1);
    m_activation_h = get_activation_function(2);
}

op::v4::LSTMCell::LSTMCell(const Output<Node>& X,
                           const Output<Node>& initial_hidden_state,
                           const Output<Node>& initial_cell_state,
                           const Output<Node>& W,
                           const Output<Node>& R,
                           size_t hidden_size,
                           const std::vector<std::string>& activations,
                           const std::vector<float>& activations_alpha,
                           const std::vector<float>& activations_beta,
                           float clip)
    : RNNCellBase({X, initial_hidden_state, initial_cell_state, W, R},
                  hidden_size,
                  clip,
                  activations,
                  activations_alpha,
                  activations_beta),
      m_activation_f{get_activation_function(0)},
      m_activation_g{get_activation_function(1)},
      m_activation_h{get_activation_function(2)} {
    set_argument(5, get_default_bias_input());
    constructor_validate_and_infer_types();
}

op::v4::LSTMCell::LSTMCell(const Output<Node>& X,
                           const Output<Node>& initial_hidden_state,
                           const Output<Node>& initial_cell_state,
                           const Output<Node>& W,
                           const Output<Node>& R,
                           const Output<Node>& B,
                           size_t hidden_size,
                           const std::vector<std::string>& activations,
                           const std::vector<float>& activations_alpha,
                           const std::vector<float>& activations_beta,
                           float clip)
    : RNNCellBase({X, initial_hidden_state, initial_cell_state, W, R, B},
                  hidden_size,
                  clip,
                  activations,
                  activations_alpha,
                  activations_beta),
      m_activation_f{get_activation_function(0)},
      m_activation_g{get_activation_function(1)},
      m_activation_h{get_activation_function(2)} {
    constructor_validate_and_infer_types();
}

bool op::v4::LSTMCell::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v4_LSTMCell_visit_attributes);
    return op::util::RNNCellBase::visit_attributes(visitor);
}

void op::v4::LSTMCell::validate_and_infer_types() {
    OV_OP_SCOPE(v4_LSTMCell_validate_and_infer_types);

    auto result_et = element::dynamic;

    // Get input partial shape for all inputs
    const auto& x_pshape = get_input_partial_shape(0);
    const auto& ht_pshape = get_input_partial_shape(1);
    const auto& ct_pshape = get_input_partial_shape(2);
    const auto& w_pshape = get_input_partial_shape(3);
    const auto& r_pshape = get_input_partial_shape(4);
    const auto& b_pshape = get_input_partial_shape(5);

    // Validate input element types and save result for output type
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(2)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(3)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(4)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(5)),
                          "Element types for X, initial_hidden_state, initial_cell_state, W, R and B do not "
                          "match.");

    std::vector<ov::PartialShape> input_shapes = {x_pshape, ht_pshape, ct_pshape, w_pshape, r_pshape, b_pshape};
    std::vector<ov::PartialShape> output_shapes = shape_infer(this, input_shapes);
    // Mark inputs which are relevant to output parameters
    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);
    set_input_is_relevant_to_shape(4);

    // Set output size, type and shape
    set_output_size(2);
    set_output_type(0, result_et, output_shapes[0]);
    set_output_type(1, result_et, output_shapes[1]);
}

Output<Node> op::v4::LSTMCell::get_default_bias_input() const {
    return Output<Node>{op::v0::Constant::create(get_input_element_type(0),
                                                 Shape{lstm_cell::gates_count * get_hidden_size()},
                                                 std::vector<float>{0.f})};
}

std::shared_ptr<Node> op::v4::LSTMCell::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v4_LSTMCell_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 5) {
        return std::make_shared<LSTMCell>(new_args.at(0),
                                          new_args.at(1),
                                          new_args.at(2),
                                          new_args.at(3),
                                          new_args.at(4),
                                          get_hidden_size(),
                                          get_activations(),
                                          get_activations_alpha(),
                                          get_activations_beta(),
                                          get_clip());
    } else if (new_args.size() == 6) {
        return std::make_shared<LSTMCell>(new_args.at(0),
                                          new_args.at(1),
                                          new_args.at(2),
                                          new_args.at(3),
                                          new_args.at(4),
                                          new_args.at(5),
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
