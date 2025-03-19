// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/multinomial.hpp"

#include <cstring>

#include "itt.hpp"
#include "multinomial_shape_inference.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/reference/multinomial.hpp"

namespace ov {

// ------------------------------ v13 ------------------------------

op::v13::Multinomial::Multinomial(const Output<Node>& probs,
                                  const Output<Node>& num_samples,
                                  const ov::element::Type_t convert_type,
                                  const bool with_replacement,
                                  const bool log_probs,
                                  const uint64_t global_seed,
                                  const uint64_t op_seed)
    : Op({probs, num_samples}),
      m_convert_type(convert_type),
      m_with_replacement(with_replacement),
      m_log_probs(log_probs),
      m_global_seed(global_seed),
      m_op_seed(op_seed) {
    constructor_validate_and_infer_types();
}

bool op::v13::Multinomial::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v13_Multinomial_visit_attributes);
    visitor.on_attribute("convert_type", m_convert_type);
    visitor.on_attribute("with_replacement", m_with_replacement);
    visitor.on_attribute("log_probs", m_log_probs);
    visitor.on_attribute("global_seed", m_global_seed);
    visitor.on_attribute("op_seed", m_op_seed);
    return true;
}

void op::v13::Multinomial::validate_and_infer_types() {
    OV_OP_SCOPE(v13_Multinomial_validate_and_infer_types);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    const auto output_shapes = shape_infer(this, input_shapes);

    multinomial::validate::input_types(this);

    set_output_type(0, m_convert_type, output_shapes[0]);
}

std::shared_ptr<Node> op::v13::Multinomial::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v13_Multinomial_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    return std::make_shared<op::v13::Multinomial>(new_args.at(0),
                                                  new_args.at(1),
                                                  m_convert_type,
                                                  m_with_replacement,
                                                  m_log_probs,
                                                  m_global_seed,
                                                  m_op_seed);
}

ov::element::Type_t op::v13::Multinomial::get_convert_type() const {
    return m_convert_type;
}

bool op::v13::Multinomial::get_with_replacement() const {
    return m_with_replacement;
}

bool op::v13::Multinomial::get_log_probs() const {
    return m_log_probs;
}

uint64_t op::v13::Multinomial::get_global_seed() const {
    return m_global_seed;
}

uint64_t op::v13::Multinomial::get_op_seed() const {
    return m_op_seed;
}

void op::v13::Multinomial::set_convert_type(const ov::element::Type_t convert_type) {
    m_convert_type = convert_type;
}

void op::v13::Multinomial::set_with_replacement(const bool with_replacement) {
    m_with_replacement = with_replacement;
}

void op::v13::Multinomial::set_log_probs(const bool log_probs) {
    m_log_probs = log_probs;
}

void op::v13::Multinomial::set_global_seed(const uint64_t global_seed) {
    m_global_seed = global_seed;
}

void op::v13::Multinomial::set_op_seed(const uint64_t op_seed) {
    m_op_seed = op_seed;
}

namespace op {
namespace multinomial {
namespace validate {
void input_types(const Node* op) {
    NODE_VALIDATION_CHECK(op,
                          op->get_input_element_type(0).is_real() || op->get_input_element_type(0).is_dynamic(),
                          "Expected floating point type as element type for the 'probs' input.");

    NODE_VALIDATION_CHECK(
        op,
        op->get_input_element_type(1).is_integral_number() || op->get_input_element_type(1).is_dynamic(),
        "Expected integer type as element type for the 'num_samples' input.");
}
}  // namespace validate
}  // namespace multinomial
}  // namespace op
}  // namespace ov
