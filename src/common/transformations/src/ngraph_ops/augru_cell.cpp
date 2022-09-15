// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/augru_cell.hpp"

#include <cmath>

#include "itt.hpp"

using namespace std;

BWDCMP_RTTI_DEFINITION(ov::op::internal::AUGRUCell);

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
    for (const auto& input : inputs()) {
        if (input.get_partial_shape().rank().is_dynamic()) {
            set_output_type(0, get_input_element_type(0), PartialShape::dynamic(2));
            return;
        }
    }
    auto merged_batch_size = Dimension::dynamic();
    auto merged_hidden_size = Dimension::dynamic();
    auto result_et = element::dynamic;

    // Get input partial shape for all inputs
    const auto& x_pshape = get_input_partial_shape(0);
    const auto& ht_pshape = get_input_partial_shape(1);
    const auto& w_pshape = get_input_partial_shape(2);
    const auto& r_pshape = get_input_partial_shape(3);
    const auto& b_pshape = get_input_partial_shape(4);
    const auto& a_pshape = get_input_partial_shape(5);

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

    validate_input_rank_dimension({x_pshape, ht_pshape, w_pshape, r_pshape, b_pshape});

    // `A` input shape validation // [batch_size, 1]
    NODE_VALIDATION_CHECK(this, a_pshape.rank().compatible(2), "'A' input must be a 2D tensor.");
    if (a_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(this, a_pshape[1].compatible(1), "The last dimension of `A` shape must be equal to `1`.");
    }

    // Validate input types and save result for output type
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(2)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(3)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(4)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(5)),
                          "Element types for inputs do not match.");

    // Merge batch_size dimension across all inputs to evaluate output[0] dimension
    NODE_VALIDATION_CHECK(this,
                          Dimension::merge(merged_batch_size, merged_batch_size, ht_pshape[0]) &&
                              Dimension::merge(merged_batch_size, merged_batch_size, a_pshape[0]) &&
                              Dimension::merge(merged_batch_size, merged_batch_size, x_pshape[0]),
                          "Dimension batch_size is not matched between inputs.");

    // Merge hidden_size dimension across all inputs to evaluate output[1] dimension
    NODE_VALIDATION_CHECK(this,
                          Dimension::merge(merged_hidden_size, merged_hidden_size, ht_pshape[1]) &&
                              Dimension::merge(merged_hidden_size, merged_hidden_size, r_pshape[1]),
                          "Dimension hidden_size not matched for R and initial_hidden_state inputs.");

    // Validate hidden_size value for W, B and R inputs
    if (merged_hidden_size.is_static()) {
        if (w_pshape[0].is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  w_pshape[0].compatible(merged_hidden_size * s_gates_count),
                                  "Parameter hidden_size mistmatched in W input. Current value is: ",
                                  w_pshape[0].get_length(),
                                  ", expected: ",
                                  merged_hidden_size.get_length() * s_gates_count,
                                  ".");
        }

        if (r_pshape[0].is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  r_pshape[0].compatible(merged_hidden_size * s_gates_count),
                                  "Parameter hidden_size mistmatched in R input. Current value is: ",
                                  r_pshape[0].get_length(),
                                  ", expected: ",
                                  merged_hidden_size.get_length() * s_gates_count,
                                  ".");
        }

        if (b_pshape[0].is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  b_pshape[0].compatible(merged_hidden_size * (s_gates_count + m_linear_before_reset)),
                                  "Parameter hidden_size mistmatched in B input. Current value is: ",
                                  b_pshape[0].get_length(),
                                  ", expected: ",
                                  merged_hidden_size.get_length() * (s_gates_count + m_linear_before_reset),
                                  ".");
        }
    }

    // Set output size, type and shape
    set_output_size(1);
    set_output_type(0, result_et, {merged_batch_size, merged_hidden_size});
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
