// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/rnn_cell.hpp"

#include <cmath>

#include "itt.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v0::RNNCell);

op::v0::RNNCell::RNNCell() {
    m_activations = {"tanh"};
    m_activation_f = get_activation_function(0);
}

op::v0::RNNCell::RNNCell(const Output<Node>& X,
                         const Output<Node>& initial_hidden_state,
                         const Output<Node>& W,
                         const Output<Node>& R,
                         size_t hidden_size,
                         const vector<string>& activations,
                         const vector<float>& activations_alpha,
                         const vector<float>& activations_beta,
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
                         const vector<string>& activations,
                         const vector<float>& activations_alpha,
                         const vector<float>& activations_beta,
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
    for (const auto& input : inputs()) {
        if (input.get_partial_shape().rank().is_dynamic()) {
            set_output_type(0, get_input_element_type(0), ov::PartialShape::dynamic());
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

    validate_input_rank_dimension({x_pshape, ht_pshape, w_pshape, r_pshape, b_pshape});

    // Validate input types and save result for output type
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(2)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(3)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(4)),
                          "Element types for X, initial_hidden_state, W, R and B inputs do not match.");

    // Merge batch_size dimension across all inputs to evaluate output[0] dimension
    NODE_VALIDATION_CHECK(this,
                          Dimension::merge(merged_batch_size, merged_batch_size, ht_pshape[0]) &&
                              Dimension::merge(merged_batch_size, merged_batch_size, x_pshape[0]),
                          "Parameter batch_size not matched for X and initial_hidden_state inputs.");

    // Merge hidden_size dimension across all inputs to evaluate output[1] dimension
    NODE_VALIDATION_CHECK(this,
                          Dimension::merge(merged_hidden_size, merged_hidden_size, ht_pshape[1]) &&
                              Dimension::merge(merged_hidden_size, merged_hidden_size, r_pshape[1]),
                          "Parameter hidden_size not matched for R and initial_hidden_state inputs.");

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
                                  b_pshape[0].compatible(merged_hidden_size * s_gates_count),
                                  "Parameter hidden_size mistmatched in B input. Current value is: ",
                                  b_pshape[0].get_length(),
                                  ", expected: ",
                                  merged_hidden_size.get_length() * s_gates_count,
                                  ".");
        }
    }

    // Mark inputs which are relevant to output parameters
    for (size_t i = 0; i <= 4; ++i)
        set_input_is_relevant_to_shape(i);

    // Set output size, type and shape
    set_output_size(1);
    set_output_type(0, result_et, {merged_batch_size, merged_hidden_size});
}

Output<Node> op::v0::RNNCell::get_default_bias_input() const {
    return Output<Node>{op::v0::Constant::create(get_input_element_type(0),
                                                 ov::Shape{s_gates_count * get_hidden_size()},
                                                 vector<float>(s_gates_count * get_hidden_size(), 0.f))};
}

shared_ptr<Node> op::v0::RNNCell::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_RNNCell_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 4) {
        return make_shared<RNNCell>(new_args.at(0),
                                    new_args.at(1),
                                    new_args.at(2),
                                    new_args.at(3),
                                    get_hidden_size(),
                                    get_activations(),
                                    get_activations_alpha(),
                                    get_activations_beta(),
                                    get_clip());
    } else if (new_args.size() == 5) {
        return make_shared<RNNCell>(new_args.at(0),
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
        throw ngraph_error("Incorrect number of new arguments");
    }
}
