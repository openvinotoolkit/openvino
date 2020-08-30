//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <cmath>
#include <functional>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/lstm_cell.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::LSTMCell::type_info;

op::LSTMCell::LSTMCell()
{
    m_activations = {"sigmoid", "tanh", "tanh"};
    m_activation_f = get_activation_function(0);
    m_activation_g = get_activation_function(1);
    m_activation_h = get_activation_function(2);
}

op::LSTMCell::LSTMCell(const Output<Node>& X,
                       const Output<Node>& initial_hidden_state,
                       const Output<Node>& initial_cell_state,
                       const Output<Node>& W,
                       const Output<Node>& R,
                       size_t hidden_size,
                       const vector<string>& activations,
                       const vector<float>& activations_alpha,
                       const vector<float>& activations_beta,
                       float clip)
    : RNNCellBase({X, initial_hidden_state, initial_cell_state, W, R},
                  hidden_size,
                  clip,
                  activations,
                  activations_alpha,
                  activations_beta)
    , m_activation_f{get_activation_function(0)}
    , m_activation_g{get_activation_function(1)}
    , m_activation_h{get_activation_function(2)}
{
    set_argument(5, get_default_bias_input());
    constructor_validate_and_infer_types();
}

op::LSTMCell::LSTMCell(const Output<Node>& X,
                       const Output<Node>& initial_hidden_state,
                       const Output<Node>& initial_cell_state,
                       const Output<Node>& W,
                       const Output<Node>& R,
                       const Output<Node>& B,
                       size_t hidden_size,
                       const vector<string>& activations,
                       const vector<float>& activations_alpha,
                       const vector<float>& activations_beta,
                       float clip)
    : RNNCellBase({X, initial_hidden_state, initial_cell_state, W, R, B},
                  hidden_size,
                  clip,
                  activations,
                  activations_alpha,
                  activations_beta)
    , m_activation_f{get_activation_function(0)}
    , m_activation_g{get_activation_function(1)}
    , m_activation_h{get_activation_function(2)}
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::LSTMCell::visit_attributes(AttributeVisitor& visitor)
{
    op::util::RNNCellBase::visit_attributes(visitor);
}

void op::LSTMCell::validate_and_infer_types()
{
    auto merged_batch_size = Dimension::dynamic();
    auto merged_hidden_size = Dimension::dynamic();
    auto result_et = element::dynamic;

    // Get input partial shape for all inputs
    const auto& x_pshape = get_input_partial_shape(0);
    const auto& ht_pshape = get_input_partial_shape(1);
    const auto& ct_pshape = get_input_partial_shape(2);
    const auto& w_pshape = get_input_partial_shape(3);
    const auto& r_pshape = get_input_partial_shape(4);
    const auto& b_pshape = get_input_partial_shape(5);

    // Validate rank and dimension for initial_cell_state input
    NODE_VALIDATION_CHECK(this,
                          (ct_pshape.rank().is_static()),
                          "LSTMCell input tensor initial_cell_state shall have static rank.");

    NODE_VALIDATION_CHECK(this,
                          (ct_pshape.rank().get_length() == 2),
                          "LSTMCell input tensor initial_cell_state shall have dimension 2D.");

    validate_input_rank_dimension({x_pshape, ht_pshape, w_pshape, r_pshape, b_pshape});

    // Validate input element types and save result for output type
    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
            element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
            element::Type::merge(result_et, result_et, get_input_element_type(2)) &&
            element::Type::merge(result_et, result_et, get_input_element_type(3)) &&
            element::Type::merge(result_et, result_et, get_input_element_type(4)) &&
            element::Type::merge(result_et, result_et, get_input_element_type(5)),
        "Element types for X, initial_hidden_state, initial_cell_state, W, R and B do not match.");

    // Merge batch_size dimension across all inputs to evaluate output[0] dimension
    NODE_VALIDATION_CHECK(
        this,
        Dimension::merge(merged_batch_size, merged_batch_size, ht_pshape[0]) &&
            Dimension::merge(merged_batch_size, merged_batch_size, ct_pshape[0]) &&
            Dimension::merge(merged_batch_size, merged_batch_size, x_pshape[0]),
        "Parameter batch_size not matched for X, initial_hidden_state or initial_cell_state "
        "inputs.");

    // Merge hidden_size dimension across all inputs to evaluate output[1] dimension
    NODE_VALIDATION_CHECK(
        this,
        Dimension::merge(merged_hidden_size, merged_hidden_size, ht_pshape[1]) &&
            Dimension::merge(merged_hidden_size, merged_hidden_size, ct_pshape[1]) &&
            Dimension::merge(merged_hidden_size, merged_hidden_size, r_pshape[1]),
        "Parameter hidden_size not matched for R, initial_hidden_state and initial_cell_state "
        "inputs.");

    // Validate hidden_size value for W, R and P inputs
    if (merged_hidden_size.is_static())
    {
        if (w_pshape[0].is_static())
        {
            NODE_VALIDATION_CHECK(
                this,
                w_pshape[0].compatible(merged_hidden_size * s_gates_count),
                "Parameter hidden_size mistmatched in W input. Current value is: ",
                w_pshape[0].get_length(),
                ", expected: ",
                merged_hidden_size.get_length() * s_gates_count,
                ".");
        }

        if (r_pshape[0].is_static())
        {
            NODE_VALIDATION_CHECK(
                this,
                r_pshape[0].compatible(merged_hidden_size * s_gates_count),
                "Parameter hidden_size mistmatched in R input. Current value is: ",
                r_pshape[0].get_length(),
                ", expected: ",
                merged_hidden_size.get_length() * s_gates_count,
                ".");
        }

        if (b_pshape[0].is_static())
        {
            NODE_VALIDATION_CHECK(
                this,
                b_pshape[0].compatible(merged_hidden_size * s_gates_count),
                "Parameter hidden_size mistmatched in B input. Current value is: ",
                b_pshape[0].get_length(),
                ", expected: ",
                merged_hidden_size.get_length() * s_gates_count,
                ".");
        }
    }

    // Mark inputs which are relevant to output parameters
    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);
    set_input_is_relevant_to_shape(4);

    // Set output size, type and shape
    set_output_size(2);
    set_output_type(0, result_et, {merged_batch_size, merged_hidden_size});
    set_output_type(1, result_et, {merged_batch_size, merged_hidden_size});
}

Output<Node> op::LSTMCell::get_default_bias_input() const
{
    return Output<Node>{op::Constant::create(
        get_input_element_type(0), Shape{s_gates_count * get_hidden_size()}, vector<float>{0.f})};
}

shared_ptr<Node> op::LSTMCell::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    if (new_args.size() == 5)
    {
        return make_shared<LSTMCell>(new_args.at(0),
                                     new_args.at(1),
                                     new_args.at(2),
                                     new_args.at(3),
                                     new_args.at(4),
                                     get_hidden_size(),
                                     get_activations(),
                                     get_activations_alpha(),
                                     get_activations_beta(),
                                     get_clip());
    }
    else if (new_args.size() == 6)
    {
        return make_shared<LSTMCell>(new_args.at(0),
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
    }
    else
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
}
