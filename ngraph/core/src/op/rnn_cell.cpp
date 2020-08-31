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

#include "ngraph/builder/reshape.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/rnn_cell.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

constexpr NodeTypeInfo op::RNNCell::type_info;

op::RNNCell::RNNCell()
{
    m_activations = {"tanh"};
    m_activation_f = get_activation_function(0);
}

op::RNNCell::RNNCell(const Output<Node>& X,
                     const Output<Node>& initial_hidden_state,
                     const Output<Node>& W,
                     const Output<Node>& R,
                     size_t hidden_size,
                     const vector<string>& activations,
                     const vector<float>& activations_alpha,
                     const vector<float>& activations_beta,
                     float clip)
    : FusedOp({X, initial_hidden_state, W, R})
    , RNNCellBase(hidden_size, clip, activations, activations_alpha, activations_beta)
    , m_activation_f{get_activation_function(0)}
{
    set_argument(4, get_default_bias_input());
    constructor_validate_and_infer_types();
}

op::RNNCell::RNNCell(const Output<Node>& X,
                     const Output<Node>& initial_hidden_state,
                     const Output<Node>& W,
                     const Output<Node>& R,
                     const Output<Node>& B,
                     size_t hidden_size,
                     const vector<string>& activations,
                     const vector<float>& activations_alpha,
                     const vector<float>& activations_beta,
                     float clip)
    : FusedOp({X, initial_hidden_state, W, R, B})
    , RNNCellBase(hidden_size, clip, activations, activations_alpha, activations_beta)
    , m_activation_f{get_activation_function(0)}
{
    constructor_validate_and_infer_types();
}

bool op::RNNCell::visit_attributes(AttributeVisitor& visitor)
{
    return op::util::RNNCellBase::visit_attributes(visitor);
}

void op::RNNCell::pre_validate_and_infer_types()
{
    set_output_type(0, get_input_element_type(0), PartialShape::dynamic());

    if (is_dynamic())
    {
        return;
    }

    const auto& x_pshape = get_input_partial_shape(0);
    const auto& ht_pshape = get_input_partial_shape(1);
    const auto& w_pshape = get_input_partial_shape(2);
    const auto& r_pshape = get_input_partial_shape(3);

    NODE_VALIDATION_CHECK(this,
                          (x_pshape.is_static() || w_pshape.is_static() || r_pshape.is_static() ||
                           ht_pshape.is_static()),
                          "RNNCell supports only static input tensors.");

    const Shape& x_shape{x_pshape.to_shape()};

    const size_t batch_size = x_shape.at(0);
    const size_t input_size = x_shape.at(1);

    const Shape& w_shape{w_pshape.to_shape()};
    const Shape& r_shape{r_pshape.to_shape()};
    const Shape& ht_shape{ht_pshape.to_shape()};

    NODE_VALIDATION_CHECK(this,
                          (w_shape == Shape{get_hidden_size(), input_size}),
                          "Input tensor W must have shape (",
                          get_hidden_size(),
                          ", ",
                          input_size,
                          "). Actual shape is:",
                          w_shape,
                          ".");
    NODE_VALIDATION_CHECK(this,
                          (r_shape == Shape{get_hidden_size(), get_hidden_size()}),
                          "Input tensor R must have shape (",
                          get_hidden_size(),
                          ", ",
                          get_hidden_size(),
                          "). Actual shape is:",
                          w_shape,
                          ".");
    NODE_VALIDATION_CHECK(this,
                          (ht_shape == Shape{batch_size, get_hidden_size()}),
                          "Input tensor initial_hidden_state must have shape (",
                          batch_size,
                          ", ",
                          get_hidden_size(),
                          "). Actual shape is:",
                          w_shape,
                          ".");

    const auto& b_pshape = get_input_partial_shape(4);

    NODE_VALIDATION_CHECK(
        this, b_pshape.is_static(), "RNNCell supports only static input tensors.");

    const Shape& b_shape{b_pshape.to_shape()};

    NODE_VALIDATION_CHECK(this,
                          (b_shape == Shape{get_hidden_size()}),
                          "Input tensor B must have shape (",
                          get_hidden_size(),
                          "). Actual shape is:",
                          b_shape,
                          ".");
}

void op::RNNCell::validate_and_infer_types()
{
    std::vector<ngraph::PartialShape> input_param{};

    auto merged_batch_size = Dimension::dynamic();
    auto merged_hidden_size = Dimension::dynamic();
    auto result_et = element::dynamic;

    // Copy all inputs for further validation
    for (size_t i = 0; i < get_input_size(); i++)
    {
        input_param.push_back(get_input_partial_shape(i));
    }

    // Get input partial shape for all inputs
    const auto& x_pshape = get_input_partial_shape(0);
    const auto& ht_pshape = get_input_partial_shape(1);
    const auto& w_pshape = get_input_partial_shape(2);
    const auto& r_pshape = get_input_partial_shape(3);
    const auto& b_pshape = get_input_partial_shape(4);

    validate_input_rank_dimension(input_param);

    // Validate input types and save result for output type
    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
            element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
            element::Type::merge(result_et, result_et, get_input_element_type(2)) &&
            element::Type::merge(result_et, result_et, get_input_element_type(3)) &&
            element::Type::merge(result_et, result_et, get_input_element_type(4)),
        "Element types for X, initial_hidden_state, W, R and B inputs do not match.");

    // Merge batch_size dimension across all inputs to evaluate output[0] dimension
    NODE_VALIDATION_CHECK(
        this,
        Dimension::merge(merged_batch_size, merged_batch_size, ht_pshape[0]) &&
            Dimension::merge(merged_batch_size, merged_batch_size, x_pshape[0]),
        "Parameter batch_size not matched for X and initial_hidden_state inputs.");

    // Merge hidden_size dimension across all inputs to evaluate output[1] dimension
    NODE_VALIDATION_CHECK(
        this,
        Dimension::merge(merged_hidden_size, merged_hidden_size, ht_pshape[1]) &&
            Dimension::merge(merged_hidden_size, merged_hidden_size, r_pshape[1]),
        "Parameter hidden_size not matched for R and initial_hidden_state inputs.");

    // Validate hidden_size value for W, B and R inputs
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
    set_input_is_relevant_to_shape(3);
    set_input_is_relevant_to_shape(4);

    // Set output size, type and shape
    set_output_size(1);
    set_output_type(0, result_et, {merged_batch_size, merged_hidden_size});
}

OutputVector op::RNNCell::decompose_op() const
{
    // ------ VARIABLE'S NAMES AND ACRONYM DEFINITIONS ------
    // The names used below are analogous to the one used in ONNX documentation.
    //
    // ------ ACRONYMS ------
    // i_t - input gate at current time step
    // t - time step (t-1 means previous time step)
    // X   - The input data tensor. Shape: [batch_size, input_size].
    // W   - The weight tensor for input gate. Shape: [hidden_size, input_size].
    // R   - The recurrence weight tensor for input gate. Shape: [hidden_size, hidden_size].
    // H_t - The hidden state tensor at current time step. Shape: [batch_size, hidden_size].
    // B   - The bias tensor for the input gate. Shape: [hidden_size].
    // Wb  - W bias vectors for input gate.
    // Rb  - R bias vectors for input gate.
    // ------ VARIABLE NAMES ------
    // Xt_W    - Input sequence multiplied by weights tensor at current time step.
    // Ht_R    - Hidden state multiplied by weights tensor at current time step.

    // (.) - Denotes element-wise multiplication.
    // *   - Denotes dot product.

    // ---- Equations ----
    // f - is activation functions.
    // Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
    // --------------------

    Output<Node> X = input_value(0);
    Output<Node> H_t = input_value(1);
    Output<Node> W = input_value(2);
    Output<Node> R = input_value(3);
    Output<Node> bias = input_value(4);

    // Xt*(W^T)
    auto Xt_W = std::make_shared<op::Dot>(X, builder::opset1::transpose(W));
    // Ht-1*(R^T)
    auto Ht_R = std::make_shared<op::Dot>(H_t, builder::opset1::transpose(R));
    // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb
    auto i_t = add(Xt_W, add(Ht_R, bias));

    // f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
    i_t = m_activation_f(clip(i_t));

    return {i_t};
}

Output<Node> op::RNNCell::get_default_bias_input() const
{
    return Output<Node>{
        op::Constant::create(get_input_element_type(0),
                             Shape{s_gates_count * get_hidden_size()},
                             vector<float>(s_gates_count * get_hidden_size(), 0.f))};
}

shared_ptr<Node> op::RNNCell::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    if (new_args.size() == 4)
    {
        return make_shared<RNNCell>(new_args.at(0),
                                    new_args.at(1),
                                    new_args.at(2),
                                    new_args.at(3),
                                    get_hidden_size(),
                                    get_activations(),
                                    get_activations_alpha(),
                                    get_activations_beta(),
                                    get_clip());
    }
    else if (new_args.size() == 5)
    {
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
    }
    else
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
}
