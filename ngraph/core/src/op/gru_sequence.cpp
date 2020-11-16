//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include <memory>
#include <string>
#include <vector>

#include "ngraph/op/gru_sequence.hpp"
#include "ngraph/op/util/recurrent_sequence.hpp"
#include "ngraph/opsets/opset4.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v5::GRUSequence, "GRUSequence", 5);

op::v5::GRUSequence::GRUSequence()
    : m_direction(op::RecurrentSequenceDirection::FORWARD)
    , m_linear_before_reset(false)
{
}

op::v5::GRUSequence::GRUSequence(const Output<Node>& X,
                                 const Output<Node>& H_t,
                                 const Output<Node>& sequence_lengths,
                                 const Output<Node>& W,
                                 const Output<Node>& R,
                                 const Output<Node>& B,
                                 std::size_t hidden_size,
                                 op::RecurrentSequenceDirection direction,
                                 const std::vector<std::string>& activations,
                                 const std::vector<float>& activations_alpha,
                                 const std::vector<float>& activations_beta,
                                 float clip,
                                 bool linear_before_reset)
    : RNNCellBase({X, H_t, sequence_lengths, W, R, B},
                  hidden_size,
                  clip,
                  activations,
                  activations_alpha,
                  activations_beta)
    , m_direction(direction)
    , m_linear_before_reset(linear_before_reset)
{
    constructor_validate_and_infer_types();
}

void op::v5::GRUSequence::validate_and_infer_types()
{
    for (const auto& input : inputs())
    {
        if (input.get_partial_shape().rank().is_dynamic())
        {
            set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
            set_output_type(1, get_input_element_type(0), PartialShape::dynamic());
            return;
        }
    }

    auto gru_seq_gates_count = 3;
    auto merged_batch_size = Dimension::dynamic();
    auto merged_hidden_size = Dimension::dynamic();
    auto merged_num_directions = Dimension::dynamic();
    auto result_et = element::dynamic;

    auto x_pshape = get_input_partial_shape(0);
    auto ht_pshape = get_input_partial_shape(1);
    auto sl_pshape = get_input_partial_shape(2);
    auto w_pshape = get_input_partial_shape(3);
    auto r_pshape = get_input_partial_shape(4);
    auto b_pshape = get_input_partial_shape(5);

    ngraph::op::util::validate_seq_input_rank_dimension(
        {x_pshape, ht_pshape, sl_pshape, w_pshape, r_pshape, b_pshape});

    // Validate input types and save result for output type
    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
            element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
            element::Type::merge(result_et, result_et, get_input_element_type(3)) &&
            element::Type::merge(result_et, result_et, get_input_element_type(4)) &&
            element::Type::merge(result_et, result_et, get_input_element_type(5)),
        "Element types for X, initial_hidden_state, W, R and B inputs do not "
        "match.");

    // Merge batch_size dimension across all inputs to evaluate output[0] dimension
    NODE_VALIDATION_CHECK(this,
                          Dimension::merge(merged_batch_size, merged_batch_size, ht_pshape[0]) &&
                              Dimension::merge(merged_batch_size, merged_batch_size, x_pshape[0]) &&
                              Dimension::merge(merged_batch_size, merged_batch_size, sl_pshape[0]),
                          "Parameter batch_size not matched in RNNSequence.");

    // Merge hidden_size dimension across all inputs to evaluate output dimension
    NODE_VALIDATION_CHECK(this,
                          Dimension::merge(merged_hidden_size, merged_hidden_size, ht_pshape[2]) &&
                              Dimension::merge(merged_hidden_size, merged_hidden_size, r_pshape[2]),
                          "Parameter hidden_size not matched RNNSequence.");

    // Merge num_directions dimension across all inputs to evaluate output dimension
    NODE_VALIDATION_CHECK(
        this,
        Dimension::merge(merged_num_directions, merged_num_directions, ht_pshape[1]) &&
            Dimension::merge(merged_num_directions, merged_num_directions, w_pshape[0]) &&
            Dimension::merge(merged_num_directions, merged_num_directions, r_pshape[0]) &&
            Dimension::merge(merged_num_directions, merged_num_directions, b_pshape[0]),
        "Parameter num_directions not matched in RNNSequence.");

    // Validate hidden_size value for W, R, B inputs
    if (merged_hidden_size.is_static())
    {
        if (w_pshape[1].is_static())
        {
            NODE_VALIDATION_CHECK(
                this,
                w_pshape[1].compatible(merged_hidden_size * gru_seq_gates_count),
                "Parameter hidden_size mistmatched in W input. Current value is: ",
                w_pshape[1].get_length(),
                ", expected: ",
                merged_hidden_size.get_length() * gru_seq_gates_count,
                ".");
        }

        if (r_pshape[1].is_static())
        {
            NODE_VALIDATION_CHECK(
                this,
                r_pshape[1].compatible(merged_hidden_size * gru_seq_gates_count),
                "Parameter hidden_size mistmatched in R input. Current value is: ",
                r_pshape[1].get_length(),
                ", expected: ",
                merged_hidden_size.get_length() * gru_seq_gates_count,
                ".");
        }

        if (b_pshape[1].is_static())
        {
            NODE_VALIDATION_CHECK(
                this,
                b_pshape[1].compatible(merged_hidden_size * (m_linear_before_reset
                                                                 ? (gru_seq_gates_count + 1)
                                                                 : gru_seq_gates_count)),
                "Parameter hidden_size mistmatched in B input. Current value is: ",
                b_pshape[1].get_length(),
                ", expected: ",
                merged_hidden_size.get_length() *
                    (m_linear_before_reset ? (gru_seq_gates_count + 1) : gru_seq_gates_count),
                ".");
        }
    }

    // Mark inputs which are relevant to output parameters
    for (size_t i = 0; i <= 5; ++i)
        set_input_is_relevant_to_shape(i);

    // Set output size, type and shape
    set_output_size(2);
    set_output_type(
        0, result_et, {merged_batch_size, merged_num_directions, x_pshape[1], merged_hidden_size});
    set_output_type(1, result_et, {merged_batch_size, merged_num_directions, merged_hidden_size});
}

bool op::v5::GRUSequence::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("direction", m_direction);
    visitor.on_attribute("linear_before_reset", m_linear_before_reset);
    return op::util::RNNCellBase::visit_attributes(visitor);
}

shared_ptr<Node> op::v5::GRUSequence::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v5::GRUSequence>(new_args.at(0),
                                            new_args.at(1),
                                            new_args.at(2),
                                            new_args.at(3),
                                            new_args.at(4),
                                            new_args.at(5),
                                            m_hidden_size,
                                            m_direction,
                                            m_activations,
                                            m_activations_alpha,
                                            m_activations_beta,
                                            m_clip,
                                            m_linear_before_reset);
}
