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
    element::Type arg_type = get_input_element_type(0);
    PartialShape output_shape_0{PartialShape::dynamic(4)};
    PartialShape output_shape_1{PartialShape::dynamic(3)};
    auto x_pshape = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(
        this, x_pshape.rank().compatible(3), "The 'X' input must be a 3D tensor.");
    if (x_pshape.is_static())
    {
        size_t gates_count = 3;
        size_t batch_size = get_input_partial_shape(0).get_shape()[0];
        size_t seq_length = get_input_partial_shape(0).get_shape()[1];
        size_t input_size = get_input_partial_shape(0).get_shape()[2];
        size_t num_directions =
            m_direction == op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;
        output_shape_0 = Shape{batch_size, num_directions, seq_length, m_hidden_size};
        output_shape_1 = Shape{batch_size, num_directions, m_hidden_size};

        auto h_state_pshape = get_input_partial_shape(1);
        auto seq_lengths_pshape = get_input_partial_shape(2);
        auto w_pshape = get_input_partial_shape(3);
        auto r_pshape = get_input_partial_shape(4);
        auto b_pshape = get_input_partial_shape(5);

        if (h_state_pshape.is_static())
        {
            auto h_state_shape = h_state_pshape.to_shape();
            NODE_VALIDATION_CHECK(
                this,
                (h_state_shape == Shape{batch_size, num_directions, m_hidden_size}),
                "Input tensor initial_hidden_state must have shape (",
                batch_size,
                ", ",
                num_directions,
                ", ",
                m_hidden_size,
                "). Actual shape is:",
                h_state_shape,
                ".");
        }

        if (seq_lengths_pshape.is_static())
        {
            const Shape& seq_length_shape = seq_lengths_pshape.to_shape();
            NODE_VALIDATION_CHECK(this,
                                  (seq_length_shape == Shape{batch_size}),
                                  "Input tensor sequence_lengths must have shape (",
                                  batch_size,
                                  "). Actual shape is:",
                                  seq_length_shape,
                                  ".");
        }

        if (w_pshape.is_static())
        {
            auto w_shape = w_pshape.to_shape();
            NODE_VALIDATION_CHECK(
                this,
                (w_shape == Shape{num_directions, gates_count * m_hidden_size, input_size}),
                "Input tensor W must have shape (",
                num_directions,
                ", ",
                gates_count * m_hidden_size,
                ", ",
                input_size,
                "). Actual shape is:",
                w_shape,
                ".");
        }

        if (r_pshape.is_static())
        {
            auto r_shape = r_pshape.to_shape();
            NODE_VALIDATION_CHECK(
                this,
                (r_shape == Shape{num_directions, gates_count * m_hidden_size, m_hidden_size}),
                "Input tensor R must have shape (",
                num_directions,
                ", ",
                m_hidden_size,
                ", ",
                m_hidden_size,
                "). Actual shape is:",
                r_shape,
                ".");
        }

        if (b_pshape.is_static())
        {
            auto b_shape = b_pshape.to_shape();
            NODE_VALIDATION_CHECK(
                this,
                (b_shape ==
                 Shape{num_directions, (gates_count + m_linear_before_reset) * m_hidden_size}),
                "Input tensor B must have shape (",
                num_directions,
                ", ",
                (gates_count + m_linear_before_reset) * m_hidden_size,
                "). Actual shape is:",
                b_shape,
                ".");
        }
    }
    set_output_type(0, arg_type, output_shape_0);
    set_output_type(1, arg_type, output_shape_1);
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
