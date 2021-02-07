//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "ngraph/op/loop.hpp"
#include <ngraph/validation_util.hpp>
#include "itt.hpp"
#include "ngraph/factory.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "ngraph/specialize_function.hpp"

#include "ngraph/runtime/reference/loop.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v5::Loop, "Loop", 5);

op::v5::Loop::Loop(const Output<Node>& trip_count, const Output<Node>& execution_condition)
{
    set_argument(0, trip_count);
    set_argument(1, execution_condition);
}

bool op::v5::Loop::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v5_Loop_visit_attributes);
    visitor.on_attribute("body", m_body);
    visitor.on_attribute("input_descriptions", m_input_descriptions);
    visitor.on_attribute("output_descriptions", m_output_descriptions);
    visitor.on_attribute("special_body_ports", m_special_body_ports);

    return true;
}

void op::v5::Loop::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v5_Loop_validate_and_infer_types);
    if (m_special_body_ports.current_iteration_input_idx >= 0)
    {
        const auto& cur_iter_rank = m_body->get_parameters()
                                        .at(m_special_body_ports.current_iteration_input_idx)
                                        ->get_partial_shape()
                                        .rank();
        if (cur_iter_rank.is_static())
        {
            NODE_VALIDATION_CHECK(this,
                                  cur_iter_rank.compatible(1) || cur_iter_rank.compatible(0),
                                  "Rank of CurrentIteration input must be equal to 0 or 1");
        }
    }
    bool zero_number_of_iter = false;
    const auto& loop_execution_condition = input_value(1);
    const auto& loop_condition_rank = loop_execution_condition.get_partial_shape().rank();
    if (loop_condition_rank.is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              loop_condition_rank.compatible(1) ||
                                  loop_condition_rank.compatible(0),
                              "Rank of ExecutionCondition input must be equal to 0 or 1");
    }
    if (const auto& cond_value = get_constant_from_source(loop_execution_condition))
    {
        auto val = cond_value->cast_vector<bool>();
        NODE_VALIDATION_CHECK(this,
                              val.size() == 1,
                              "The number of values in the Condition constant is greater than 1");

        if (!val[0])
        {
            zero_number_of_iter = true;
        }
    }

    bool condition_always_true = false;
    NODE_VALIDATION_CHECK(this,
                          m_special_body_ports.body_condition_output_idx >= 0,
                          "Condition body output is not provided. "
                          "Condition is a mandatory output of the body in Loop op.");
    const auto& body_execution_condition =
        m_body->get_results().at(m_special_body_ports.body_condition_output_idx)->input_value(0);
    const auto& body_condition_rank = body_execution_condition.get_partial_shape().rank();
    if (body_condition_rank.is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              body_condition_rank.compatible(0) ||
                                  body_condition_rank.compatible(1),
                              "Rank of BodyExecutionCondition output must be equal to 0 or 1");
    }
    if (const auto& cond_value = get_constant_from_source(body_execution_condition))
    {
        auto val = cond_value->cast_vector<bool>();
        NODE_VALIDATION_CHECK(this,
                              val.size() == 1,
                              "The number of values in the Condition constant is greater than 1");

        if (val[0])
        {
            condition_always_true = true;
        }
        else
        {
            m_num_iterations = 1; // condition_always_false, do_while mode
        }
    }
    else if (const auto& cond_param = std::dynamic_pointer_cast<const ngraph::opset5::Parameter>(
                 body_execution_condition.get_node_shared_ptr()))
    {
        // Const(true or false) -> Loop (body: Parameter -> execution_condition output)
        for (const auto& desc : get_input_descriptions())
        {
            if (m_body->get_parameters().at(desc->m_body_parameter_index) == cond_param)
            {
                if (const auto& cond_value =
                        get_constant_from_source(input_value(desc->m_input_index)))
                {
                    auto val = cond_value->cast_vector<bool>();
                    NODE_VALIDATION_CHECK(
                        this,
                        val.size() == 1,
                        "The number of values in the Condition constant is greater than 1");

                    if (val[0])
                    {
                        condition_always_true = true;
                    }
                    else
                    {
                        m_num_iterations = 1; // condition_always_false, do_while mode
                    }
                }
            }
        }
    }

    const auto& trip_count = input_value(0);
    const auto& trip_count_rank = trip_count.get_partial_shape().rank();
    if (trip_count_rank.is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              trip_count_rank.compatible(1) || trip_count_rank.compatible(0),
                              "Rank of TripCount input must be equal to 0 or 1");
    }
    if (const auto& trip_count_val = get_constant_from_source(trip_count))
    {
        auto val = trip_count_val->cast_vector<int64_t>();
        NODE_VALIDATION_CHECK(this,
                              val.size() == 1,
                              "The number of values in the TripCount constant is greater than 1");
        if (condition_always_true)
            m_num_iterations = val[0];
    }

    // WA: input description with index 0 or 1 means that Loop consructor will duplicate it in
    // the inputs.
    // When using visit_attributes() no duplication occurs, input_offset shall be decremented.
    size_t input_offset = 2;
    for (const auto& in_desc : m_input_descriptions)
    {
        if (in_desc->m_input_index == 0 || in_desc->m_input_index == 1)
        {
            input_offset--;
        }
    }
    // input_offset < 0 means that there are several duplications of external_port_id
    // (the same ext_port_id is connected to several Parameters in the port map) in input_desc,
    // this can lead to wrong or undefined behavior, so throw exception here. Ticket: 47302
    NODE_VALIDATION_CHECK(this, input_offset >= 0, "External port id 0 or 1 is duplicated.");

    NODE_VALIDATION_CHECK(this,
                          get_input_size() == m_input_descriptions.size() + input_offset,
                          "Number of inputs must be the same as number of input descriptions");

    // Input
    uint64_t index_it = input_offset;
    for (const auto& input_description : m_input_descriptions)
    {
        auto index = input_description->m_input_index;
        NODE_VALIDATION_CHECK(this, index == index_it, "Input_index not in order");
        index_it++;

        if (auto slice_input_description = as_type_ptr<SliceInputDescription>(input_description))
        {
            auto body_parameter =
                m_body->get_parameters().at(slice_input_description->m_body_parameter_index);
            const auto& input_partial_shape =
                inputs().at(index).get_source_output().get_partial_shape();
            if (input_partial_shape.rank().is_dynamic())
            {
                body_parameter->set_partial_shape(PartialShape::dynamic());
            }
            else
            {
                auto out_shape = input_partial_shape;
                const auto axis = ngraph::normalize_axis(
                    this, slice_input_description->m_axis, input_partial_shape.rank());
                out_shape[axis] = slice_input_description->m_part_size;
                body_parameter->set_partial_shape(out_shape);
            }
        }
        else if (auto merged_input_description =
                     as_type_ptr<MergedInputDescription>(input_description))
        {
            auto body_value =
                m_body->get_results().at(merged_input_description->m_body_value_index);

            auto body_parameter =
                m_body->get_parameters().at(merged_input_description->m_body_parameter_index);

            auto body_param_partial_shape = body_parameter->get_partial_shape();
            auto input_partial_shape = input(index).get_partial_shape();

            body_parameter->set_partial_shape(input_partial_shape);
        }
        else if (auto invariant_input_description =
                     as_type_ptr<TensorIterator::InvariantInputDescription>(input_description))
        {
            auto body_parameter =
                m_body->get_parameters().at(invariant_input_description->m_body_parameter_index);

            auto body_param_partial_shape = body_parameter->get_partial_shape();
            auto input_partial_shape = input(index).get_partial_shape();

            body_parameter->set_partial_shape(input_partial_shape);
        }
    }

    // Body
    m_body->validate_nodes_and_infer_types();

    // Output
    index_it = 0;
    for (const auto& output_description : m_output_descriptions)
    {
        auto index = output_description->m_output_index;
        NODE_VALIDATION_CHECK(this, index == index_it, "Output_index not in order");
        index_it++;

        auto body_value =
            m_body->get_results().at(output_description->m_body_value_index)->input_value(0);

        if (auto concat_output_description =
                as_type_ptr<TensorIterator::ConcatOutputDescription>(output_description))
        {
            const auto& body_value_partial_shape = body_value.get_partial_shape();
            auto out_shape = body_value_partial_shape;
            if (zero_number_of_iter)
            {
                out_shape = PartialShape{0};
            }
            else if (out_shape.rank().is_static())
            {
                const auto axis = ngraph::normalize_axis(
                    this, concat_output_description->m_axis, out_shape.rank());
                const auto rank = out_shape.rank().get_length();
                if (rank == 0)
                {
                    out_shape = PartialShape{1};
                }

                if (out_shape[axis].is_static() && m_num_iterations != -1)
                {
                    out_shape[axis] = Dimension{out_shape[axis].get_length() * m_num_iterations};
                }
                else
                {
                    out_shape[axis] = Dimension::dynamic();
                }
            }
            set_output_type(index, body_value.get_element_type(), out_shape);
        }

        else if (auto body_output_description =
                     as_type_ptr<TensorIterator::BodyOutputDescription>(output_description))
        {
            const PartialShape& ps = body_value.get_partial_shape();
            if (ps.is_dynamic())
            {
                set_output_type(index, body_value.get_element_type(), ps);
            }
            else
            {
                auto shape = ps.get_shape();
                if (zero_number_of_iter)
                {
                    shape.at(0) = 0;
                }
                set_output_type(index, body_value.get_element_type(), shape);
            }
        }
    }

    NODE_VALIDATION_CHECK(this,
                          get_output_size() == m_output_descriptions.size(),
                          "Number of outputs must be the same as number of output descriptions");
}

std::shared_ptr<Node> op::v5::Loop::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v5_Loop_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    auto op = make_shared<op::v5::Loop>();
    NGRAPH_CHECK(op.get(),
                 op != nullptr,
                 "Cannot clone ",
                 description(),
                 " operation with name ",
                 get_friendly_name());
    clone_to(*op, new_args);
    return op;
}

Output<Node> op::v5::Loop::get_concatenated_slices(const Output<Node>& value,
                                                   int64_t start,
                                                   int64_t stride,
                                                   int64_t part_size,
                                                   int64_t end,
                                                   int64_t axis)
{
    NGRAPH_CHECK(start == 0 && stride == 1 && part_size == 1 && end == -1,
                 "Invalid start, stride, part_size, or end attribute values in Loop op. "
                 "Supported values for start {0}, for stride and part_size {1}, for end "
                 "{-1}");
    return SubGraphOp::get_concatenated_slices(value, start, stride, part_size, end, axis);
}

bool op::v5::Loop::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v5_Loop_evaluate);
    runtime::reference::loop(
        m_body, m_output_descriptions, m_input_descriptions, m_special_body_ports, outputs, inputs);
    return true;
}

void op::v5::Loop::clone_to(op::v5::Loop& dst, const OutputVector& new_args) const
{
    dst.set_arguments(new_args);
    dst.set_output_size(m_output_descriptions.size());

    dst.m_num_iterations = m_num_iterations;
    dst.m_special_body_ports = m_special_body_ports;

    dst.m_body = clone_function(*get_function());

    for (auto& input_description : m_input_descriptions)
    {
        dst.m_input_descriptions.push_back(input_description->copy());
    }
    for (auto& output_description : m_output_descriptions)
    {
        dst.m_output_descriptions.push_back(output_description->copy());
    }
}

op::v5::Loop::Loop(const op::v5::Loop& other)
{
    other.clone_to(*this, other.input_values());
}

namespace ngraph
{
    constexpr DiscreteTypeInfo AttributeAdapter<op::v5::Loop::SpecialBodyPorts>::type_info;
}
