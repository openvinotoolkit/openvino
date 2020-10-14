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

#include "ngraph/op/loop.hpp"
#include "ngraph/factory.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "ngraph/specialize_function.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v5::Loop::type_info;

op::v5::Loop::Loop(const Output<Node>& trip_count,
                   const Output<Node>& condition,
                   const OutputVector& values)
    : op::util::SubGraphOp({trip_count, condition})
{
    set_arguments(values);
}

bool op::v5::Loop::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("body", m_body);
    visitor.on_attribute("input_descriptions", m_input_descriptions);
    visitor.on_attribute("output_descriptions", m_output_descriptions);

    return false;
}

void op::v5::Loop::validate_and_infer_types()
{
    bool zero_number_of_iter = false;
    const auto& loop_condition = input_value(1);
    if (const auto& cond_value = std::dynamic_pointer_cast<const ngraph::opset5::Constant>(
            loop_condition.get_node_shared_ptr()))
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
    const auto& body_condition =
        m_body->get_results()[m_special_body_ports.body_condition_output_idx]->input_value(0);
    if (const auto& cond_value = std::dynamic_pointer_cast<const ngraph::opset5::Constant>(
            body_condition.get_node_shared_ptr()))
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

    const auto& trip_count = input_value(0);
    if (const auto& trip_count_val = std::dynamic_pointer_cast<const ngraph::opset5::Constant>(
            trip_count.get_node_shared_ptr()))
    {
        auto val = trip_count_val->cast_vector<int64_t>();
        NODE_VALIDATION_CHECK(this,
                              val.size() == 1,
                              "The number of values in the TripCount constant is greater than 1");
        if (condition_always_true)
            m_num_iterations = val[0];
    }

    NODE_VALIDATION_CHECK(this,
                          get_input_size() == m_input_descriptions.size() + 2,
                          "Number of inputs must be the same as number of input descriptions");

    NODE_VALIDATION_CHECK(this,
                          get_output_size() == m_output_descriptions.size(),
                          "Number of outputs must be the same as number of output descriptions");

    std::vector<std::shared_ptr<Node>> ends;

    // Input
    uint64_t index_it = 2;
    for (const auto& input_description : m_input_descriptions)
    {
        auto index = input_description->m_input_index;
        NODE_VALIDATION_CHECK(this, index == index_it, "Input_index not in order");
        index_it++;

        if (auto merged_input_description = as_type_ptr<MergedInputDescription>(input_description))
        {
            auto body_value =
                m_body->get_results().at(merged_input_description->m_body_value_index)->input(0);
            ends.push_back(body_value.get_node()->shared_from_this());

            const auto& body_value_partial_shape = body_value.get_partial_shape();
            auto body_parameter =
                m_body->get_parameters().at(merged_input_description->m_body_parameter_index);

            auto body_param_partial_shape = body_parameter->get_partial_shape();
            auto input_partial_shape = inputs().at(index).get_source_output().get_partial_shape();
            NODE_VALIDATION_CHECK(this,
                                  body_value_partial_shape.compatible(body_param_partial_shape),
                                  "Iterator successive value is not compatible with body param");
            NODE_VALIDATION_CHECK(this,
                                  input_partial_shape.compatible(body_param_partial_shape),
                                  "Iterator initial value is not compatible with body param");

            if (input_partial_shape.is_static())
            {
                auto input_shape = input_partial_shape.to_shape();
                // infer type for body_parameter
                if (body_param_partial_shape.is_dynamic())
                {
                    body_parameter->set_partial_shape(input_shape);
                }
            }
        }
        else if (auto invariant_input_description =
                     as_type_ptr<TensorIterator::InvariantInputDescription>(input_description))
        {
            auto body_parameter =
                m_body->get_parameters().at(invariant_input_description->m_body_parameter_index);

            auto body_param_partial_shape = body_parameter->get_partial_shape();
            auto input_partial_shape = inputs().at(index).get_source_output().get_partial_shape();
            NODE_VALIDATION_CHECK(this,
                                  input_partial_shape.compatible(body_param_partial_shape),
                                  "Iterator initial value is not compatible with body param");

            if (input_partial_shape.is_static())
            {
                auto input_shape = input_partial_shape.to_shape();
                // infer type for m_body_parameter
                if (body_param_partial_shape.is_dynamic())
                {
                    body_parameter->set_partial_shape(input_shape);
                }
            }
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
            set_output_type(index, body_value.get_element_type(), PartialShape::dynamic());
            if (body_value_partial_shape.is_static())
            {
                auto body_value_shape = body_value_partial_shape.to_shape();
                auto part_size = concat_output_description->m_part_size;
                auto axis = concat_output_description->m_axis;

                Shape out_shape{body_value_shape};

                if (body_value_shape.empty())
                {
                    NODE_VALIDATION_CHECK(
                        this,
                        axis == 0,
                        "Axis must be equal to 0 if concatenated output tensor slices are scalars. "
                        "Loop output index: ",
                        index);
                    out_shape = Shape(1);
                }

                if (m_num_iterations != -1)
                {
                    // for simple RNN case where stride is the same as part_size
                    out_shape[axis] = m_num_iterations * part_size;
                    if (zero_number_of_iter)
                    {
                        out_shape.at(0) = 0;
                    }
                    set_output_type(index, body_value.get_element_type(), out_shape);
                }
            }
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
}

std::shared_ptr<Node> op::v5::Loop::clone_with_new_inputs(const OutputVector& new_args) const
{
    // 0 - trip_count, 1 - execution condition, these inputs are not connected to the body params
    const OutputVector body_params_args(new_args.begin() + 2, new_args.end());
    auto op = make_shared<op::v5::Loop>(new_args[0], new_args[1], body_params_args);
    NGRAPH_CHECK(op.get(),
                 op != nullptr,
                 "Cannot clone ",
                 description(),
                 " operation with name ",
                 get_friendly_name());
    op->set_output_size(m_output_descriptions.size());

    std::vector<::ngraph::element::Type> types(m_body->get_parameters().size());
    std::vector<::ngraph::PartialShape> new_shapes(m_body->get_parameters().size());

    for (size_t input_index = 0; input_index < new_args.size(); ++input_index)
    {
        for (auto& input_description : m_input_descriptions)
        {
            if (input_description->m_input_index == input_index)
            {
                types[input_description->m_body_parameter_index] =
                    new_args[input_index].get_element_type();
                new_shapes[input_description->m_body_parameter_index] =
                    new_args[input_index].get_partial_shape();
            }
        }
    }

    op->m_num_iterations = m_num_iterations;
    auto func = std::make_shared<Function>(m_body->get_results(), m_body->get_parameters());
    auto spec_func = specialize_function(
        func, types, new_shapes, std::vector<void*>(body_params_args.size(), nullptr));
    op->m_body = std::make_shared<Function>(spec_func->get_results(), spec_func->get_parameters());

    for (auto& input_description : m_input_descriptions)
    {
        op->m_input_descriptions.push_back(input_description->copy());
    }
    for (auto& output_description : m_output_descriptions)
    {
        op->m_output_descriptions.push_back(output_description->copy());
    }
    return move(op);
}
