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
#include "loop.hpp"

#include <iterator>
#include <memory>

#include "ngraph/function.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "onnx_import/core/graph.hpp"
#include "onnx_import/core/null_node.hpp"
#include "onnx_import/default_opset.hpp"
#include "onnx_import/exceptions.hpp"
#include "onnx_import/utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                namespace
                {
                    /// \brief      Check if termination condition is true during all Loop
                    /// iterations.
                    ///             It allows to replace termination condition body output with
                    ///             Constant.
                    ///             As a result ngraph Loop shape iference is able to handle more
                    ///             cases.
                    ///
                    /// \param[in]  loop_cond       Termination loop condition input of Loop
                    ///                             operator (initial value).
                    /// \param[in]  body_cond       Termination loop condition input of the body of
                    ///                             the Loop (value updated during Loop iterations).
                    ///
                    /// \return true if termination condition is true and it cannot be changed
                    ///         during Loop iterations, false otherwise.
                    bool is_termination_condition_always_true(const Output<ngraph::Node>& loop_cond,
                                                              const Output<ngraph::Node>& body_cond)
                    {
                        bool loop_cond_value = false;
                        if (ngraph::op::is_constant(loop_cond.get_node()) &&
                            loop_cond.get_element_type() == element::boolean)
                        {
                            loop_cond_value = as_type_ptr<default_opset::Constant>(
                                                  loop_cond.get_node_shared_ptr())
                                                  ->cast_vector<bool>()
                                                  .at(0);
                        }
                        // According to ONNX skipped cond input (is_null) means
                        // that is has true value
                        bool is_loop_cond_true =
                            ngraph::op::is_null(loop_cond) || loop_cond_value == true;

                        if (!is_loop_cond_true)
                        {
                            return false;
                        }

                        // If body termination condition input matches Indentity op pattern the has
                        // value of loop_cond - true
                        // Identity op for boolean value is represented by LogicalOr op whose second
                        // input is always false
                        if (is_type<default_opset::LogicalOr>(body_cond.get_node_shared_ptr()))
                        {
                            const auto second_input = body_cond.get_node_shared_ptr()
                                                          ->input_value(1)
                                                          .get_node_shared_ptr();
                            if (ngraph::op::is_constant(second_input) &&
                                second_input->get_element_type() == element::boolean &&
                                as_type_ptr<default_opset::Constant>(second_input)
                                        ->cast_vector<bool>()
                                        .at(0) == false)
                            {
                                return true;
                            }
                        }
                        return false;
                    }
                }

                OutputVector loop(const Node& node)
                {
                    const auto& ng_inputs = node.get_ng_inputs();
                    // optional inputs
                    Output<ngraph::Node> trip_count;
                    if (ngraph::op::is_null(ng_inputs.at(0))) // trip count skipped
                    {
                        // -1 means infinite Loop
                        trip_count = ngraph::op::Constant::create(ngraph::element::i64, {}, {-1});
                    }
                    else
                    {
                        trip_count = ng_inputs.at(0);
                    }

                    Output<ngraph::Node> termination_cond;
                    if (ngraph::op::is_null(ng_inputs.at(1))) // termination condition skipped
                    {
                        // true means that first interation should be run
                        termination_cond =
                            ngraph::op::Constant::create(ngraph::element::boolean, {}, {true});
                    }
                    else
                    {
                        termination_cond = ng_inputs.at(1);
                    }

                    const OutputVector loop_carried_dependencies{std::next(ng_inputs.begin(), 2),
                                                                 ng_inputs.end()};

                    const Subgraph& body_graph{node.get_attribute_value<Subgraph>("body")};
                    auto body_outputs = body_graph.get_ng_outputs();
                    const auto& body_inputs = body_graph.get_ng_parameters();

                    const int64_t concat_axis = 0;
                    const auto concat_axis_const =
                        ngraph::op::Constant::create(ngraph::element::i64, {}, {concat_axis});
                    // provide scalar handing for scan outputs
                    for (int i = loop_carried_dependencies.size() + 1; i < body_outputs.size(); ++i)
                    {
                        auto body_output_shape = body_outputs[i].get_partial_shape();
                        if (body_output_shape.is_dynamic() ||
                            (body_output_shape.is_static() &&
                             ngraph::is_scalar(body_output_shape.to_shape())))
                        {
                            body_outputs[i] = std::make_shared<default_opset::Unsqueeze>(
                                body_outputs[i], concat_axis_const);
                        }
                    }

                    const auto& body_loop_cond = body_outputs.at(0).get_node_shared_ptr();
                    // optimization allow to improve nG Loop shape inference
                    if (is_termination_condition_always_true(termination_cond, body_loop_cond))
                    {
                        body_outputs[0] =
                            ngraph::op::Constant::create(ngraph::element::boolean, {}, {true});
                    }

                    CHECK_VALID_NODE(node,
                                     body_inputs.size() >= loop_carried_dependencies.size() + 2,
                                     "The provided loop body graph inputs size (",
                                     body_inputs.size(),
                                     "), is not greater than the sum of loop carried dependencies "
                                     "and two mandatory"
                                     " inputs (",
                                     loop_carried_dependencies.size() + 2,
                                     ")");

                    CHECK_VALID_NODE(node,
                                     body_outputs.size() >= loop_carried_dependencies.size() + 1,
                                     "The provided loop body graph outputs size (",
                                     body_outputs.size(),
                                     ") is not greater than number of outpus. Required at least: ",
                                     loop_carried_dependencies.size() + 1);

                    const auto body = std::make_shared<ngraph::Function>(body_outputs, body_inputs);
                    auto loop = std::make_shared<default_opset::Loop>(trip_count, termination_cond);
                    loop->set_function(body);
                    loop->set_special_body_ports(ngraph::opset5::Loop::SpecialBodyPorts{-1, 0});

                    // Setting up other Loop body inputs.
                    // body_inputs[0] is iteration number, body_inputs[1] is termination condition
                    auto body_inputs_it = std::next(body_inputs.begin(), 2);
                    // body_outputs[0] is termination condition output
                    auto body_outputs_it = std::next(body_outputs.begin(), 1);

                    // Set-up loop carried dependencies and final output values
                    OutputVector final_values;
                    for (const auto& dep : loop_carried_dependencies)
                    {
                        loop->set_merged_input(*body_inputs_it++, dep, *body_outputs_it);
                        final_values.push_back(loop->get_iter_value(*body_outputs_it++, -1));
                    }

                    // Set-up scan outputs
                    OutputVector scan_outputs;
                    for (; body_outputs_it != body_outputs.end(); body_outputs_it++)
                    {
                        // start=0, stride=1, part_size=1, end=-1, axis=0
                        scan_outputs.push_back(loop->get_concatenated_slices(
                            *body_outputs_it, 0, 1, 1, -1, concat_axis));
                    }

                    OutputVector node_outputs;
                    for (const auto& v : final_values)
                    {
                        node_outputs.push_back(v);
                    }
                    for (const auto& v : scan_outputs)
                    {
                        node_outputs.push_back(v);
                    }
                    return node_outputs;
                }
            } // namespace set_1
        }     // namespace op
    }         // namespace onnx_import
} // namespace ngraph
