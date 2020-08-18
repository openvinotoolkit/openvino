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

NGRAPH_SUPPRESS_DEPRECATED_START

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
                    /// \brief      The termination condition input is not supported by
                    ///             TensorIterator now. If it is possible to determine
                    //              that termination condition is always true,
                    //              it can be ignored and execute using current version
                    //              of TensorIterator.
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
                    const Output<ngraph::Node> trip_count = ng_inputs.at(0);
                    const Output<ngraph::Node> loop_cond = ng_inputs.at(1);

                    // At this moment nGraph TensorIterator doesn't have support for conditional
                    // termination of iterations.
                    CHECK_VALID_NODE(node,
                                     !ngraph::op::is_null(trip_count),
                                     "Currently nGraph requires trip count input to be provided.");

                    const OutputVector loop_carried_dependencies{std::next(ng_inputs.begin(), 2),
                                                                 ng_inputs.end()};

                    // required
                    const Subgraph& body_graph{node.get_attribute_value<Subgraph>("body")};
                    const auto& graph_outputs = body_graph.get_ng_outputs();
                    const auto& graph_inputs = body_graph.get_ng_parameters();

                    CHECK_VALID_NODE(
                        node,
                        graph_inputs.size() == loop_carried_dependencies.size() + 2,
                        "The provided loop body graph inputs size (",
                        graph_inputs.size(),
                        "), is not equal to the sum of loop carried dependencies and two mandatory"
                        " inputs (",
                        loop_carried_dependencies.size() + 2,
                        ")");

                    CHECK_VALID_NODE(node,
                                     graph_outputs.size() >= loop_carried_dependencies.size() + 1,
                                     "The provided loop body graph outputs size (",
                                     graph_outputs.size(),
                                     ") has to small number of outpus. Required at least: ",
                                     loop_carried_dependencies.size() + 1);

                    const auto& body_loop_cond = graph_outputs.at(0).get_node_shared_ptr();
                    CHECK_VALID_NODE(
                        node,
                        is_termination_condition_always_true(loop_cond, body_loop_cond),
                        "Given termination loop condition input is not supported by Loop operator");

                    // TODO: Remove when loop condition would be supported.
                    const auto& cond_node =
                        default_opset::Constant::create(element::boolean, Shape{}, {true});

                    // create the loop body
                    const auto body = std::make_shared<ngraph::op::TensorIterator::BodyLambda>(
                        graph_outputs, graph_inputs);
                    auto tensor_iterator = std::make_shared<ngraph::op::TensorIterator>();
                    tensor_iterator->set_body(body);

                    // TensorIterator need to iterate over some input, thus we have to create
                    // 1 dim tensor with number of values equal to value provided by trip_count
                    // input.
                    const auto loop_trip_count = std::make_shared<default_opset::Range>(
                        default_opset::Constant::create(
                            trip_count.get_element_type(), Shape{}, {0}),
                        ngraph::onnx_import::reshape::interpret_as_scalar(trip_count),
                        default_opset::Constant::create(
                            trip_count.get_element_type(), Shape{}, {1}));

                    // We iterate over trip_count input.
                    // start=0, stride=1, part_size=1, end=-1, axis=0
                    tensor_iterator->set_sliced_input(
                        graph_inputs.at(0), loop_trip_count, 0, 1, 1, -1, 0);

                    // Set loop condition input, which should be changing over the iterations.
                    tensor_iterator->set_merged_input(
                        graph_inputs.at(1), cond_node, graph_outputs.at(0));

                    // Setting up other Loop body inputs.
                    auto graph_inputs_it = std::next(graph_inputs.begin(), 2);
                    auto graph_outputs_it = std::next(graph_outputs.begin(), 1);

                    // Set-up loop carried dependencies and final output values
                    OutputVector final_values;
                    for (const auto& dep : loop_carried_dependencies)
                    {
                        tensor_iterator->set_merged_input(
                            *graph_inputs_it++, dep, *graph_outputs_it);
                        final_values.push_back(
                            tensor_iterator->get_iter_value(*graph_outputs_it++, -1));
                    }

                    // Set-up scan outputs
                    OutputVector scan_outputs;
                    for (; graph_outputs_it != graph_outputs.end(); graph_outputs_it++)
                    {
                        // TODO: does concatenating along 0 axis is right?
                        // start=0, stride=1, part_size=1, end=-1, axis=0
                        scan_outputs.push_back(tensor_iterator->get_concatenated_slices(
                            *graph_outputs_it, 0, 1, 1, -1, 0));
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
