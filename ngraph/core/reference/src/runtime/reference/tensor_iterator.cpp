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

#include "runtime/reference/tensor_iterator.hpp"
#include "runtime/reference/concat.hpp"
#include "runtime/reference/function.hpp"
#include "runtime/reference/split.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void tensor_iterator(uint64_t num_iterations,
                                 const std::shared_ptr<Function>& func,
                                 const op::util::OutputDescriptionVector& out_descs,
                                 const op::util::InputDescriptionVector& input_descs,
                                 const HostTensorVector& out,
                                 const HostTensorVector& args,
                                 const custom_evaluate_function& evaluate)
            {
                HostTensorVector inputs_to_body;
                for (int64_t i = 0; i < input_descs.size(); ++i)
                    inputs_to_body.push_back(
                        std::make_shared<HostTensor>(element::dynamic, PartialShape::dynamic()));

                // Port map processing: inputs and back edges
                struct BackEdge
                {
                    uint64_t param_idx;
                    uint64_t result_idx;
                };
                std::vector<BackEdge> back_edges;
                for (const auto& desc : input_descs)
                {
                    inputs_to_body[desc->m_body_parameter_index] = args[desc->m_input_index];
                    if (const auto& merged_desc =
                            std::dynamic_pointer_cast<opset5::Loop::MergedInputDescription>(desc))
                    {
                        back_edges.push_back(
                            {merged_desc->m_body_parameter_index, merged_desc->m_body_value_index});
                    }
                }
                // Find all ConcatOutputDescription
                std::vector<std::shared_ptr<opset5::TensorIterator::ConcatOutputDescription>>
                    concat_outputs;
                for (const auto& desc : out_descs)
                {
                    if (const auto& concat_desc = std::dynamic_pointer_cast<
                            opset5::TensorIterator::ConcatOutputDescription>(desc))
                    {
                        concat_outputs.push_back(concat_desc);
                    }
                }

                // Slicing
                std::vector<std::shared_ptr<opset5::TensorIterator::SliceInputDescription>>
                    slice_inputs;
                std::vector<HostTensorVector> sliced_values;
                int slice_in_idx = 0;
                for (const auto& desc : input_descs)
                {
                    if (const auto& slice_desc = std::dynamic_pointer_cast<
                            opset5::TensorIterator::SliceInputDescription>(desc))
                    {
                        const auto el_size =
                            args[slice_desc->m_input_index]->get_element_type().size();
                        slice_inputs.push_back(slice_desc);
                        auto shape = args[slice_desc->m_input_index]->get_shape();
                        shape.at(slice_desc->m_axis) = 1;
                        sliced_values.emplace_back(HostTensorVector());
                        for (int i = 0; i < num_iterations; ++i)
                        {
                            sliced_values.back().emplace_back(std::make_shared<HostTensor>(
                                args[slice_desc->m_input_index]->get_element_type(), shape));
                        }
                        std::vector<char*> pointers_to_data(num_iterations);
                        for (size_t j = 0; j < pointers_to_data.size(); ++j)
                        {
                            pointers_to_data[j] =
                                sliced_values[slice_in_idx][j]->get_data_ptr<char>();
                        }
                        reference::split(args[slice_desc->m_input_index]->get_data_ptr<char>(),
                                         args[slice_desc->m_input_index]->get_shape(),
                                         el_size,
                                         slice_desc->m_axis,
                                         num_iterations,
                                         pointers_to_data.data());
                        slice_in_idx++;
                    }
                }

                // Allocate vectors for store output values
                std::vector<HostTensorVector> values_to_concat(concat_outputs.size());
                HostTensorVector body_outputs;

                for (int64_t cur_iter = 0; cur_iter < num_iterations; ++cur_iter)
                {
                    // Copy new values for sliced inputs
                    for (size_t i = 0; i < slice_inputs.size(); ++i)
                    {
                        inputs_to_body[slice_inputs[i]->m_body_parameter_index] =
                            sliced_values[i][cur_iter];
                    }

                    // Evaluate body
                    if (!evaluate)
                    {
                        reference::function(func, inputs_to_body, body_outputs);
                    }
                    else
                    {
                        evaluate(func, inputs_to_body, body_outputs);
                    }

                    // Store values for later concatenation
                    for (size_t i = 0; i < values_to_concat.size(); ++i)
                    {
                        values_to_concat[i].push_back(
                            body_outputs[concat_outputs[i]->m_body_value_index]);
                    }

                    // Back-edge processing
                    for (auto& back_edge : back_edges)
                    {
                        inputs_to_body[back_edge.param_idx] = body_outputs[back_edge.result_idx];
                    }
                }

                for (const auto& desc : out_descs)
                {
                    if (const auto& body_desc = std::dynamic_pointer_cast<
                            opset5::TensorIterator::BodyOutputDescription>(desc))
                    {
                        // Copy output values from the last iteration
                        out[body_desc->m_output_index]->write(
                            body_outputs[body_desc->m_body_value_index]->get_data_ptr(),
                            body_outputs[body_desc->m_body_value_index]->get_size_in_bytes());
                    }
                }

                // Concatenate and copy all values stored in values_to_concat vector to outputs
                for (size_t i = 0; i < concat_outputs.size(); ++i)
                {
                    const auto& concat_desc = concat_outputs[i];
                    auto shape =
                        func->get_results().at(concat_desc->m_body_value_index)->get_shape();
                    std::vector<Shape> shapes_to_concat(values_to_concat[i].size(), shape);
                    shape.at(concat_desc->m_axis) = values_to_concat[i].size();
                    out[concat_desc->m_output_index]->set_shape(shape);
                    std::vector<const char*> pointers_on_values;
                    pointers_on_values.reserve(values_to_concat[i].size());
                    for (const auto& vec : values_to_concat[i])
                    {
                        pointers_on_values.push_back(vec->get_data_ptr<char>());
                    }
                    reference::concat(pointers_on_values,
                                      out[concat_desc->m_output_index]->get_data_ptr<char>(),
                                      shapes_to_concat,
                                      shape,
                                      concat_desc->m_axis,
                                      out[concat_desc->m_output_index]->get_element_type().size());
                }
            }
        }
    }
}
