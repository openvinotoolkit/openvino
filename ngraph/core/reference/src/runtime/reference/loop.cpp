// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/loop.hpp"
#include "ngraph/runtime/reference/concat.hpp"
#include "ngraph/runtime/reference/function.hpp"
#include "ngraph/runtime/reference/split.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void loop(const std::shared_ptr<Function>& func,
                      const op::util::OutputDescriptionVector& out_descs,
                      const op::util::InputDescriptionVector& input_descs,
                      const opset5::Loop::SpecialBodyPorts& special_ports,
                      const HostTensorVector& out,
                      const HostTensorVector& args)
            {
                const auto& cur_iter_idx = special_ports.current_iteration_input_idx;
                auto val =
                    std::find_if(input_descs.begin(),
                                 input_descs.end(),
                                 [&cur_iter_idx](const op::util::InputDescriptionPtr& in_desc) {
                                     return in_desc->m_body_parameter_index ==
                                            static_cast<uint64_t>(cur_iter_idx);
                                 });
                bool cur_iter_initial_value_exist = val != input_descs.end();
                bool cur_iter_back_edge_exist = false;

                // If current_iteration_input is exist and initial value is not provided, we
                // should allocate input_descs.size() + 1 inputs and set default value (0) for
                // current_iteration input.
                int64_t inputs_count =
                    input_descs.size() + (cur_iter_idx >= 0 ? !cur_iter_initial_value_exist : 0);
                HostTensorVector inputs_to_body;
                for (int64_t i = 0; i < inputs_count; ++i)
                    inputs_to_body.push_back(
                        std::make_shared<HostTensor>(element::dynamic, PartialShape::dynamic()));
                if (cur_iter_idx >= 0 && !cur_iter_initial_value_exist)
                {
                    const auto& cur_iter = func->get_parameters().at(cur_iter_idx);
                    if (cur_iter->get_partial_shape().is_dynamic())
                    {
                        cur_iter->set_partial_shape(Shape{1});
                        cur_iter->validate_and_infer_types();
                    }

                    auto init = std::make_shared<opset5::Constant>(
                        func->get_parameters().at(cur_iter_idx)->get_element_type(),
                        func->get_parameters().at(cur_iter_idx)->get_shape(),
                        0);
                    inputs_to_body.at(cur_iter_idx)->initialize(init);
                    // reinterpret_cast<int64_t*>(inputs_to_body.at(cur_iter_idx).data())[0] = 0;
                }

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
                        cur_iter_back_edge_exist |= merged_desc->m_body_parameter_index ==
                                                    static_cast<uint64_t>(cur_iter_idx);
                    }
                }

                // Get TripCount
                int64_t trip_count = 0;
                if (args[0]->get_element_type() == ngraph::element::i32)
                {
                    auto* trip_count_p = args[0]->get_data_ptr<int32_t>();
                    trip_count = trip_count_p[0];
                }
                else if (args[0]->get_element_type() == ngraph::element::i64)
                {
                    auto* trip_count_p = args[0]->get_data_ptr<int64_t>();
                    trip_count = trip_count_p[0];
                }
                else
                {
                    NGRAPH_CHECK(
                        false,
                        "Unsupported element type for trip_count input. Expected int32 or int64.");
                }
                NGRAPH_CHECK(trip_count != 0, "Zero count of iteration not supported");

                // Loop iterations
                auto exec_condition = args[1]->get_data_ptr<bool>();
                if (exec_condition[0])
                {
                    // Find all ConcatOutputDescription
                    std::vector<std::shared_ptr<opset5::Loop::ConcatOutputDescription>>
                        concat_outputs;
                    for (const auto& desc : out_descs)
                    {
                        if (const auto& concat_desc =
                                std::dynamic_pointer_cast<opset5::Loop::ConcatOutputDescription>(
                                    desc))
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
                            uint64_t num_iterations = shape.at(slice_desc->m_axis);
                            shape.at(slice_desc->m_axis) = 1;
                            sliced_values.emplace_back(HostTensorVector());
                            for (uint64_t i = 0; i < num_iterations; ++i)
                            {
                                sliced_values.back().emplace_back(std::make_shared<HostTensor>(
                                    args[slice_desc->m_input_index]->get_element_type(), shape));
                            }
                            std::vector<char*> pointers_to_data(num_iterations);
                            for (size_t j = 0; j < pointers_to_data.size(); ++j)
                            {
                                pointers_to_data[slice_desc->m_stride > 0
                                                     ? j
                                                     : (pointers_to_data.size() - j - 1)] =
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

                    // Negative value means infinity count of iterations
                    trip_count = trip_count >= 0 ? trip_count : std::numeric_limits<int64_t>::max();
                    for (int64_t cur_iter = 0; cur_iter < trip_count; ++cur_iter)
                    {
                        // Copy new values for sliced inputs
                        for (size_t i = 0; i < slice_inputs.size(); ++i)
                        {
                            if (static_cast<int64_t>(sliced_values[i].size()) > cur_iter)
                                inputs_to_body[slice_inputs[i]->m_body_parameter_index] =
                                    sliced_values[i][cur_iter];
                        }

                        // Evaluate body
                        body_outputs.clear();
                        reference::function(func, inputs_to_body, body_outputs);

                        // Store values for later concatenation
                        for (size_t i = 0; i < values_to_concat.size(); ++i)
                        {
                            values_to_concat[i].push_back(
                                body_outputs[concat_outputs[i]->m_body_value_index]);
                        }

                        // Check execution condition
                        bool body_exec_condition(false);
                        if (static_cast<int64_t>(body_outputs.size()) >
                                special_ports.body_condition_output_idx &&
                            body_outputs[special_ports.body_condition_output_idx])
                            body_outputs[special_ports.body_condition_output_idx]->read(
                                &body_exec_condition, sizeof(bool));
                        if (!body_exec_condition)
                            break;

                        // If there are no rules for calculating the current iteration, just
                        // increment it.
                        if (cur_iter_idx >= 0 && !cur_iter_back_edge_exist)
                        {
                            const auto& cur_iter_param = func->get_parameters().at(cur_iter_idx);
                            int64_t iter_num = cur_iter + 1;
                            if (cur_iter_param->get_element_type() == element::i64)
                                inputs_to_body.at(cur_iter_idx)
                                    ->write(&iter_num, cur_iter_param->get_element_type().size());
                            else if (cur_iter_param->get_element_type() == element::i32)
                            {
                                int32_t iter_num_i32 = static_cast<int32_t>(iter_num);
                                inputs_to_body.at(cur_iter_idx)
                                    ->write(&iter_num_i32,
                                            cur_iter_param->get_element_type().size());
                            }
                            else
                                NGRAPH_CHECK(false,
                                             "Unsupported element type for current iteration "
                                             "input. Expected int32 or int64.");
                        }

                        // Back-edge processing
                        for (auto& back_edge : back_edges)
                        {
                            inputs_to_body[back_edge.param_idx] =
                                body_outputs[back_edge.result_idx];
                        }
                    }

                    for (const auto& desc : out_descs)
                    {
                        if (const auto& body_desc =
                                std::dynamic_pointer_cast<opset5::Loop::BodyOutputDescription>(
                                    desc))
                        {
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
                        reference::concat(
                            pointers_on_values,
                            out[concat_desc->m_output_index]->get_data_ptr<char>(),
                            shapes_to_concat,
                            shape,
                            concat_desc->m_axis,
                            out[concat_desc->m_output_index]->get_element_type().size());
                    }
                }
                else
                {
                    NGRAPH_CHECK(
                        false,
                        "ExecutionCondition is false. Zero count of iteration not supported.");
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
