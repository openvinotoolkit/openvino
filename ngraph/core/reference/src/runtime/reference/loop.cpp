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

#include "runtime/reference/loop.hpp"
#include "runtime/reference/function.hpp"

namespace ngraph {
    namespace runtime {
        namespace reference {
            void loop(const std::shared_ptr<Function>& body,
                      const std::vector<std::shared_ptr<opset5::TensorIterator::OutputDescription>>& out_descs,
                      const std::vector<std::shared_ptr<opset5::TensorIterator::InputDescription>>& input_descs,
                      opset5::Loop::SpecialBodyPorts special_ports,
                      const std::vector<std::shared_ptr<HostTensor>> &out,
                      const std::vector<std::shared_ptr<HostTensor>> &args) {
                const auto &func = body;
                const auto &special_body_ports = special_ports;
                const auto &cur_iter_idx = special_body_ports.current_iteration_input_idx;

                // -2 due to trip_count and execution_condition inputs which aren't map to body inputs
                std::vector<std::vector<std::uint8_t>> inputs_to_body(
                        args.size() - 2 + (special_body_ports.current_iteration_input_idx >= 0));
                // param_idx, result_idx in the body
                std::vector<std::pair<uint64_t, uint64_t>> back_edges;

                // Port map : inputs and back edges
                for (const auto &desc : input_descs) {
                    auto *data_ptr = args[desc->m_input_index]->get_data_ptr<uint8_t>();
                    auto size_bytes = args[desc->m_input_index]->get_size_in_bytes();
                    inputs_to_body[desc->m_body_parameter_index].resize(size_bytes);
                    std::memcpy(inputs_to_body[desc->m_body_parameter_index].data(), data_ptr, size_bytes);
                    if (const auto &merged_desc = std::dynamic_pointer_cast<opset5::Loop::MergedInputDescription>(
                            desc)) {
                        back_edges.emplace_back(merged_desc->m_body_parameter_index, merged_desc->m_body_value_index);
                    }
                }

                if (cur_iter_idx >= 0) {
                    PartialShape pshape(Shape{1});
                    body->get_parameters()[cur_iter_idx]->set_partial_shape(pshape);
                    body->get_parameters()[cur_iter_idx]->set_element_type(ngraph::element::i64);
                    body->get_parameters()[cur_iter_idx]->validate_and_infer_types();
                }
                if (cur_iter_idx >= 0 && inputs_to_body.at(cur_iter_idx).empty()) {
                    // todo issue?
                    inputs_to_body.at(cur_iter_idx).resize(sizeof(int64_t));
                }

                auto type = out[0]->get_element_type(); // todo: check this, not sure about
                bool is_dynamic_shape = false;
                std::vector<std::vector<std::uint8_t>> outs;
                auto exec_condition = args[1]->get_data_ptr<bool>();

                int64_t trip_count = 0;
                if (args[0]->get_element_type() == ngraph::element::i32) {
                    auto *trip_count_p = args[0]->get_data_ptr<int32_t>();
                    trip_count = trip_count_p[0];
                } else if (args[0]->get_element_type() == ngraph::element::i64) {
                    auto *trip_count_p = args[0]->get_data_ptr<int64_t>();
                    trip_count = trip_count_p[0];
                } else {
                    // todo issue, not supported type
                }
                if (exec_condition[0]) {
                    for (int64_t cur_iter = 0; cur_iter < (trip_count >= 0 ? trip_count : std::numeric_limits<int64_t>::max()); ++cur_iter) {
                        // evaluate body
                        if (cur_iter_idx >= 0) {
                            // todo issue?
                            reinterpret_cast<int64_t *>(inputs_to_body.at(cur_iter_idx).data())[0] = cur_iter;
                        }
                        outs = reference::function(func, inputs_to_body);
                        // Port map: outputs
                        for (const auto &desc : out_descs) {
                            if (const auto &concat_desc = std::dynamic_pointer_cast<opset5::Loop::ConcatOutputDescription>(
                                    desc)) {
                                // if the output shape wasn't set during shape inference
                                if (out[concat_desc->m_output_index]->get_partial_shape().is_dynamic()) {
                                    auto cur_shape = func->get_results()[concat_desc->m_body_value_index]->get_shape();
                                    out[concat_desc->m_output_index]->set_shape(cur_shape);
                                    is_dynamic_shape = true;
                                } else if (is_dynamic_shape) {
                                    // increase the size of concat output
                                    auto cur_shape = out[concat_desc->m_output_index]->get_shape();
                                    auto el_size = out[concat_desc->m_output_index]->get_element_type().size();
                                    std::vector<uint8_t> tmp_buffer(ngraph::shape_size(cur_shape) * el_size);
                                    Shape old_shape = cur_shape;
                                    std::memcpy(tmp_buffer.data(), out[concat_desc->m_output_index]->get_data_ptr(),
                                                el_size * ngraph::shape_size(old_shape));
                                    cur_shape.at(concat_desc->m_axis) += 1;
                                    out[concat_desc->m_output_index]->set_shape(cur_shape);
                                    std::memcpy(out[concat_desc->m_output_index]->get_data_ptr(), tmp_buffer.data(),
                                                el_size * ngraph::shape_size(old_shape));
                                }
                                auto part_size = outs[concat_desc->m_body_value_index].size();
                                // copy the output from each iteration
                                std::memcpy(
                                        out[concat_desc->m_output_index]->get_data_ptr<uint8_t>() +
                                        cur_iter * part_size,
                                        outs[concat_desc->m_body_value_index].data(), part_size);
                            }
                        }
                        for (int i = 0; i < back_edges.size(); ++i) {
                            inputs_to_body[back_edges[i].first] = outs[back_edges[i].second];
                        }
                        bool *body_exec_condition = reinterpret_cast<bool *>(outs[special_body_ports.body_condition_output_idx].data());
                        if (!body_exec_condition[0])
                            break;
                    }

                    for (const auto &desc : out_descs) {
                        if (const auto &body_desc = std::dynamic_pointer_cast<opset5::Loop::BodyOutputDescription>(
                                desc)) {
                            // copy output values from the last iteration
                            std::memcpy(out[body_desc->m_output_index]->get_data_ptr(),
                                        outs[body_desc->m_body_value_index].data(),
                                        outs[body_desc->m_body_value_index].size());
                        }
                    }
                }
            }
        }
    }
}


