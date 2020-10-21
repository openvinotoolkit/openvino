/*
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

#include "ngraph/runtime/reference/loop.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void loop(ngraph::opset5::Loop& loop,
                      const std::vector<std::shared_ptr<HostTensor>>& out,
                      const std::vector<std::shared_ptr<HostTensor>>& args)
            {
                */
/*const auto& func = loop.get_function();
                const auto& special_body_ports = loop.get_special_body_ports();
                const auto& cur_iter_idx = special_body_ports.current_iteration_input_idx;

                // -2 due to trip_count and execution_condition inputs which aren't map to body inputs
                std::vector<std::vector<std::uint8_t>> inputs_to_body(args.size() - 2);
                // param_idx, result_idx in the body
                std::vector<std::pair<uint64_t, uint64_t>> back_edges;

                // Port map : inputs and back edges
                for (const auto& desc : loop.get_input_descriptions()) {
                    auto* data_ptr = args[desc->m_input_index]->get_data_ptr<uint8_t>();
                    auto size_bytes = args[desc->m_input_index]->get_size_in_bytes();
                    inputs_to_body[desc->m_body_parameter_index].resize(size_bytes);
                    std::memcpy(inputs_to_body[desc->m_body_parameter_index].data(), data_ptr, size_bytes);
                    if (const auto &merged_desc = std::dynamic_pointer_cast<opset5::Loop::MergedInputDescription>(desc)) {
                        back_edges.emplace_back(merged_desc->m_body_parameter_index, merged_desc->m_body_value_index);
                    }
                }

                if (cur_iter_idx >= 0 && inputs_to_body.at(cur_iter_idx).empty()) {
                    // todo issue?
                }

                auto type = out[0]->get_element_type(); // todo: check this, not sure about
                bool is_dynamic_shape = false;
                std::vector<std::vector<std::uint8_t>> outs;
                auto exec_condition = args[1]->get_data_ptr<bool>();

                int64_t trip_count = 0;
                if (args[0]->get_element_type() == ngraph::element::i32) {
                    auto* trip_count_p = args[0]->get_data_ptr<int32_t>();
                    trip_count = trip_count_p[0];
                } else if (args[0]->get_element_type() == ngraph::element::i64) {
                    auto* trip_count_p = args[0]->get_data_ptr<int64_t>();
                    trip_count = trip_count_p[0];
                } else {
                    // todo issue, not supported type
                }
                if(exec_condition[0]) {
                    for (int64_t cur_iter = 0; cur_iter < trip_count; ++cur_iter) {
                        // evaluate body
                        outs = interpreterFunction(func, inputs_to_body, type);

                        // Port map: outputs
                        for (const auto &desc : loop.get_output_descriptions()) {
                            if (const auto &concat_desc = std::dynamic_pointer_cast<opset5::Loop::ConcatOutputDescription>(
                                    desc)) {
                                // if the output shape wasn't set during shape inference
                                if (out[concat_desc->m_output_index]->get_partial_shape().is_dynamic()) {
                                    auto cur_shape = out[concat_desc->m_output_index]->get_shape();
                                    out[concat_desc->m_output_index]->set_shape(cur_shape);
                                    is_dynamic_shape = true;
                                } else if (is_dynamic_shape) {
                                    // increase the size of concat output
                                    auto cur_shape = out[concat_desc->m_output_index]->get_shape();
                                    cur_shape.at(concat_desc->m_axis) += 1;
                                    out[concat_desc->m_output_index]->set_shape(cur_shape);
                                }

                                auto part_size = outs[concat_desc->m_body_value_index].size();
                                // copy the output from each iteration
                                std::memcpy(out[concat_desc->m_output_index]->get_data_ptr<uint8_t>() + cur_iter * part_size,
                                            outs[concat_desc->m_body_value_index].data(), part_size);
                            }

                            for (auto &back_edge : back_edges) {
                                if (back_edge.second == desc->m_body_value_index) {
                                    inputs_to_body[back_edge.first] = outs[desc->m_body_value_index];
                                }
                            }
                        }
                        bool* body_exec_condition = reinterpret_cast<bool*>(outs[special_body_ports.body_condition_output_idx].data());
                        if (!body_exec_condition[0])
                            break;
                    }

                    for (const auto &desc : loop.get_output_descriptions()) {
                        if (const auto &body_desc = std::dynamic_pointer_cast<opset5::Loop::BodyOutputDescription>(
                                desc)) {
                            // copy output values from the last iteration
                            std::memcpy(out[body_desc->m_output_index]->get_data_ptr(),
                                        outs[body_desc->m_body_value_index].data(),
                                        outs[body_desc->m_body_value_index].size());
                        }
                    }
                }*/ /*

             }
         }

         template<typename fromPrec, typename toPrec>
         std::vector<std::uint8_t>
         convertPrecision(std::vector<std::uint8_t> &buffer, const size_t elementsCount, const
 size_t elementSize) {
             std::vector<std::uint8_t> convertedData(elementsCount * elementSize);
             const fromPrec *src = reinterpret_cast<const fromPrec *>(buffer.data());
             toPrec *dst = reinterpret_cast<toPrec *>(convertedData.data());
             for (size_t i = 0; i < elementsCount; i++)
                 dst[i] = static_cast<toPrec>(src[i]);
             return convertedData;
         }

         std::vector<std::uint8_t>
         convertOutputPrecision(std::vector<std::uint8_t> &output, const element::Type_t
 &fromPrecision,
                                const element::Type_t &toPrecision, const size_t elementsCount) {
             switch (fromPrecision) {
                 case element::Type_t::u8: {
                     switch (toPrecision) {
                         case element::Type_t::u8: {
                             return convertPrecision<uint8_t, uint8_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::u16: {
                             return convertPrecision<uint8_t, uint16_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i8: {
                             return convertPrecision<uint8_t, int8_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i16: {
                             return convertPrecision<uint8_t, int16_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i32: {
                             return convertPrecision<uint8_t, int32_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i64: {
                             return convertPrecision<uint8_t, int64_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::f32: {
                             return convertPrecision<uint8_t, float>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::u64: {
                             return convertPrecision<uint8_t, uint64_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         default:
                             throw std::runtime_error("convertOutputPrecision can't convert from: "
 + element::Type(fromPrecision).get_type_name() + " to: " +
                                                      element::Type(toPrecision).get_type_name());
                     }
                 }
                 case element::Type_t::u16: {
                     switch (toPrecision) {
                         case element::Type_t::u8: {
                             return convertPrecision<uint16_t, uint8_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::u16: {
                             return convertPrecision<uint16_t, uint16_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i8: {
                             return convertPrecision<uint16_t, int8_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i16: {
                             return convertPrecision<uint16_t, int16_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i32: {
                             return convertPrecision<uint16_t, int32_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i64: {
                             return convertPrecision<uint16_t, int64_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::f32: {
                             return convertPrecision<uint16_t, float>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::u64: {
                             return convertPrecision<uint16_t, uint64_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         default:
                             throw std::runtime_error("convertOutputPrecision can't convert from: "
 + element::Type(fromPrecision).get_type_name() + " to: " +
                                                      element::Type(toPrecision).get_type_name());
                     }
                 }
                 case element::Type_t::i8: {
                     switch (toPrecision) {
                         case element::Type_t::u8: {
                             return convertPrecision<int8_t, uint8_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::u16: {
                             return convertPrecision<int8_t, uint16_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i8: {
                             return convertPrecision<int8_t, int8_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i16: {
                             return convertPrecision<int8_t, int16_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i32: {
                             return convertPrecision<int8_t, int32_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i64: {
                             return convertPrecision<int8_t, int64_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::f32: {
                             return convertPrecision<int8_t, float>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::u64: {
                             return convertPrecision<int8_t, uint64_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         default:
                             throw std::runtime_error("convertOutputPrecision can't convert from: "
 + element::Type(fromPrecision).get_type_name() + " to: " +
                                                      element::Type(toPrecision).get_type_name());
                     }
                 }
                 case element::Type_t::i16: {
                     switch (toPrecision) {
                         case element::Type_t::u8: {
                             return convertPrecision<int16_t, uint8_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::u16: {
                             return convertPrecision<int16_t, uint16_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i8: {
                             return convertPrecision<int16_t, int8_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i16: {
                             return convertPrecision<int16_t, int16_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i32: {
                             return convertPrecision<int16_t, int32_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i64: {
                             return convertPrecision<int16_t, int64_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::f32: {
                             return convertPrecision<int16_t, float>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::u64: {
                             return convertPrecision<int16_t, uint64_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         default:
                             throw std::runtime_error("convertOutputPrecision can't convert from: "
 + element::Type(fromPrecision).get_type_name() + " to: " +
                                                      element::Type(toPrecision).get_type_name());
                     }
                 }
                 case element::Type_t::i32: {
                     switch (toPrecision) {
                         case element::Type_t::u8: {
                             return convertPrecision<int32_t, uint8_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::u16: {
                             return convertPrecision<int32_t, uint16_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i8: {
                             return convertPrecision<int32_t, int8_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i16: {
                             return convertPrecision<int32_t, int16_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i32: {
                             return convertPrecision<int32_t, int32_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i64: {
                             return convertPrecision<int32_t, int64_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::f32: {
                             return convertPrecision<int32_t, float>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::u64: {
                             return convertPrecision<int32_t, uint64_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         default:
                             throw std::runtime_error("convertOutputPrecision can't convert from: "
 + element::Type(fromPrecision).get_type_name() + " to: " +
                                                      element::Type(toPrecision).get_type_name());
                     }
                 }
                 case element::Type_t::i64: {
                     switch (toPrecision) {
                         case element::Type_t::u8: {
                             return convertPrecision<int64_t, uint8_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::u16: {
                             return convertPrecision<int64_t, uint16_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i8: {
                             return convertPrecision<int64_t, int8_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i16: {
                             return convertPrecision<int64_t, int16_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i32: {
                             return convertPrecision<int64_t, int32_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i64: {
                             return convertPrecision<int64_t, int64_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::f32: {
                             return convertPrecision<int64_t, float>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::u64: {
                             return convertPrecision<int64_t, uint64_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         default:
                             throw std::runtime_error("convertOutputPrecision can't convert from: "
 + element::Type(fromPrecision).get_type_name() + " to: " +
                                                      element::Type(toPrecision).get_type_name());
                     }
                 }
                 case element::Type_t::u64: {
                     switch (toPrecision) {
                         case element::Type_t::u8: {
                             return convertPrecision<uint64_t, uint8_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::u16: {
                             return convertPrecision<uint64_t, uint16_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i8: {
                             return convertPrecision<uint64_t, int8_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i16: {
                             return convertPrecision<uint64_t, int16_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i32: {
                             return convertPrecision<uint64_t, int32_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i64: {
                             return convertPrecision<uint64_t, int64_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::f32: {
                             return convertPrecision<uint64_t, float>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::u64: {
                             return convertPrecision<uint64_t, uint64_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         default:
                             throw std::runtime_error("convertOutputPrecision can't convert from: "
 + element::Type(fromPrecision).get_type_name() + " to: " +
                                                      element::Type(toPrecision).get_type_name());
                     }
                 }
                 case element::Type_t::f32: {
                     switch (toPrecision) {
                         case element::Type_t::u8: {
                             return convertPrecision<float, uint8_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::u16: {
                             return convertPrecision<float, uint16_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i8: {
                             return convertPrecision<float, int8_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i16: {
                             return convertPrecision<float, int16_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i32: {
                             return convertPrecision<float, int32_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i64: {
                             return convertPrecision<float, int64_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::f32: {
                             return convertPrecision<float, float>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::f16: {
                             // ngraph float16 has single ctor from float
                             return convertPrecision<float, ngraph::float16>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::u64: {
                             return convertPrecision<float, uint64_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         default:
                             throw std::runtime_error("convertOutputPrecision can't convert from: "
 + element::Type(fromPrecision).get_type_name() + " to: " +
                                                      element::Type(toPrecision).get_type_name());
                     }
                 }
                 case element::Type_t::boolean: {
                     switch (toPrecision) {
                         case element::Type_t::u8: {
                             return convertPrecision<bool, uint8_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::u16: {
                             return convertPrecision<bool, uint16_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i8: {
                             return convertPrecision<bool, int8_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i16: {
                             return convertPrecision<bool, int16_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i32: {
                             return convertPrecision<bool, int32_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::i64: {
                             return convertPrecision<bool, int64_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::f32: {
                             return convertPrecision<bool, float>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         case element::Type_t::u64: {
                             return convertPrecision<bool, uint64_t>(output, elementsCount,
 element::Type(toPrecision).size());
                         }
                         default:
                             throw std::runtime_error("convertOutputPrecision can't convert from: "
 + element::Type(fromPrecision).get_type_name() + " to: " +
                                                      element::Type(toPrecision).get_type_name());
                     }
                 }
                 default:
                     throw std::runtime_error("convertOutputPrecision can't convert from: " +
 element::Type(fromPrecision).get_type_name() + " precision");
             }
         }

         std::vector<std::vector<std::uint8_t>> interpreterFunction(const std::shared_ptr<Function>
 &function,
                                                                    const
 std::vector<std::vector<std::uint8_t>> &inputs,
                                                                    element::Type_t convertType) {
             runtime::Backend::set_backend_shared_library_search_directory("");
             auto backend = runtime::Backend::create("INTERPRETER");

             const auto &parameters = function->get_parameters();
             const auto &parametersNumber = parameters.size();
             const auto &inputsNumber = inputs.size();
             NGRAPH_CHECK(parametersNumber == inputsNumber,
                          "Got function (", function->get_friendly_name(), ") with ",
 parametersNumber, " parameters, but ",
                          inputsNumber, " input blobs");

             auto inputTensors = std::vector<std::shared_ptr<runtime::Tensor>>{};
             for (const auto &parameter : parameters) {
                 const auto &parameterIndex = function->get_parameter_index(parameter);
                 const auto &parameterShape = parameter->get_shape();
                 const auto &parameterType = parameter->get_element_type();
                 const auto &parameterSize = shape_size(parameterShape) * parameterType.size();

                 const auto &input = inputs[parameterIndex];
                 const auto &inputSize = input.size();
                 NGRAPH_CHECK(parameterSize == inputSize,
                              "Got parameter (", parameter->get_friendly_name(), ") of size ",
 parameterSize,
                              " bytes, but corresponding input with index ", parameterIndex,
                              " has ", inputSize, " bytes");

                 auto tensor = backend->create_tensor(parameterType, parameterShape);
                 tensor->write(input.data(), parameterSize);
                 inputTensors.push_back(tensor);
             }

             auto outputTensors = std::vector<std::shared_ptr<runtime::Tensor>>{};
             const auto &results = function->get_results();
             for (size_t i = 0; i <results.size(); ++i) {
                 outputTensors.push_back(std::make_shared<HostTensor>());
             }

             auto handle = backend->compile(function);
             handle->call_with_validate(outputTensors, inputTensors);
             auto outputs = std::vector<std::vector<std::uint8_t>>(results.size());
             for (const auto &result : results) {
                 const auto &resultIndex = function->get_result_index(result);
                 auto &output = outputs[resultIndex];
                 output.resize(shape_size(result->get_shape()) * result->get_element_type().size());
                 outputTensors[resultIndex]->read(output.data(), output.size());
                 if (convertType != element::Type_t::undefined)
                     output = convertOutputPrecision(output, result->get_element_type(),
 convertType, shape_size(result->get_shape()));
             }

             return outputs;
         }
     }
 }
 */
