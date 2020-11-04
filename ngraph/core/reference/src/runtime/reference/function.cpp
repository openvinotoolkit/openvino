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

#include <cstring>
#include <ngraph/pass/visualize_tree.hpp>

#include "ngraph/opsets/opset5.hpp"
#include "ngraph/runtime/reference/function.hpp"

#include "ngraph/runtime/tensor.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/concat.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {

            bool call(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                      const std::vector<std::shared_ptr<runtime::Tensor>>& inputs,
                      const std::shared_ptr<ngraph::Function>& function)
            {
                // convert inputs to HostTensor
                std::vector<std::shared_ptr<HostTensor>> func_inputs;
                for (const auto& tensor : inputs)
                {
                    auto host_tensor = std::static_pointer_cast<runtime::HostTensor>(tensor);
                    func_inputs.push_back(host_tensor);
                }

                // convert outputs to HostTensor
                std::vector<std::shared_ptr<HostTensor>> func_outputs;
                for (const auto& tensor : outputs)
                {
                    auto host_tensor = std::static_pointer_cast<runtime::HostTensor>(tensor);
                    func_outputs.push_back(host_tensor);
                }

                // map function params -> HostTensor
                std::unordered_map<descriptor::Tensor*, std::shared_ptr<HostTensor>> tensor_map;
                size_t input_count = 0;
                for (const auto& param : function->get_parameters())
                {
                    for (size_t i = 0; i < param->get_output_size(); ++i)
                    {
                        descriptor::Tensor* tensor = &param->output(i).get_tensor();
                        tensor_map.insert({tensor, func_inputs[input_count++]});
                    }
                }

                // map function outputs -> HostTensor
                for (size_t output_count = 0; output_count < function->get_results().size(); ++output_count)
                {
                    auto output = function->get_results()[output_count];
                    if (!is_type<op::Result>(output))
                    {
                        throw ngraph_error("One of function's outputs isn't op::Result");
                    }
                    descriptor::Tensor* tensor = &output->get_output_tensor(0);
                    tensor_map.insert({tensor, func_outputs[output_count]});
                }

                ngraph::pass::VisualizeTree vt("loop_test.svg");
                vt.run_on_function(function);
                // for each ordered op in the graph
                for (const auto& op : function->get_ordered_ops())
                {
                    if (op::is_parameter(op))
                    {
                        continue;
                    }

                    // get op inputs from map
                    std::vector<std::shared_ptr<HostTensor>> op_inputs;
                    for (auto input : op->inputs())
                    {
                        descriptor::Tensor* tensor = &input.get_tensor();
                        op_inputs.push_back(tensor_map.at(tensor));
                    }

                    // get op outputs from map or create
                    std::vector<std::shared_ptr<HostTensor>> op_outputs;
                    for (size_t i = 0; i < op->get_output_size(); ++i)
                    {
                        descriptor::Tensor* tensor = &op->output(i).get_tensor();
                        std::shared_ptr<HostTensor> host_tensor;
                        auto it = tensor_map.find(tensor);
                        if (it == tensor_map.end())
                        {
                            host_tensor = std::make_shared<HostTensor>(op->output(i));
                            tensor_map.insert({tensor, host_tensor});
                        }
                        else
                        {
                            host_tensor = it->second;
                        }
                        op_outputs.push_back(host_tensor);
                    }

                    if (!op->evaluate(op_outputs, op_inputs))
                    {
                        // exception
                        std::cout << "ERRROR!11" << std::endl;
                        // generate_calls(type, *op.get(), op_outputs, op_inputs);
                    }

                }
                return true;
            }

            std::shared_ptr<runtime::Tensor> create_tensor()
            {
                return std::make_shared<runtime::HostTensor>();
            }

            std::shared_ptr<runtime::Tensor>
            create_tensor(const element::Type& type, const Shape& shape)
            {
                return std::make_shared<runtime::HostTensor>(type, shape);
            }

            std::vector<std::vector<std::uint8_t>> function(const std::shared_ptr<ngraph::Function> &function, const std::vector<std::vector<std::uint8_t>> &inputs) {
                const auto &parameters = function->get_parameters();
                const auto &parametersNumber = parameters.size();
                const auto &inputsNumber = inputs.size();
                NGRAPH_CHECK(parametersNumber == inputsNumber,
                             "Got function (", function->get_friendly_name(), ") with ", parametersNumber, " parameters, but ",
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
                                 "Got parameter (", parameter->get_friendly_name(), ") of size ", parameterSize,
                                 " bytes, but corresponding input with index ", parameterIndex,
                                 " has ", inputSize, " bytes");

                    auto tensor = create_tensor(parameterType, parameterShape);
                    tensor->write(input.data(), parameterSize);
                    inputTensors.push_back(tensor);
                }

                const auto &results = function->get_results();
                std::vector<std::shared_ptr<ngraph::runtime::Tensor>> outputTensors;
                for (size_t i = 0; i < results.size(); ++i) {
                    outputTensors.push_back(std::make_shared<HostTensor>());
                }
                call(outputTensors, inputTensors, function);
                std::vector<std::vector<std::uint8_t>> outputs(results.size());
                for (const auto &result : results) {
                    const auto &resultIndex = function->get_result_index(result);
                    auto &output = outputs[resultIndex];
                    output.resize(shape_size(result->get_shape()) * result->get_element_type().size());
                    outputTensors[resultIndex]->read(output.data(), output.size());
                }
                return outputs;
            }
        }
    }
}
