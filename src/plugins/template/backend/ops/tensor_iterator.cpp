// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/tensor_iterator.hpp"

#include "backend.hpp"
#include "evaluates_map.hpp"
#include "openvino/op/tensor_iterator.hpp"
namespace ti_v0 {
ngraph::runtime::reference::custom_evaluate_function evaluate = [](const std::shared_ptr<ngraph::Function>& function,
                                                                   const ov::HostTensorVector& inputs,
                                                                   ov::HostTensorVector& outputs) -> void {
    const auto& parameters = function->get_parameters();
    const auto& parametersNumber = parameters.size();
    const auto& inputsNumber = inputs.size();
    NGRAPH_CHECK(parametersNumber == inputsNumber,
                 "Got function (",
                 function->get_friendly_name(),
                 ") with ",
                 parametersNumber,
                 " parameters, but ",
                 inputsNumber,
                 " input blobs");

    auto inputTensors = std::vector<std::shared_ptr<ngraph::runtime::Tensor>>{};
    for (const auto& parameter : parameters) {
        const auto& parameterIndex = function->get_parameter_index(parameter);
        const auto& parameterShape = parameter->get_shape();
        const auto& parameterType = parameter->get_element_type();
        const auto& parameterSize = ov::shape_size(parameterShape) * parameterType.size();

        const auto& input = inputs[parameterIndex];
        const auto& inputSize = input->get_size_in_bytes();
        NGRAPH_CHECK(parameterSize == inputSize,
                     "Got parameter (",
                     parameter->get_friendly_name(),
                     ") of size ",
                     parameterSize,
                     " bytes, but corresponding input with index ",
                     parameterIndex,
                     " has ",
                     inputSize,
                     " bytes");

        auto tensor = std::make_shared<ngraph::runtime::HostTensor>(parameterType, parameterShape);
        tensor->write(input->get_data_ptr(), parameterSize);
        inputTensors.push_back(tensor);
    }

    const auto& results = function->get_results();
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> outputTensors;
    outputTensors.reserve(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        outputTensors.push_back(std::make_shared<ov::HostTensor>());
    }
    auto backend = ngraph::runtime::Backend::create();
    auto handle = backend->compile(function);
    handle->call_with_validate(outputTensors, inputTensors);

    outputs.reserve(outputTensors.size());
    for (const auto& tensor : outputTensors) {
        auto host_tensor = std::static_pointer_cast<ngraph::runtime::HostTensor>(tensor);
        outputs.push_back(host_tensor);
    }
};
}  // namespace ti_v0

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v0::TensorIterator>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    ngraph::runtime::reference::tensor_iterator(op->get_num_iterations(),
                                                op->get_function(),
                                                op->get_output_descriptions(),
                                                op->get_input_descriptions(),
                                                outputs,
                                                inputs,
                                                ti_v0::evaluate);
    return true;
}
