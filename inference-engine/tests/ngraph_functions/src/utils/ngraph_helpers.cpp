// Copyright (C) 2019 Intel Corporationconvert2OutputVector
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include <ngraph/opsets/opset1.hpp>

#include <ngraph_functions/utils/ngraph_helpers.hpp>

namespace ngraph {
namespace helpers {

ngraph::OutputVector convert2OutputVector(const std::vector<std::shared_ptr<ngraph::Node>> &nodes) {
    ngraph::OutputVector outs;
    std::for_each(nodes.begin(), nodes.end(), [&outs](const std::shared_ptr<ngraph::Node> &n) {
        for (const auto &out_p : n->outputs()) {
            outs.push_back(out_p);
        }
    });
    return outs;
}

std::vector<std::vector<std::uint8_t>> interpreterFunction(const std::shared_ptr<Function>& function, const std::vector<std::vector<std::uint8_t>>& inputs) {
    ngraph::runtime::Backend::set_backend_shared_library_search_directory("");
    ngraph_register_interpreter_backend();
    auto backend = ngraph::runtime::Backend::create("INTERPRETER");

    const auto& parameters = function->get_parameters();
    const auto& parametersNumber = parameters.size();
    const auto& inputsNumber = inputs.size();
    NGRAPH_CHECK(parametersNumber == inputsNumber,
        "Got function (", function->get_friendly_name(), ") with ", parametersNumber, " parameters, but ", inputsNumber, " input blobs");

    auto inputTensors = std::vector<std::shared_ptr<runtime::Tensor>>{};
    for (const auto& parameter : parameters) {
        const auto& parameterIndex = function->get_parameter_index(parameter);
        const auto& parameterShape = parameter->get_shape();
        const auto& parameterType  = parameter->get_element_type();
        const auto& parameterSize  = ngraph::shape_size(parameterShape) * parameterType.size();

        const auto& input = inputs[parameterIndex];
        const auto& inputSize = input.size();
        NGRAPH_CHECK(parameterSize == inputSize,
            "Got parameter (", parameter->get_friendly_name(), ") of size ", parameterSize, " bytes, but corresponding input with index ", parameterIndex,
            " has ", inputSize, " bytes");

        auto tensor = backend->create_tensor(parameterType, parameterShape);
        tensor->write(input.data(), parameterSize);
        inputTensors.push_back(tensor);
    }

    auto outputTensors = std::vector<std::shared_ptr<runtime::Tensor>>{};
    const auto& results = function->get_results();
    std::transform(results.cbegin(), results.cend(), std::back_inserter(outputTensors), [&backend](const std::shared_ptr<op::Result>& result) {
        return backend->create_tensor(result->get_element_type(), result->get_shape()); });

    auto handle = backend->compile(function);
    handle->call_with_validate(outputTensors, inputTensors);
    auto outputs = std::vector<std::vector<std::uint8_t>>(results.size());
    for (const auto& result : results) {
        const auto& resultIndex = function->get_result_index(result);
        auto& output = outputs[resultIndex];
        output.resize(ngraph::shape_size(result->get_shape()) * result->get_element_type().size());
        outputTensors[resultIndex]->read(output.data(), output.size());
    }

    return outputs;
}

}  // namespace helpers
}  // namespace ngraph
