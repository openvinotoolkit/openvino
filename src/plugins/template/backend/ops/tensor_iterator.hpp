// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/tensor_iterator.hpp"
#include "tensor_conversion_util.hpp"

namespace ti_v0 {
ngraph::runtime::reference::custom_evaluate_function evaluate = [](const std::shared_ptr<ngraph::Function>& function,
                                                                   const ngraph::HostTensorVector& inputs,
                                                                   ngraph::HostTensorVector& outputs) -> void {
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

    const auto& results = function->get_results();
    outputs.reserve(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        outputs.push_back(std::make_shared<ngraph::HostTensor>(results[i]->output(0)));
    }

    auto backend = ov::runtime::Backend::create();
    auto handle = backend->compile(function);
    OPENVINO_SUPPRESS_DEPRECATED_START
    auto outputTensors = ov::util::wrap_tensors(outputs);
    auto inputTensors = ov::util::wrap_tensors(inputs);
    handle->call_with_validate(outputTensors, inputTensors);

    ov::util::update_output_host_tensors(outputs, outputTensors);
    OPENVINO_SUPPRESS_DEPRECATED_END
};
}  // namespace ti_v0

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v0::TensorIterator>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    ngraph::runtime::reference::tensor_iterator(op->get_num_iterations(),
                                                op->get_function(),
                                                op->get_output_descriptions(),
                                                op->get_input_descriptions(),
                                                outputs,
                                                inputs,
                                                ti_v0::evaluate);
    return true;
}