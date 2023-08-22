// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/tensor_iterator.hpp"

#include "backend.hpp"
#include "evaluate_node.hpp"
#include "tensor_conversion_util.hpp"

namespace ti_v0 {
ov::reference::custom_evaluate_function evaluate = [](const std::shared_ptr<ngraph::Function>& function,
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
    ov::reference::tensor_iterator(op->get_num_iterations(),
                                   op->get_function(),
                                   op->get_output_descriptions(),
                                   op->get_input_descriptions(),
                                   outputs,
                                   inputs,
                                   ti_v0::evaluate);
    return true;
}

template <>
bool evaluate_node<ngraph::op::v0::TensorIterator>(std::shared_ptr<ngraph::Node> node,
                                                   const ngraph::HostTensorVector& outputs,
                                                   const ngraph::HostTensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ngraph::op::v1::Select>(node) || ov::is_type<ngraph::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ngraph::element::Type_t::boolean:
        return evaluate<ngraph::element::Type_t::boolean>(ov::as_type_ptr<ngraph::op::v0::TensorIterator>(node),
                                                          outputs,
                                                          inputs);
    case ngraph::element::Type_t::bf16:
        return evaluate<ngraph::element::Type_t::bf16>(ov::as_type_ptr<ngraph::op::v0::TensorIterator>(node),
                                                       outputs,
                                                       inputs);
    case ngraph::element::Type_t::f16:
        return evaluate<ngraph::element::Type_t::f16>(ov::as_type_ptr<ngraph::op::v0::TensorIterator>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::f64:
        return evaluate<ngraph::element::Type_t::f64>(ov::as_type_ptr<ngraph::op::v0::TensorIterator>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::f32:
        return evaluate<ngraph::element::Type_t::f32>(ov::as_type_ptr<ngraph::op::v0::TensorIterator>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i4:
        return evaluate<ngraph::element::Type_t::i4>(ov::as_type_ptr<ngraph::op::v0::TensorIterator>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::i8:
        return evaluate<ngraph::element::Type_t::i8>(ov::as_type_ptr<ngraph::op::v0::TensorIterator>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::i16:
        return evaluate<ngraph::element::Type_t::i16>(ov::as_type_ptr<ngraph::op::v0::TensorIterator>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i32:
        return evaluate<ngraph::element::Type_t::i32>(ov::as_type_ptr<ngraph::op::v0::TensorIterator>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i64:
        return evaluate<ngraph::element::Type_t::i64>(ov::as_type_ptr<ngraph::op::v0::TensorIterator>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u1:
        return evaluate<ngraph::element::Type_t::u1>(ov::as_type_ptr<ngraph::op::v0::TensorIterator>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u4:
        return evaluate<ngraph::element::Type_t::u4>(ov::as_type_ptr<ngraph::op::v0::TensorIterator>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u8:
        return evaluate<ngraph::element::Type_t::u8>(ov::as_type_ptr<ngraph::op::v0::TensorIterator>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u16:
        return evaluate<ngraph::element::Type_t::u16>(ov::as_type_ptr<ngraph::op::v0::TensorIterator>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u32:
        return evaluate<ngraph::element::Type_t::u32>(ov::as_type_ptr<ngraph::op::v0::TensorIterator>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u64:
        return evaluate<ngraph::element::Type_t::u64>(ov::as_type_ptr<ngraph::op::v0::TensorIterator>(node),
                                                      outputs,
                                                      inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}
