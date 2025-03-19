// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/tensor_iterator.hpp"

#include "backend.hpp"
#include "evaluate_node.hpp"

namespace ti_v0 {
ov::reference::custom_evaluate_function evaluate =
    [](const std::shared_ptr<ov::Model>& function, const ov::TensorVector& inputs, ov::TensorVector& outputs) -> void {
    const auto& parameters = function->get_parameters();
    const auto& parametersNumber = parameters.size();
    const auto& inputsNumber = inputs.size();
    OPENVINO_ASSERT(parametersNumber == inputsNumber,
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
        outputs.push_back(ov::Tensor(results[i]->output(0)));
    }

    auto backend = ov::runtime::Backend::create();
    auto handle = backend->compile(function);
    handle->call_with_validate(outputs, inputs);
};
}  // namespace ti_v0

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v0::TensorIterator>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
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
bool evaluate_node<ov::op::v0::TensorIterator>(std::shared_ptr<ov::Node> node,
                                               ov::TensorVector& outputs,
                                               const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v0::TensorIterator>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v0::TensorIterator>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v0::TensorIterator>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v0::TensorIterator>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v0::TensorIterator>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v0::TensorIterator>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v0::TensorIterator>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v0::TensorIterator>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v0::TensorIterator>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v0::TensorIterator>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v0::TensorIterator>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v0::TensorIterator>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v0::TensorIterator>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v0::TensorIterator>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v0::TensorIterator>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v0::TensorIterator>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
