// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/detection_output.hpp"

#include "evaluate_node.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v0::DetectionOutput>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::referenceDetectionOutput<T> refDetOut(op->get_attrs(),
                                                                      op->get_input_shape(0),
                                                                      op->get_input_shape(1),
                                                                      op->get_input_shape(2),
                                                                      op->get_output_shape(0));
    if (op->get_input_size() == 3) {
        refDetOut.run(inputs[0]->get_data_ptr<const T>(),
                      inputs[1]->get_data_ptr<const T>(),
                      inputs[2]->get_data_ptr<const T>(),
                      nullptr,
                      nullptr,
                      outputs[0]->get_data_ptr<T>());
    } else if (op->get_input_size() == 5) {
        refDetOut.run(inputs[0]->get_data_ptr<const T>(),
                      inputs[1]->get_data_ptr<const T>(),
                      inputs[2]->get_data_ptr<const T>(),
                      inputs[3]->get_data_ptr<const T>(),
                      inputs[4]->get_data_ptr<const T>(),
                      outputs[0]->get_data_ptr<T>());
    } else {
        OPENVINO_THROW("DetectionOutput layer supports only 3 or 5 inputs");
    }
    return true;
}

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v8::DetectionOutput>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::referenceDetectionOutput<T> refDetOut(op->get_attrs(),
                                                                      op->get_input_shape(0),
                                                                      op->get_input_shape(1),
                                                                      op->get_input_shape(2),
                                                                      op->get_output_shape(0));
    if (op->get_input_size() == 3) {
        refDetOut.run(inputs[0]->get_data_ptr<const T>(),
                      inputs[1]->get_data_ptr<const T>(),
                      inputs[2]->get_data_ptr<const T>(),
                      nullptr,
                      nullptr,
                      outputs[0]->get_data_ptr<T>());
    } else if (op->get_input_size() == 5) {
        refDetOut.run(inputs[0]->get_data_ptr<const T>(),
                      inputs[1]->get_data_ptr<const T>(),
                      inputs[2]->get_data_ptr<const T>(),
                      inputs[3]->get_data_ptr<const T>(),
                      inputs[4]->get_data_ptr<const T>(),
                      outputs[0]->get_data_ptr<T>());
    } else {
        OPENVINO_THROW("DetectionOutput layer supports only 3 or 5 inputs");
    }
    return true;
}

template <>
bool evaluate_node<ngraph::op::v0::DetectionOutput>(std::shared_ptr<ngraph::Node> node,
                                                    const ngraph::HostTensorVector& outputs,
                                                    const ngraph::HostTensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ngraph::op::v1::Select>(node) || ov::is_type<ngraph::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ngraph::element::Type_t::boolean:
        return evaluate<ngraph::element::Type_t::boolean>(ov::as_type_ptr<ngraph::op::v0::DetectionOutput>(node),
                                                          outputs,
                                                          inputs);
    case ngraph::element::Type_t::bf16:
        return evaluate<ngraph::element::Type_t::bf16>(ov::as_type_ptr<ngraph::op::v0::DetectionOutput>(node),
                                                       outputs,
                                                       inputs);
    case ngraph::element::Type_t::f16:
        return evaluate<ngraph::element::Type_t::f16>(ov::as_type_ptr<ngraph::op::v0::DetectionOutput>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::f64:
        return evaluate<ngraph::element::Type_t::f64>(ov::as_type_ptr<ngraph::op::v0::DetectionOutput>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::f32:
        return evaluate<ngraph::element::Type_t::f32>(ov::as_type_ptr<ngraph::op::v0::DetectionOutput>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i4:
        return evaluate<ngraph::element::Type_t::i4>(ov::as_type_ptr<ngraph::op::v0::DetectionOutput>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::i8:
        return evaluate<ngraph::element::Type_t::i8>(ov::as_type_ptr<ngraph::op::v0::DetectionOutput>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::i16:
        return evaluate<ngraph::element::Type_t::i16>(ov::as_type_ptr<ngraph::op::v0::DetectionOutput>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i32:
        return evaluate<ngraph::element::Type_t::i32>(ov::as_type_ptr<ngraph::op::v0::DetectionOutput>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i64:
        return evaluate<ngraph::element::Type_t::i64>(ov::as_type_ptr<ngraph::op::v0::DetectionOutput>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u1:
        return evaluate<ngraph::element::Type_t::u1>(ov::as_type_ptr<ngraph::op::v0::DetectionOutput>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u4:
        return evaluate<ngraph::element::Type_t::u4>(ov::as_type_ptr<ngraph::op::v0::DetectionOutput>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u8:
        return evaluate<ngraph::element::Type_t::u8>(ov::as_type_ptr<ngraph::op::v0::DetectionOutput>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u16:
        return evaluate<ngraph::element::Type_t::u16>(ov::as_type_ptr<ngraph::op::v0::DetectionOutput>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u32:
        return evaluate<ngraph::element::Type_t::u32>(ov::as_type_ptr<ngraph::op::v0::DetectionOutput>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u64:
        return evaluate<ngraph::element::Type_t::u64>(ov::as_type_ptr<ngraph::op::v0::DetectionOutput>(node),
                                                      outputs,
                                                      inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}

template <>
bool evaluate_node<ngraph::op::v8::DetectionOutput>(std::shared_ptr<ngraph::Node> node,
                                                    const ngraph::HostTensorVector& outputs,
                                                    const ngraph::HostTensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ngraph::op::v1::Select>(node) || ov::is_type<ngraph::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ngraph::element::Type_t::boolean:
        return evaluate<ngraph::element::Type_t::boolean>(ov::as_type_ptr<ngraph::op::v8::DetectionOutput>(node),
                                                          outputs,
                                                          inputs);
    case ngraph::element::Type_t::bf16:
        return evaluate<ngraph::element::Type_t::bf16>(ov::as_type_ptr<ngraph::op::v8::DetectionOutput>(node),
                                                       outputs,
                                                       inputs);
    case ngraph::element::Type_t::f16:
        return evaluate<ngraph::element::Type_t::f16>(ov::as_type_ptr<ngraph::op::v8::DetectionOutput>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::f64:
        return evaluate<ngraph::element::Type_t::f64>(ov::as_type_ptr<ngraph::op::v8::DetectionOutput>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::f32:
        return evaluate<ngraph::element::Type_t::f32>(ov::as_type_ptr<ngraph::op::v8::DetectionOutput>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i4:
        return evaluate<ngraph::element::Type_t::i4>(ov::as_type_ptr<ngraph::op::v8::DetectionOutput>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::i8:
        return evaluate<ngraph::element::Type_t::i8>(ov::as_type_ptr<ngraph::op::v8::DetectionOutput>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::i16:
        return evaluate<ngraph::element::Type_t::i16>(ov::as_type_ptr<ngraph::op::v8::DetectionOutput>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i32:
        return evaluate<ngraph::element::Type_t::i32>(ov::as_type_ptr<ngraph::op::v8::DetectionOutput>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i64:
        return evaluate<ngraph::element::Type_t::i64>(ov::as_type_ptr<ngraph::op::v8::DetectionOutput>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u1:
        return evaluate<ngraph::element::Type_t::u1>(ov::as_type_ptr<ngraph::op::v8::DetectionOutput>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u4:
        return evaluate<ngraph::element::Type_t::u4>(ov::as_type_ptr<ngraph::op::v8::DetectionOutput>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u8:
        return evaluate<ngraph::element::Type_t::u8>(ov::as_type_ptr<ngraph::op::v8::DetectionOutput>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u16:
        return evaluate<ngraph::element::Type_t::u16>(ov::as_type_ptr<ngraph::op::v8::DetectionOutput>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u32:
        return evaluate<ngraph::element::Type_t::u32>(ov::as_type_ptr<ngraph::op::v8::DetectionOutput>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u64:
        return evaluate<ngraph::element::Type_t::u64>(ov::as_type_ptr<ngraph::op::v8::DetectionOutput>(node),
                                                      outputs,
                                                      inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}
